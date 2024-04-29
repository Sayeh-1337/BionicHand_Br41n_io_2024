# Copyright (C) 2023 Langlois Quentin, ICTEAM, UCLouvain. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
import pandas as pd

from copy import deepcopy

from utils.eeg_utils import get_event_samples, augment_dataset_windowing, split_channels, select_channel, normalize_dataset
from datasets import get_dataset, train_test_split

logger = logging.getLogger(__name__)

_dataset_summary_format = """
====================
Dataset information
====================

General :
  - sampling rate  : {rate}
  - # EEG channels : {n_channels}
{label_infos}{window_infos}
# Samples :
  - Train size    : {train_size}
  - Valid size    : {valid_size}
  - Test size     : {test_size}
  - valid == test : {val_is_test}

Subjects :
  - # subject(s) in train : {train_subj}
  - # subject(s) in valid : {valid_subj}
  - # subject(s) in test  : {test_subj}
  - # subject(s) in train and valid : {train_val_subj}
  - # subject(s) in train and test  : {train_test_subj}

Sessions :
  - # session(s) in train : {train_sess}
  - # session(s) in valid : {valid_sess}
  - # session(s) in test  : {test_sess}
  - # session(s) in train and valid : {train_val_sess}
  - # session(s) in train and test  : {train_test_sess}
"""

_simple_label_info  = "  - Labels (n = {n_train_labels}) : {train_labels}\n"
_per_set_label_info = """
Labels :
  - Label in train (n = {n_train_labels}) : {train_labels}
  - Label in valid (n = {n_valid_labels}) : {valid_labels}
  - Label in test  (n = {n_test_labels}) : {test_labels}
  - # label(s) in train and valid : {n_train_val_labels}
  - # label(s) in train and test  : {n_train_test_labels}
"""

_windowing_info = """
Data augmentation (windowing) :
  - # windows / sample in train set : {train_windows:.3f}
  - # windows / sample in valid set : {valid_windows:.3f}
"""

def get_experimental_data(config, ** kwargs):
    config = deepcopy(config)
    train, valid, test = _get_experimental_data(
        ** config['dataset_config'], config = config, ** kwargs, keep_passive = 'passive' in config['model_name']
    )

    if config['model_config']['rate'] is None:
        rate = list(set(train['rate'].unique().tolist()).union(valid['rate'].unique().tolist()))
        config['model_config']['rate'] = rate[0] if len(rate) == 1 else -1

    if 'channels' not in config['model_config']:
        channels = 1 if config['dataset_config']['split_channels'] else train.iloc[0]['channels']
        config['model_config']['channels'] = channels

    config['model_config']['labels'] = sorted(train['label'].unique().tolist())
    return train, valid, test, config

def _get_experimental_data(dataset,

                           channel        = None,
                           subjects       = None,
                           keep_artifacts = True,
                           keep_passive   = False,
                           per_user_label = False,

                           loso           = None,
                           test_sessions  = None,
                          
                           task         = 'classification',
                           test_task    = 'classification',
                           val_split    = None,
                           test_split   = 0.2,
                           stratify     = True,
                           random_state = None,

                           global_normalize = False,
                           
                           time_window    = 0,
                           offset         = 0,
                          
                           n_window       = 10,
                           window_len     = None,
                           window_step    = None,
                           n_train_window = None,
                           n_valid_window = None,
                          
                           ** kwargs
                          ):
    """
        Loads the dataset and makes the train-val-test split + possible data augmentation, based on the given configuration

        Arguments :
            - dataset    : the dataset name(s)
            # forwarded to `get_dataset`
            - subjects   : the (list of) subject id(s)
            - keep_artifacts : whether to keep events marked with "artifact"
            - keep_passive   : whether to keep "break" / "passive" events
            - per_user_label : whether to make each label subject-specific (e.g., subject-1-left hand, subject-2-left hand, ...)

            - loso           : (int) the number of subjects to remove from train/val, to use as test set
            - test_sessions  : (int) the number of sessions to remove from train/val, to use as test set

            - task           : (str) task name to prepare the data labels (see `prepare_dataset_for_training_task` for more info)
            - test_task      : (str) task for the test set (i.e., the training and test tasks may be different)
            - val_split      : (float) the fraction of the training data to use as validation data
                               `None` means to use the test set as validation data (**not recommanded for rigorous evaluation**)
            - test_split     : (fload) the fraction of the dataset to use as test set (if the dataset does not have a split)
            - random_state   : the seed to use for the train-val split (the train-test split has always the seed "0")
            # data pre-processing, forwarded to `get_event_samples`
            - time_window    : (int or float) the time window to keep from the event samples
            - offset         : (int or float) the time offset to skip from each event samples
            # data augmentation, forwarded to `augment_dataset_windowing`
            - n_window       : (int) the number of windows to use
            - window_len     : (int or float) the number of samples for each window
            - window_step    : (int or float) the number of samples for the stride
            - n_train_window : equivalent to `n_window` but specific to the train-set
            - n_valid_window : equivalent to `n_window` but specific to the valid set
        Return :
            - train / valid / test : `pd.DataFrame` representing the 3 sets
            - dataset_infos        : `dict` with the dataset information (notably `rate`, `channels`, `labels`)
    """
    if loso and test_sessions:
        raise RuntimeError('`loso = {}` is incompatible with `test_sessions = {}`'.format(loso, test_sessions))

    if subjects == 'all': subjects = None
    dataset      = get_dataset(
        dataset,
        subset = '5F',
        subjects       = 'B',
        keep_artifacts = True,
        keep_passive   = keep_passive,
        per_user_label = False
    )
    if isinstance(dataset, dict):
        if 'session' not in dataset['train'].columns:
            dataset['train']['session'] = 'session 1'
            dataset['valid']['session'] = 'session 2'
        if loso or test_sessions == 0:
            dataset = pd.concat(list(dataset.values()), axis = 0)
    
    if isinstance(dataset, dict):
        train, test = dataset['train'], dataset['valid']
    elif test_sessions:
        train, test = train_test_split(
            dataset,
            valid_size      = test_sessions,
            split_by_unique = True,
            split_column    = 'session',
            random_state    = 0,
            shuffle         = True,
            labels          = dataset['label'].values
        )
    elif loso:
        if isinstance(loso, int):
            train, test = train_test_split(
                dataset,
                valid_size      = loso,
                split_by_unique = True,
                split_col       = 'id',
                random_state    = 0,
                shuffle         = True,
                labels          = dataset['label'].values
            )
        else:
            if not isinstance(loso, list): loso = [loso]
            loso = [s[1:] for s in loso]
            mask = dataset['id'].apply(lambda subj: any(subj.endswith(loso_subj) for loso_subj in loso))
            train, test = dataset[~mask], dataset[mask]
    else:
        train, test = train_test_split(
            dataset,
            valid_size      = test_split,
            random_state    = 0,
            shuffle         = True,
            labels          = dataset['label'].values
        )
    
    if val_split is None:
        logger.warning('`val_split is None`, which makes the validation and test sets equal ! Make sure that it is expected')
        valid = None
    else:
        train, valid = train_test_split(
            train,
            valid_size = val_split,
            labels     = train['label'].values if stratify else None,
            shuffle    = True,
            random_state    = random_state
        )

    if channel is not None:
        train = select_channel(train, channel)
        if valid is not None:
            valid = select_channel(valid, channel)
        test  = select_channel(test, channel)

    if window_len:
        if n_train_window is None: n_train_window = n_window
        if n_valid_window is None: n_valid_window = n_window
        train = augment_dataset_windowing(train, n_window = n_train_window, window_len = window_len, window_step = window_step)
        if valid is not None:
            valid = augment_dataset_windowing(valid, n_window = n_valid_window, window_len = window_len, window_step = window_step)
    else:
        train = get_event_samples(train, offset = offset, time_window = time_window)
        if valid is not None:
            valid = get_event_samples(valid, offset = offset, time_window = time_window)
    test  = get_event_samples(test, offset = offset, time_window = time_window)

    if global_normalize:
        if valid is None:
            train, test = normalize_dataset(train, test = test)
        else:
            train, valid, test = normalize_dataset(train, valid, test = test)

    if val_split is None: valid = test

    train = prepare_dataset_for_training_task(task, train, ** kwargs)
    valid = prepare_dataset_for_training_task(task, valid, ** kwargs)
    test  = prepare_dataset_for_training_task(test_task, test, is_test = True, ** kwargs)

    summarize_dataset(train, valid, test)

    return train, valid, test

def prepare_dataset_for_training_task(task, data, config, is_test = False, ** _):
    """
        Modifies the `data['label']` field to fit the given `task`

        Arguments :
            - task : the expected model task to perform
            - data : `pd.DataFrame` the data subset
        Return :
            - data : inplace updated version of `data`

        Supported tasks :
            - classification : simply return `data`
            - person-identification (pi) : `label` becomes `id` to classify the subjects
            - channel-person-identification (cpi) : `label` becomes `id` + channels are splitted
            - channel-person-label-identification (cpli) : `label` becomes `{id}-{label}` + channels are splitted
            - channel-sample-identification (csi) : `label` becomes the sample index + channels are splitted
            - channel-label-identification (cli)  : `label` is unchanged (i.e., the event name) + channels are splitted

        Except for "classification", a new `event` key is added to store the event name
        "labes are splitted" means that each channel become an individual data (see `split_channels` for more info)
    """
    if 'p' in task and len(data['id'].unique()) == 1: raise RuntimeError('Make sure to have multiple subjects for a person-specific task !')
    
    data['event'] = data['label']
    label_format, split = None, False
    if task == 'classification':
        pass
    elif task in ('pi', 'person-identification'):
        data['label'] = data['id']
    else:
        if task in ('si', 'sample-identification'):
            label_format = '{sample_id}'
        elif task in ('cpi', 'channel-person-identification'):
            label_format = '{id}-{channels[0]}'
        elif task in ('cli', 'channel-label-identification'):
            label_format = '{label}-{channels[0]}'
        elif task in ('cpli', 'channel-person-label-identification'):
            label_format = '{id}-{label}-{channels[0]}'
        else:
            raise ValueError('Unknown task : {}'.format(task))
        split = 'channels' in label_format

    if (not is_test and config) and (config['dataset_config'].get('split_channels', False) or split):
        config['dataset_config']['split_channels'] = True
        data = split_channels(data)

    if label_format:
        if 'sample_id' in label_format: data['sample_id'] = list(range(len(data)))
        data['label'] = data.apply(lambda row: label_format.format(** row), axis = 1)
        if 'sample_id' in label_format: data.pop('sample_id')
    
    return data

def summarize_dataset(train, valid, test):
    """ Displays a (big) message describing main information about the data subsets """
    train_labs, valid_labs, test_labs = set(train['label'].unique()), set(valid['label'].unique()), set(test['label'].unique())
    train_subj, valid_subj, test_subj = set(train['id'].values), set(valid['id'].values), set(test['id'].values)
    train_sess, valid_sess, test_sess = set([1]), set([1]), set([1])
    if 'session' in train: train_sess = set(train['session'].values)
    if 'session' in valid: valid_sess = set(valid['session'].values)
    if 'session' in test:  test_sess  = set(test['session'].values)
    
    label_format = _simple_label_info if train_labs == valid_labs == test_labs else _per_set_label_info
    label_infos  = {
        'train_labels' : train_labs if len(train_labs) < 5 else '{}, ...]'.format(str(list(train_labs)[:5])[:-1]),
        'valid_labels' : valid_labs if len(valid_labs) < 5 else '{}, ...]'.format(str(list(valid_labs)[:5])[:-1]),
        'test_labels'  : test_labs if len(test_labs) < 5 else '{}, ...]'.format(str(list(test_labs)[:5])[:-1]),
        'n_train_labels' : len(train_labs),
        'n_valid_labels' : len(valid_labs),
        'n_test_labels'  : len(test_labs),
        'n_train_val_labels'  : len(train_labs.intersection(valid_labs)),
        'n_train_test_labels' : len(train_labs.intersection(test_labs)),
    }

    window_format, window_infos = '', {}
    if 'window_index' in train.columns:
        window_format = _windowing_info
        window_infos  = {
            'train_windows' : np.mean([len(samp_data) for samp_idx, samp_data in train.groupby('original_sample')]),
            'valid_windows' : np.mean([len(samp_data) for samp_idx, samp_data in valid.groupby('original_sample')]) if 'original_sample' in valid.columns else 1
        }

    row = train.iloc[0]
    dataset_infos = {
        'rate'         : row['rate'],
        'n_channels'   : len(row['channels']),
        'label_infos'  : label_format.format(** label_infos),
        'window_infos' : window_format.format(** window_infos),
        
        'train_size'  : len(train),
        'valid_size'  : len(valid),
        'test_size'   : len(test),
        'val_is_test' : valid is test,
        
        'train_subj' : len(train_subj),
        'valid_subj' : len(valid_subj),
        'test_subj'  : len(test_subj),
        'train_val_subj'  : len(train_subj.intersection(valid_subj)),
        'train_test_subj' : len(train_subj.intersection(test_subj)),
        
        'train_sess' : len(train_sess),
        'valid_sess' : len(valid_sess),
        'test_sess'  : len(test_sess),
        'train_val_sess'  : len(train_sess.intersection(valid_sess)),
        'train_test_sess' : len(train_sess.intersection(test_sess)),
    }
    logger.info(_dataset_summary_format.format(** dataset_infos))
    return dataset_infos


