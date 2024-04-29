
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

import os
import glob
import logging
import scipy.io
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from functools import wraps

from loggers import timer, time_logger
from datasets.custom_datasets import add_dataset

from utils.eeg_utils import read_eeg_events, get_event_samples

logger = logging.getLogger(__name__)

PASSIVE_LABELS = ['passive', 'break']

_eegmmidb_events = {
    0 : 'open eyes', 1 : 'close eyes', 2 : 'left hand', 3 : 'right hand',
    4 : 'left hand (IM)', 5 : 'right hand (IM)', 6 : 'both fists', 7 : 'both feet',
    8 : 'both fists (IM)', 9 : 'both feet (IM)', 10 : 'passive'
}
_eegmmidb_channels = [
    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
    'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
    'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ'
]

_bciiv_events = {
    '276' : 'passive eyes open', '277' : 'passive eyes closed', '768' : 'start trial', '769' : 'left hand (IM)',
    '770' : 'right hand (IM)', '771' : 'feet (IM)', '772' : 'tongue (IM)', '783' : 'unknown', '1023' : 'rejected',
    '1072' : 'passive eyes movement', '32766' : 'start run'
}
_bciiv_matlab_events = {i + 1 : _bciiv_events[idx] for i, idx in enumerate(['769', '770', '771', '772'])}
_bciiv_matlab_events[-1] = 'passive'

_large_scale_bci_5f_marker   = {
    0 : 'passive', 1 : 'thumb finger (IM)', 2 : 'index finger (IM)', 3 : 'middle finger (IM)',
    4 : 'ring finger (IM)', 5 : 'pinkie finger (IM)', 90 : 'break', 91 : 'break', 92 : 'experiment end', 99 : 'relaxation'
}
_large_scale_bci_5f_marker   = {
    0 : 'passive', 1 : 'Finger 1 (thumb)', 2 : 'Finger 2 (index)', 3 : 'Finger 3 (middle)',
    4 : 'Finger 4 (ring)', 5 : 'Finger 5 (pinkie)', 90 : 'passive', 91 : 'passive', 92 : 'experiment end', 99 : 'passive'
}
_large_scale_bci_cla_marker = {
    0 : 'passive', 1 : 'left hand (IM)', 2 : 'right hand (IM)', 3 : 'neutral', 4 : 'left leg (IM)',
    5 : 'tongue (IM)', 6 : 'right leg (IM)', 90 : 'break', 91 : 'break', 92 : 'experiment end', 99 : 'relaxation'
}



def eeg_dataset_wrapper(name, task, ** default_config):
    def wrapper(dataset_loader):
        """
            Wrapper for EEG datasets.
            
            The returned dataset is expected to be a `list` of `dict` (the events) with keys :
                - id       : the subject id (default to dataset name)
                - label    : the event name
                - start    : the EEG starting position (in sample) from the original EEG data
                - end      : the EEG ending position (in sample) from the original EEG data
                - time     : the EEG time (in sample)
                Optional :
                - eeg      : 2-D np.ndarray (the EEG signal)
                - channels : the position's name for the electrodes (list, same length as eeg)
                - rate     : the default EEG sampling rate
            The wrapper adds the keys :
                - n_channels    : equivalent to len(channels), the number of eeg channels (electrodes)
                - dataset_name  : the dataset name
        """
        @wraps(dataset_loader)
        @timer(name = '{} loading'.format(name))
        def _load_and_process(directory,
                              * args,
                              offset      = 0,
                              time_window = 0,
                              
                              per_user_session = True,
                              per_user_label   = False,
                              keep_artifacts   = True,
                              keep_passive     = True,
                              
                              ** kwargs
                             ):
            with time_logger.timer('data loading'):
                dataset = dataset_loader(directory, * args, ** kwargs)

            dataset = get_event_samples(dataset, offset = offset, time_window = time_window)
            
            if not keep_passive:
                dataset = [data for data in dataset if not any(
                    passive_label in data['label' if 'label' in data else 'event_name']for passive_label in PASSIVE_LABELS
                )]
            
            if not keep_artifacts:
                dataset = [data for data in dataset if not data.get('has_artifact', False)]
            
            with time_logger.timer('update events'):
                for event in dataset:
                    if 'pos' in event:       event['channels'] = event.pop('pos')
                    if 'label' not in event: event['label'] = event['event_name']
                    event.update({
                        'id'           : '{}-{}'.format(name, event['id']) if 'id' in event else name,
                        'time'         : event['end'] - event['start'],
                        'n_channels'   : len(event['channels']),
                        'dataset_name' : name
                    })
                    if 'session' not in event and event.get('meas_date', None):
                        event['session'] = event['meas_date']
                    if per_user_label:
                        event['label'] = '{}-{}'.format(event['id'], event['label'])
                    if per_user_session and 'session' in event:
                        event['session'] = '{}-{}'.format(event['id'], event['session'])

            with time_logger.timer('dataframe creation'):
                dataset = pd.DataFrame(dataset)
            
            return dataset
        
        add_dataset(name, processing_fn = _load_and_process, task = task, ** default_config)
        
        return _load_and_process
    return wrapper

@eeg_dataset_wrapper(
    name  = 'BCI-IV 2a', task = 'BCI',
    train = {'directory' : '{}/BCI-IV/dataset_2/a', 'subset' : 'train'},
    valid = {'directory' : '{}/BCI-IV/dataset_2/a', 'subset' : 'test'}
)
def preprocess_bciiv_2a_annots(directory, subset, subjects = None, overwrite = False, tqdm = lambda x: x, ** kwargs):
    if isinstance(subjects, (int, str)): subjects = [subjects]
    elif isinstance(subjects, range):    subjects = list(subjects)
    if subjects: subjects = [str(s) for s in subjects]
    
    _is_original = any(f.endswith('gdf') for f in os.listdir(directory))
    
    dataset = []
    for file in os.listdir(directory):
        if _is_original and not file.endswith('gdf'): continue
        elif subset == 'train' and 'T' not in file: continue
        elif subset == 'test' and 'T' in file: continue
        elif subjects and file[2] not in subjects: continue
        
        filename = os.path.join(directory, file)
        
        events = read_eeg_events(
            filename,
            add_infos   = True,
            add_signal  = True,
            skip_empty  = True,
            event_names = _bciiv_events if _is_original else _bciiv_matlab_events,
            target_channels = 22,
            ** kwargs
        )
        events = [ev for ev in events if ev['event_id'] != -1 and 'rejected' not in ev['event_name']]
        for ev in events:
            ev.update({'id' : file[2], 'session' : 'session 1' if 'T' in file else 'session 2'})
        
        if _is_original and subset == 'test':
            targets = scipy.io.loadmat(filename.replace('.gdf', '.mat'))['classlabel'][:, 0]
            idx = 0
            for ev in events:
                if ev['event_name'] == 'unknown':
                    ev.update({'event_id' : targets[idx], 'event_name' : _matlab_events[targets[idx]]})
                    idx += 1
        
        dataset.extend(events)
    
    return dataset

    
@eeg_dataset_wrapper(name = 'Brain MNIST', task = 'BCI', directory = '{}/Brain_MNIST')
def preprocess_brain_mnist_annots(directory, subset = None, ** kwargs):
    if isinstance(subset, str): subset = [subset]
    
    dataset = []
    for filename in glob.glob(os.path.join(directory, '*.txt')):
        if subset and not any(sub in filename for sub in subset): continue
            
        dataset.extend(read_eeg_events(
            filename,
            add_infos   = True,
            add_signal  = True,
            ** kwargs
        ))
    
    return dataset

for subset in ('IN', 'MU', 'MW', 'EP'):
    add_dataset(
        'MNIST {}'.format(subset), processing_fn = 'brain mnist', task = 'BCI', directory = '{}/Brain_MNIST', subset = subset
    )


@eeg_dataset_wrapper(name = 'EEGMMIDB', task = 'BCI', directory = '{}/EEGMMIDB')
def preprocess_eegmmidb_annots(directory, subjects = None, tqdm = lambda x: x, ** kwargs):
    if isinstance(subjects, (int, str)): subjects = [subjects]
    elif isinstance(subjects, range):    subjects = list(subjects)
    elif not subjects:                   subjects = list(range(1, 110))
    
    dataset = []
    for subj_id in tqdm(subjects):
        filename = os.path.join(directory, '{}.npy'.format(subj_id))
        
        events   = read_eeg_events(
            filename,
            add_signal = True,
            add_infos  = True,
            rate       = 160,
            channels   = _eegmmidb_channels,
            event_names  = _eegmmidb_events,
            ** kwargs
        )
        for ev in events: ev.update({'id' : str(subj_id), 'device' : 'BCI 2000'})

        dataset.extend(events)
    
    return dataset

@eeg_dataset_wrapper(
    name = 'CINC', task = 'Sleep analysis', train = {'directory' : '{}/cinc/train'}, valid = {'directory' : '{}/cinc/test'}
)
def preprocess_brain_mnist_annots(directory, extension = '.arousal', tqdm = lambda x: x, ** kwargs):
    dataset = []
    for subj_dir in tqdm(os.listdir(directory)):
        if subj_dir in ('train', 'test'): continue
        
        events = read_eeg_events(
            os.path.join(directory, subj_dir, subj_dir + extension),
            add_infos   = True,
            add_signal  = False,
            ** kwargs
        )
        for ev in events: ev['id'] = subj_dir
        
        dataset.extend(events)
    
    return dataset

@eeg_dataset_wrapper(name = 'Large scale BCI', task = 'BCI', directory = '{}/Large_scale_BCI')
def process_large_scale_bci_annots(directory, subset = None, subjects = None, tqdm = lambda x: x, ** kwargs):
    if subset is not None:
        if not isinstance(subset, (list, tuple)): subset = [subset]
        subset = [s.lower() for s in subset]
    
    dataset = []
    for file in tqdm(os.listdir(directory)):
        if 'HFREQ' in file: continue
        if not file.endswith(('.mat', '.fif')): continue

        if '-' in file:
            task, subj_id, date = file.split('-')[:3]
        else:
            subj_idx, n = file.index('Subject'), len('Subject')
            task, subj_id, date = file[:subj_idx], file[subj_idx : subj_idx + n + 1], file[subj_idx + n + 1 : subj_idx + n + 7]
        subj_id, task = subj_id[-1:], task.lower()
        
        if subset and task not in subset: continue
        if subjects and subj_id not in subjects: continue
        
        filename = os.path.join(directory, file)
        
        events = read_eeg_events(
            filename,
            add_signal = True,
            add_infos  = True,
            stop_label = 92,
            event_names = _large_scale_bci_5f_marker if task == '5f' else _large_scale_bci_cla_marker,
            target_rate = 200
        )
        for ev in events:
            ev.update({'id' : subj_id, 'task' : task, 'device' : 'EEG 1200', 'meas_date' : date})
        
        dataset.extend(events)
    
    return dataset

@eeg_dataset_wrapper(name = 'SEED', task = 'EEG emotion recognition', directory = '{}/SEED-V')
def process_seed_annots(directory, subset = None, subjects = None, tqdm = lambda x: x,
                        ** kwargs):
    import mne
    
    _session_timestamps = {
        1 : [
            [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
            [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]
        ],
        2 : [
            [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
            [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]
        ],
        3 : [
            [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
            [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
        ]
    }
    _labels = {0 : 'disgust', 1 : 'fear', 2 : 'sad', 3 : 'neutral', 4 : 'happy'}
    
    session_label   = load_data(os.path.join(directory, 'emotion_label_and_stimuli_order.xlsx'))
    session_label   = session_label.iloc[-3:, 2:].values
    
    if subset is not None and not isinstance(subset, (list, tuple)): subset = [subset]
    if subjects is not None and not isinstance(subjects, (list, tuple)): subjects = [subjects]

    rate        = 1000

    dataset = []
    for file in tqdm(os.listdir(os.path.join(directory, 'EEG_raw'))):
        if not file.endswith('.cnt'): continue

        subj_id     = int(file.split('_')[0])
        session     = int(file.split('_')[1])
        
        if subset and session not in subset: continue
        if subjects and subj_id not in subjects: continue
        
        eeg_file    = mne.io.read_raw_cnt('D:/datasets/SEED-V/EEG_raw/6_3_20180802.cnt')
        eeg         = eeg_file.get_data()

        for i, (start, end) in enumerate(zip(* _session_timestamps[session])):
            dataset.append({
                'id'    : subj_id,
                'label' : session_label[session - 1, i],
                'rate'  : rate,
                'size'  : (end - start) * rate,
                'time'  : end - start,
                'pos'   : eeg_file.ch_names,
                'eeg'   : eeg[:, start * rate : end * rate]
            })
    
    return pd.DataFrame(dataset)
