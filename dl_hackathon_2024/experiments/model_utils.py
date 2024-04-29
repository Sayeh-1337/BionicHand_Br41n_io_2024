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

import json
import logging
import tensorflow as tf

from copy import deepcopy

from utils import var_from_str
from models import get_pretrained
from models.model_utils import is_model_name
from models.siamese import EEGEncoder
from models.classification import EEGClassifier

from utils.eeg_utils import EEGNormalization

logger = logging.getLogger(__name__)

def get_model_config(model_name, ** kwargs):
    """
        Returns a model-specific configuration based on its name

        Arguments :
            - model_name : the (formatted) model name. Format looks like "{key1}-{value1}_{key2}-{value2}"
            - kwargs     : additional config
        Return :
            - config     : `dict` with all configuration for buildig, training, compiling, ...
                Main keys are the following : 
                    - scenario   : (int) the scenario number
                    - model_name : the model name
                    - model_type : (str) model type (e.g., classifier, encoder, siamese, ...)
                    - pretrained : the model's pretrained name (i.e., for fine-tuning)
                    - skip_new   : whether to skip new models training or not (force as kwarg)
                    - scenario_config : empty `dict`, filled in later by `add_scenario_config`
                    - dataset_config  : `dict` with the supported kwargs of `get_experimental_data`
                    - model_config    : `dict` to initialize the model (`build_model`)
                    - compile_config  : `dict` to compile the model (`model.compile`)
                    - train_config    : `dict` to call the `train` method with an additional `callbacks_config`
                    - test_config     : `dict` to call the `evaluate_model` method, where the main key is `test_name`
    """
    parts  = model_name.split('_')

    if 'norm-no' in parts:
        normalization = EEGNormalization.NORMAL
    elif 'norm-mima' in parts:
        normalization = EEGNormalization.MIN_MAX
    elif 'orig' in parts or 'norm-gno' in parts:
        normalization = EEGNormalization.GlobalNormal
    elif 'norm-sess' in parts:
        normalization = None
    else:
        normalization = EEGNormalization.NORMAL

    batch_size = get_model_info(parts, 'bs', default = kwargs.get('batch_size', None))
    
    model_type, model_config, compile_config, training_config = 'classifier', {}, {}, {}
    if 'ge2e' in model_name:
        model_type = 'encoder'
        
        n_ut = get_model_info(parts, 'nut', default = 16)
        if batch_size is None: batch_size = n_ut * get_model_info(parts, 'nlab', default = 16)
        n_lab = batch_size // n_ut
        training_config.update({
            'generator_config' : {
                'n_round'        : get_model_info(parts, 'round', default = 100),
                'batch_size'     : batch_size,
                'n_utterance'    : n_ut,
                'min_round_size' : n_lab
            }
        })
        model_config.update({
            'distance_metric' : get_model_info(parts, 'metric', default = 'cosine'),
            'embedding_dim'   : get_model_info(parts, 'dim', default = 32),
            'final_activation' : get_model_info(parts, 'act', default = None)
        })
        compile_config.setdefault('loss_config', {}).update({'mode' : get_model_info(parts, 'ge2e', val_if_empty = 'softmax')})
    elif 'siamese' in model_name:
        model_type = 'siamese'
        raise NotImplementedError()

    if parts[0].startswith('singlechannelcombination'):
        if '-' in parts[0]:
            model_config['backbone']          = parts[0].split('-')[1]
        model_config['channel_embedding_dim'] = get_model_info(parts, 'cdim', default = 8)
        model_config['channel_drop_rate']     = get_model_info(parts, 'cdrop', default = 25) / 100.
    
    win_length = get_model_info(parts, 'winlen', default = 4.5)

    subjects = get_model_info(parts, 'subj', default = None, val_if_empty = None)
    if subjects in ('all', '{subject}'): subjects = None

    normalize_config = {
        'normalize' : normalization, 'per_channel' : get_model_info(parts, 'perchan', default = True)
    } if normalization is not None else {}
    if 'clean' in parts:
        normalize_config.update({
            'detrend' : True, 'remove_environmental_noise' : True, 'filtering_freqs' : [3, 45]
        })
    
    config = {
        'scenario'        : get_model_info(parts, 'scenario', default = 3),
        'model_name'      : model_name,
        'model_type'      : model_type,
        'pretrained'      : None if 'ft' not in parts else model_name.replace('_ft_', ''),
        'skip_new'        : kwargs.get('skip_new', False),
        'overwrite'       : kwargs.get('overwrite', False),
        'scenario_config' : {},
        'dataset_config'  : {
            'task'           : get_model_info(parts, 'task',  default = 'classification'),
            'test_task'      : get_model_info(parts, 'ttask', default = 'classification'),
            'dataset'        : get_model_info(parts, 'ds', default = 'bci-iv_2a'),
            'subjects'       : kwargs.get('subjects', subjects),
            'channel'        : get_model_info(parts, 'ch', default = None),
            'loso'           : get_model_info(parts, 'loso', default = 0, val_if_empty = 1),
            'val_split'      : None if 'orig' in parts else 0.2,
            'stratify'       : False if 'nostrat' in parts else True,
            'global_normalize' : False if 'normalize' in normalize_config else True,
            'test_split'     : get_model_info(parts, 'testsplit', default = 20) / 100,
            'split_channels' : get_model_info(parts, 'split', default = False),
            'per_user_label' : 'lss' in parts
        },
        'model_config'    : {
            'nom'                      : model_name,
            'rate'                     : get_model_info(parts, 'rate', default = None),
            'use_ea'                   : get_model_info(parts, 'ea', default = False),
            'architecture_name'        : parts[0].split('-')[0],
            'normalization_config'     : normalize_config,
            'keep_spatial_information' : 'auto',
            'max_input_length'         : win_length,
            'use_fixed_length_input'   : True if win_length else False,
            'multi_subjects'           : subjects is None or isinstance(subjects, (list, tuple, range)),
            ** model_config,
            ** kwargs.get('model_config', {})
        },
        'compile_config'  : {
            'optimizer'        : get_model_info(parts, 'optim', kwargs.get('optimizer', 'adam')),
            'optimizer_config' : {'lr' : kwargs.get('lr', 1e-3)},
            'reduction'        : 'auto' if 'fit' in parts else 'none',
            ** compile_config,
            ** kwargs.get('compile_config', {})
        },
        'train_config'    : {
            'epochs'           : 1000 if 'orig' in parts else kwargs.get('epochs', 500 if 'nostop' in parts else None),
            'batch_size'       : batch_size if batch_size else 64,
            'mixup'            : any(p.startswith('mixup') for p in parts),
            'mixup_prct'       : get_model_info(parts, 'mixup', default = 0., val_if_empty = 1.),
            'callbacks_config' : {
                'monitor'    : 'val_accuracy' if 'orig' in parts else 'val_loss',
                'patience'   : 300 if 'orig' in parts else None,
                'use_early_stopping' : False if 'nostop' in parts else True
            },
            ** training_config,
            ** kwargs.get('training_config', {})
        },
        'test_config'     : {
            'test_name'      : {'test' : {}}
        }
    }
    if 'nc' in parts: config['model_config']['channels'] = None

    if config['dataset_config']['per_user_label']:
        config['test_config']['test_name']['test_masked'] = {'mask_by_id' : True}

    if config['dataset_config']['loso']:
        config['test_config']['test_name']['test_offline'] = {'add_loso_samples' : True}
    
    if config['model_config']['use_fixed_length_input']:
        config['dataset_config'].update({
            'time_window' : win_length,
            'offset'      : get_model_info(parts, 'winoff', default = 1.5, val_if_empty = 1.5)
        })

    # data augmentation related config (windowing)
    if 'win' in parts or any(p.startswith('win-') for p in parts):
        config['dataset_config'].update({
            'window_len' : win_length,
            'n_window'   : int(get_model_info(parts, 'win', val_if_empty = 10))
        })
    
    return config
    
def build_model(model_name, config = None, ** kwargs):
    """
        Initializes and compile the model based on the given `config` and `dataset_infos`

        Arguments :
            - model_name    : used to call `get_model_config` if `config` is not provided
            - dataset_infos : `dict` containing dataset-specific information (notably `rate`, `channels` and `labels`)
            - config        : config returned by `get_model_config(model_name, ** kwargs)`
            - kwargs        : forwarded to `get_model_config` if `config` is not provided
        Return :
            - model         : the initialized model
            - config        : an updated (new) version of `config`
    """
    if config is None: config = get_model_config(model_name, ** kwargs)
    else:              config = deepcopy(config)

    if is_model_name(model_name):
        model = get_pretrained(model_name)
    else:
        if config['model_type'] == 'classifier':
            model_class = EEGClassifier
        elif config['model_type'] == 'encoder':
            model_class = EEGEncoder
        elif config['model_type'] == 'siamese':
            raise NotImplementedError()

        logger.info('Building model with config : {}'.format(config['model_config']))
        if config['pretrained']:
            model = model_class.from_pretrained(config['pretrained'], ** config['model_config'])
        else:
            model = model_class(** config['model_config'])

    if model.epochs == 0:
        model.compile(** config['compile_config'])
    
    logger.info(model)
    
    return model, config

def get_training_callbacks(use_early_stopping = True, ** kwargs):
    """ Returns a list of `tf.keras.callbacks.Callback` for training """
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = kwargs.get('monitor', 'val_loss'),
            verbose  = kwargs.get('verbose', 0),
            factor   = kwargs.get('lr_factor', 0.9),
            patience = kwargs.get('lr_patience', 20),
            min_lr   = kwargs.get('lr', 1e-3) / 10
        ),
        tf.keras.callbacks.ModelCheckpoint(
            kwargs['filepath'],
            monitor = kwargs.get('monitor', 'val_loss'),
            verbose = kwargs.get('verbose', 1),
            save_best_only    = True,
            save_weights_only = True
        )
    ]
    if use_early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor  = kwargs.get('monitor', 'val_loss'),
            verbose  = kwargs.get('verbose', 1),
            patience = kwargs.get('patience', 250)
        ))
    return callbacks

def get_model_info(model_name, info, default = None, val_if_empty = True):
    """
        Return the value for the given `info` based on the formatted `model_name`

        Arguments :
            - model_name : (str) formatted model name or (list) the splitted model name
            - info       : (str) the key name in the model name format
            - default    : default value if the key is not present
            - val_if_empty : value if the key is provided without specific value
        Return :
            - value : the value (if provided in the formatted name), `default` otherwise, and `val_if_empty` if provided without value

        Example :
        ```python
        print(get_model_info('atcnet_subject-1_scenario-4', 'scenario', default = 3)) # displays "4"
        print(get_model_info('atcnet_subject-1_run-1', 'scenario', default = 3)) # displays "3"
        print(get_model_info('atcnet_win_subject-1_run-1', 'win', default = -1, val_if_empty = 10)) # displays "10"
        ```
    """
    if isinstance(model_name, list):
        parts = model_name
    else:
        if ' ' in model_name: model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        parts = model_name.split('_')
    if info in parts: return val_if_empty
    
    parts = [p for p in parts if p.startswith(info + '-')]
    if len(parts) == 0: return default
    elif len(parts) > 1: raise RuntimeError('Multiple parts match {} : {}'.format(info, parts))
    
    parts = parts[0].split('-')
    if len(parts) == 3 and not parts[1]: return var_from_str('-' + parts[2])
    return var_from_str(parts[1]) if len(parts) == 2 else [var_from_str(p) for p in parts[1:]]
    