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

import numpy as np

from copy import deepcopy

from .model_utils import get_model_info

_scenarios = {
    1 : {'subjects' : 'single', 'sessions' : 'merged'},
    2 : {'subjects' : 'multi',  'sessions' : 'merged'},
    3 : {'subjects' : 'single', 'sessions' : 'bi'},
    4 : {'subjects' : 'multi',  'sessions' : 'bi'},
    5 : {'subjects' : 'loso'}
}

_default_scenario_config       = {
    'default'  : {'gpu' : None, 'gpu_memory' : None},
    'subjects' : {
        'single' : {
            'scenario_config' : {'train_on_multi_subjects' : False},
            'train_config'    : {'epochs' : 1000, 'callbacks_config' : {'patience' : 300}}
        },
        'multi'  : {
            'scenario_config' : {'train_on_multi_subjects' : True},
            'train_config'    : {'epochs' : 750,  'callbacks_config' : {'patience' : 100}}
        },
        'loso'   : {
            'scenario_config' : {'train_on_multi_subjects' : True},
            'train_config'    : {'epochs' : 750,  'callbacks_config' : {'patience' : 100}}
        }
    },
    'sessions' : {
        'merged' : {
            'dataset_config'  : {'test_sessions' : 0}
        },
        'bi'     : {
            'dataset_config'  : {'test_sessions' : 1}
        }
    },
    'tasks'    : {}
}

_is_single_subject  = lambda scenario: _scenarios[scenario]['subjects'] == 'single'
_is_multi_subjects  = lambda scenario: _scenarios[scenario]['subjects'] == 'multi'
_is_loso            = lambda scenario: _scenarios[scenario]['subjects'] == 'loso'

_is_merged_sessions = lambda scenario: _scenarios[scenario].get('sessions', None) == 'merged'
_is_bi_sessions     = lambda scenario: _scenarios[scenario].get('sessions', None) == 'bi'
_is_multi_sessions  = lambda scenario: _scenarios[scenario].get('sessions', None) == 'multi'

_is_loto            = lambda scenario: _scenarios[scenario].get('tasks', None) == 'loto'


class InvalidScenarioException(Exception):
    pass

def format_model_name(model_name, ** kwargs):
    """
        Return a (list of) formatted model names, with filled-in generic configuration
        Example : "atcnet_subject-{subject}_run-{run}_scenario-3" will iterate over each combination of "subject" and "run"

        Note that if the "{subject}" format is expected, and the scenario is a single-subject scenario,
        kwargs **must** contain the subject(s) to use
        If the scenario is a multi-subjects one, and the "subject" key is not provided, it is set to "all"
    """
    if isinstance(model_name, list):
        return [format_model_name(name, ** kwargs) for name in model_name]
    if '{' not in model_name: return model_name

    if '{loso}' in model_name:
        if not _is_loso(get_model_info(model_name, 'scenario')):
            raise RuntimeError('{loso} tag is not supported for non-loso scenario !')
        if 'loso' not in kwargs and not isinstance(kwargs.get('subject', None), (list, tuple, range)):
            raise RuntimeError('When using the {loso} format, you must explicitely provide the loso subject, or a list of `subject`')

        loso_subject = kwargs.pop('loso', kwargs.get('subject', None))
        if isinstance(loso_subject, (list, tuple, range)):
            return [
                format_model_name(model_name, loso = subj, ** kwargs)
                for subj in loso_subject
            ]
        
        if loso_subject in ('all', None):
            raise RuntimeError('`loso` subject cannot be "all" or None !')

        loso_subject = str(loso_subject)
        kwargs['loso'] = 's{}'.format(loso_subject) if not loso_subject.startswith('s') else loso_subject
        
    if '{run}' in model_name:
        kwargs.setdefault('run', None)
        if isinstance(kwargs.get('run', None), (list, tuple, range)):
            return [
                format_model_name(model_name, run = run, ** kwargs)
                for run in kwargs.pop('run')
            ]

    if '{channel}' in model_name and isinstance(kwargs['channel'], (list, range, tuple)):
        return [
            format_model_name(model_name, channel = chanel, ** kwargs) for chanel in kwargs.pop('channel')
        ]
        
    if '{subject}' in model_name:
        scenario = get_model_info(model_name, 'scenario')
        if 'subject' not in kwargs: kwargs['subject'] = 'all'

        if _is_single_subject(scenario):
            if kwargs['subject'] == 'all':
                raise RuntimeError(
                    'single-subject scnearios do not accept `subject = all`. Make sure to explicitely provide `subject`'
                )
            elif isinstance(kwargs['subject'], (list, tuple, range)):
                return [
                    format_model_name(model_name, subject = subj, ** kwargs)
                    for subj in kwargs.pop('subject')
                ]
        elif _is_multi_subjects(scenario) or _is_loso(scenario):
            if kwargs['subject'] != 'all' and not isinstance(kwargs['subject'], (list, tuple, range)):
                raise RuntimeError(
                    'multi-subjects scnearios expect multiple subjects but got {}'.format(kwargs['subject'])
                )
            elif isinstance(kwargs['subject'], (list, tuple, range)):
                kwargs['subject'] = '-'.join([str(s) for s in kwargs['subject']])

    return model_name.format(** kwargs)


def add_scenario_config(config, ** kwargs):
    """ Updates `config` (not inplace) based on the scenario-specific configuration specified in `_default_scenario_config` """
    scenario = config['scenario']
    config   = deepcopy(config)

    config.update({k : kwargs.get(k, v) for k, v in _default_scenario_config['default'].items()})
    config.update({
        'run'     : get_model_info(config['model_name'], 'run', default = None),
        'metrics' : {k : None for k in get_expected_metrics(config)}
    })

    for k in ('subjects', 'sessions', 'tasks'):
        config = _nested_update(
            config, _default_scenario_config.get(k, {}).get(_scenarios[scenario].get(k, None), {})
        )

    if config['model_type'] == 'encoder':
        config['train_config']['callbacks_config']['patience'] = min(
            75, config['train_config']['callbacks_config']['patience'] // 2
        )

    if config['dataset_config'].get('n_window', None):
        config['train_config']['callbacks_config']['patience'] = min(
            75, config['train_config']['callbacks_config']['patience'] // 2
        )

    return config

def validate_scenario_config(config):
    """ Raises an `InvalidScenarioException` if the scenario is invalid (i.e., some config do not fit the scenario specifications) """
    scenario = config['scenario']
    if scenario not in _scenarios:
        raise InvalidScenarioException('The scenario #{} is not implemented yet'.format(scenario))

    if config['model_config']['use_fixed_length_input'] and not config['model_config']['max_input_length']:
        raise RuntimeError('`max_input_length = {}` is incompatible with `use_fixed_length_input = True`'.format(
            config['model_config']['max_input_length']
        ))
    
    if _is_single_subject(scenario): # single-subject scenario
        if config['scenario_config']['train_on_multi_subjects']:
            raise RuntimeError('`train_on_multi_subjects` must be False for single-subject scenarios')
        if config['dataset_config']['per_user_label']:
            raise InvalidScenarioException('`per_user_label = True` is invalid for single-subject scenarios')
        if config['dataset_config']['loso']:
            raise InvalidScenarioException('`loso = {}` is invalid for single-subject scenarios'.format(
                config['dataset_config']['loso']
            ))
        
    elif _is_multi_subjects(scenario):
        if not config['scenario_config']['train_on_multi_subjects']:
            raise RuntimeError('`train_on_multi_subjects` must be True for multi-subjects scenarios')
        subjects = config['dataset_config'].get('subjects', None)
        if subjects is not None and not isinstance(subjects, (list, tuple, range)):
            raise InvalidScenarioException('`subject` ({}) must be an iterable for multi-subjects scenarios'.format(
                config['dataset_config']['subject']
            ))
    
    elif _is_loso(scenario):
        if not config['scenario_config']['train_on_multi_subjects']:
            raise RuntimeError('`train_on_multi_subjects` must be True for multi-subjects scenarios')
        if config['dataset_config']['loso'] == 0:
            raise RuntimeError('`loso` must not be "0" for loso scenario')

    if _is_merged_sessions(scenario):
        if config['dataset_config']['test_sessions']:
            raise InvalidScenarioException('`test_sessions = {}` is not supported for merged-sessions scenarios'.format(
                config['dataset_config']['test_sessions']
            ))
        
    elif _is_bi_sessions(scenario):
        if config['dataset_config']['test_sessions'] != 1:
            raise InvalidScenarioException('`test_sessions` must be equal to 1 for bi-sessions-scenarios')

def validate_scenario_data(config, train, valid, test):
    """ Checks that train / valid / test respects the given `config` and the given scenario specifications """
    scenario = config['scenario']
    
    train_labs, valid_labs, test_labs = set(train['label'].unique()), set(valid['label'].unique()), set(test['label'].unique())
    train_subj, valid_subj, test_subj = set(train['id'].values), set(valid['id'].values), set(test['id'].values)
    train_sess, valid_sess, test_sess = set([1]), set([1]), set([1])
    if 'session' in train: train_sess = set(train['session'].values)
    if 'session' in valid: valid_sess = set(valid['session'].values)
    if 'session' in test:  test_sess  = set(test['session'].values)

    for set_name, subset in zip(['train', 'val', 'test'], [train, valid, test]):
        if 'eeg' in subset.columns:
            if np.any(subset['eeg'].apply(lambda eeg: eeg.shape[1]).values != subset['time'].values):
                raise RuntimeError('EEG samples and expected time are inconsistent in {} set\n{}\n{}'.format(
                    set_name, config['eeg'].apply(lambda eeg: eeg.shape), config['time']
                ))
            
            n_channels = config['model_config']['channels']
            if (n_channels is not None) and (n_channels != 1 or set_name != 'test'):
                if isinstance(n_channels, list): n_channels = len(n_channels)
                if np.any(subset['eeg'].apply(len).values != n_channels):
                    raise RuntimeError('The number of channels in {} set does not match the expected number ({}) !\n{}'.format(
                        set_name, n_channels, config['eeg'].apply(lambda eeg: eeg.shape)
                    ))

        if config['model_config']['use_fixed_length_input']:
            expected_samples = config['model_config']['max_input_length']
            if isinstance(expected_samples, float):
                expected_samples = (expected_samples * subset['rate'].values).astype(np.int32)
            
            if np.any(subset['time'].values != expected_samples):
                raise RuntimeError('All data must have {} EEG samples, which is not the case in {} set\n{}'.format(
                    config['model_config']['max_input_length'], set_name, subset
                ))
    
    if _is_single_subject(scenario):
        if len(train_subj.union(valid_subj).union(test_subj)) != 1:
            raise InvalidScenarioException('The number of subject must be 1 for single-subject scenarios')

    elif _is_multi_subjects(scenario):
        if len(train_subj.union(valid_subj).union(test_subj)) <= 1:
            raise InvalidScenarioException('The number of subject must > 1 for multi-subjects scenarios')

        if any(t_subj not in train_subj for t_subj in test_subj):
            raise RuntimeError('In non-loso scenarios, all test subjects must be in the training set')
    
    elif _is_loso(scenario):
        if not config['dataset_config']['loso']:
            raise RuntimeError('`loso` must be provided for loso scenarios')
        
        if len(train_subj.intersection(test_subj)) != 0:
            raise RuntimeError('Some test subjects appear in the training set, which is forbidden in loso scenarios')

    if _is_merged_sessions(scenario):
        if train_sess != test_sess:
            raise RuntimeError('The training / test sessions must be equal in merged-sessions scenarios')

    elif _is_bi_sessions(scenario):
        if len(train_sess.intersection(test_sess)) != 0:
            raise RuntimeError('The test session appears in the training set, which is forbidden in bi-sessions scenarios')

    
    if config['model_type'] == 'classifier' and train_labs != test_labs:
        raise InvalidScenarioException('Classifier models do not support unknown test labels')

    elif _is_loto(scenario):
        if all(l in train_labs for l in test_labs):
            raise RuntimeError('All test labels are in train labels, which is unexpected for leave one task out (LOTO) scenarios !')


def get_expected_metrics(config):
    """
        Returns a list of metric names given the config
        These are typically {test_name}_*accuracy for each `test_name` in `config['test_config']['test_name']`

        Arguments :
            - config : a `dict` returned by `get_model_config`
        Return :
            - metric_names : a list of str
    """
    return [
        '{}_*accuracy'.format(prefix) for prefix in config.get('test_config', {}).get('test_name', ['test'])
    ]

def _flatten(data):
    """ Flatten a list of list into a single list """
    if not isinstance(data, list): return data
    flattened = []
    for d in data:
        d = _flatten(d)
        flattened.extend(d if isinstance(d, list) else [d])
    return flattened

def _flatten_dict(data):
    if not isinstance(data, dict): return data
    flattened = {}
    for k, v in data.items():
        v = _flatten_dict(v)
        flattened.update(v if isinstance(v, dict) else {k : v})
    return flattened

def _nested_update(config, update):
    for k, v in update.items():
        if isinstance(v, dict):
            _nested_update(config[k], v)
        elif config.get(k, None) is None:
            config[k] = v
    return config


        