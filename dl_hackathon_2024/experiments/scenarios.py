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
import re
import json
import time
import random
import logging
import warnings
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from models import get_model_history
from models.model_utils import is_model_name
from utils import limit_gpu_memory, partial, to_json, load_json, dump_json

from .data_utils import get_experimental_data
from .multiproc_utils import run_in_processes
from .model_utils import get_model_config, build_model, get_training_callbacks
from .scenarios_utils import (
    _flatten, InvalidScenarioException, format_model_name, add_scenario_config, validate_scenario_config, validate_scenario_data
)
from .masked_accuracy import MaskedAccuracy

logger = logging.getLogger(__name__)

_evaluation_dir       = 'evaluations'

_keys_to_group = {'id' : 'subj', 'session' : 'sess', 'dataset_name' : 'ds'}

WAIT_TIME_BEFORE_INIT = 5

def run_experiments(model_name, max_workers = 4, tqdm = tqdm, ** kwargs):
    """
        Runs a serie of experiments using multiprocessing to run multiple runs in parallel (at most `ma_workers`)

        General procedure :
            1) Calls `format_model_name` on each `model_name` to get all formatted model names
            2) Calls `_get_scenario_config` on each formatted model name, which internally calls :
                2.1) `get_model_config`         : to get general model config
                2.2) `add_scenario_config`      : to add the scenario-specific config
                2.3) `validate_scenario_config` : to ensure the configuration consistency and skip irrelevant experiments
            3) Calls `run_scenario` on each configuration (either in multiprocess if `max_workers > 0`, either sequentially)
            4) Checks that all experiments have been successfully executed and collects the results

        Arguments :
            - model_name : (list of) model names
                           - str, the model name
                           - dict, model-specific configuration with a "model_name" key
            - max_workers : number of experiments to run in parallel
            - kwargs      : forwarded to `format_model_name` and `_get_scenario_config` (and thus `get_model_config`)
        Return :
            - results     : a `dict` `{model_name : results}` where `results` is the same as `get_model_config`
                            with the "metrics" entry filled-in (if not "skip_new")
    """
    if not isinstance(model_name, list): model_name = [model_name]

    experiments = []
    for name in model_name:
        if isinstance(name, dict):
            name, config = name.pop('model_name'), {** kwargs, ** name}
        else:
            name, config = name, kwargs
        
        experiments.append(format_model_name(name, ** config))
    experiments = _flatten(experiments)

    experiments_config = []
    for name in experiments:
        try:
            experiments_config.append(_get_scenario_config(name, ** kwargs))
        except InvalidScenarioException as e:
            logger.info('Skipping experiment name {} due to {}'.format(name, e))

    if max_workers > 1 and len(experiments_config) > 1:
        lock = multiprocessing.Lock()
        for config in experiments_config: config.update({'init_lock' : lock, 'init_wait' : WAIT_TIME_BEFORE_INIT})
        results = run_in_processes(
            run_scenario, experiments_config, max_workers = max_workers, tqdm = tqdm
        )
    else:
        results = [run_scenario(config) for config in tqdm(experiments_config)]

    final_results = {}
    for config, res in zip(experiments_config, results):
        if isinstance(res, InvalidScenarioException):
            logger.info('Scenario {} has been interrupted due to {}\nMake sure that your dataset is compatible with the given scenario'.format(config['model_name'], res))
        elif isinstance(res, Exception):
            warnings.warn('Scenario {} has been interrupted due to an inconsistency : {}'.format(
                config['model_name'], res
            ))
        else:
            final_results[config['model_name']] = res

    return final_results

def run_scenario(config, ** kwargs):
    """
        Runs an experiment based on the given `config` (if `config` is a model name, calls `_get_scenario_config`)

        General scenario procedure :
            1) Calls `_get_metrics` to check whether the expected metrics were already computed and saved or not
            If metrics were not already computed and `config['skip_new'] is False` :
                2) Calls `_setup_gpu_config` to limit number of gpus / gpu memory visible for the experiment
                3) Calls `get_experimental_data` to get the train/val/test splits
                4) Calls `validate_scenario_data` to assess the consistency of the dataset for the given scenario
                5) Calls `build_model` to instanciate the model
                If model has not been traind yet (i.e., `model.epochs == 0`) :
                    6) Calls `validate_scenario_data` to get the training callabacks
                    7) Calls `model.train` to train the model based on the train/valid sets
                8) Calls `evaluate_model` to evaluate on the test set + save results
                9) Calls `_get_metrics` to populate the "metrics" entry of `config`
        
        Arguments :
            - config : `dict` the result of `_get_scenario_config`
            - kwargs : forwarded to `_get_scenario_config` if needed
        Return :
            - results : `dict`, deepcopy of `config` with the "metrics" entry filled-in
    """
    if isinstance(config, str): config = _get_scenario_config(config, ** kwargs)

    lock, wait = config.pop('init_lock', None), config.pop('init_wait', 0)

    name = config['model_name']
    config['metrics'] = _get_metrics(name, config = config)

    if config['overwrite'] or any(metric is None for name, metric in config['metrics'].items()):
        if config['skip_new']:
            logger.info('New model {} skipped'.format(config['model_name']))
            return config
        
        _setup_gpu_config(config, lock = lock, wait = wait)

        random_state = None if not isinstance(config['run'], int) else config['run']
        if random_state is not None:
            np.random.seed(random_state)
            tf.random.set_seed(random_state)

        train, valid, test, config = get_experimental_data(
            config, random_state = random_state
        )

        validate_scenario_data(config, train = train, valid = valid, test = test)

        model, config = build_model(name, config)

        filepath =  '{}/best_weights.keras'.format(model.save_dir)
        if model.epochs == 0 and not os.path.exists(filepath):
            if '_fit_' in model.nom:
                fit_model(model, config, train = train, valid = valid, filepath = filepath)
            else:
                train_model(model, config, train = train, valid = valid, filepath = filepath)

        metrics = evaluate_model(
            model = model, data = test, config = config, filepath = filepath, samples = train, overwrite = config['overwrite']
        )

        config['metrics'] = _get_metrics(name, config = config, metrics = metrics)
    return config

def fit_model(model, config, train, valid, filepath, verbose = False):
    from utils import plot
    
    #x_train, y_train = zip(* [model.encode_data(row) for _, row in train.iterrows()])
    #x_valid, y_valid = zip(* [model.encode_data(row) for _, row in valid.iterrows()])

    #x_train, x_valid = tf.cast(x_train, tf.float32), tf.cast(x_valid, tf.float32)
    #y_train, y_valid = tf.cast(y_train, tf.int32), tf.cast(y_valid, tf.int32)

    train_config = config['train_config'].copy()
    
    callbacks = get_training_callbacks(
        filepath = filepath, ** train_config.pop('callbacks_config', {})
    )

    train_config.update({'cache' : True, 'verbose' : verbose, 'test_size' : 0})
    hist = model.fit(
        train, validation_data = valid, callbacks = callbacks, ** train_config
    )
    model.plot_history()

    #plot(hist.history)

def train_model(model, config, train, valid, filepath, verbose = False):
    callbacks = get_training_callbacks(
        filepath = filepath, ** config['train_config'].get('callbacks_config', {})
    )

    train_config = config['train_config'].copy()
    train_config.pop('callbacks_config', None)
    train_config.update({'cache' : True, 'verbose' : verbose})
    hist = model.train(
        train, validation_data = valid, callbacks = callbacks, ** train_config
    )

    model.plot_history()

def evaluate_model(model,
                   config     = None,
                   data       = None,
                   samples    = None,
                   loso_samples     = None,
                   add_loso_samples = False,
                   
                   filepath   = None,
                   test_name  = None,
                   per_sample = False,
                   
                   overwrite  = False,
                   root       = _evaluation_dir,
                   save       = True,
                   ** kwargs
                  ):
    """
        Evaluates the model based on the given `config` and `test_name`

        Arguments :
            - model : the model or its name
            - config : `dict` returned by `_get_scenario_config`
            - data   : `pd.DataFrame`, the data to use for the evaluation
            - filepath : the filename of the model checkpoint to load before evaluation
            - test_name : the test(s) name (and config)
                - str   : single test name
                - list  : multiple test names without specific config
                - dict  : `{test_name : test_config}` pairs
            - per_sample : whether to return the prediction of each sample, or not 
            - overwrite  : whether to overwrite an already performed test
            - root       : where to save the predictions / metrics
            - save       : whether to save predictions / metrics or not
            - kwargs     : forwarded to `model.predict`
        Return :
            If `per_sample is True` :
                - list of dict, the predictions for each sample
            Otherwise :
                - dict `{metric_name : value}`

        The `model.predict` should accept a `pd.DataFrame` as argument, and return a `list` of `dict` with the following keys :
            - pred    : the predicted label name
            - score   : the (possibly unnormalized) score
            - prob    : the (normalized) probability score
            - pred_id : the predicted label id

        Note : for retro-compatibility, the method allows to load metrics from the model history

        **Warning** : the test set is typically **not** channel-splitted, meaning that if the model has been
                      trained on single-channels, the `model.predict` may have to handle multi-channel prediction
                      E.g., by making the prediction on each channel then making a majority vote over all predictions
    """
    if filepath and os.path.exists(filepath): model.load_weights(filepath)

    if not test_name: test_name = config['test_config'].get('test_name', 'test') if config else 'test'
    if isinstance(test_name, (list, dict)):
        if not isinstance(test_name, dict): test_name = {t : {} for t in test_name}
        all_metrics = {} if not per_sample else []
        for test, test_config in test_name.items():
            res = evaluate_model(
                model,
                config = config,
                data   = data,
                samples      = samples,
                loso_samples = loso_samples,
                
                test_name = test,
                
                save   = save,
                root   = root,
                overwrite = overwrite,
                ** {** kwargs, ** test_config}
            )
            if per_sample: all_metrics.extend(res)
            else:          all_metrics.update(res)
        return all_metrics

    name      = model if isinstance(model, str) else model.nom
    test_dir  = os.path.join(root, name, test_name)
    met_file  = os.path.join(test_dir, 'metrics.json')
    pred_file = os.path.join(test_dir, 'pred.json')
    if (data is not None) and (not os.path.exists(met_file) or overwrite):
        os.makedirs(test_dir, exist_ok = True)

        if add_loso_samples:
            if loso_samples is None:
                sessions = sorted(data['session'].unique())
                if len(sessions) == 1:
                    raise RuntimeError('When using an offline LOSO procedure, make sure to provide the `loso_samples`')

                logger.info('Using the 1st session of `data` to use as `loso_samples`')
                mask = data['session'] == sessions[0]
                loso_samples, data = data[mask], data[~mask]
                
            samples = pd.concat([samples, loso_samples])

        try:
            pred = model.predict(data, samples = samples, ** kwargs)
        except Exception as e:
            logger.error('Test `{}` for model `{}` has failed due to : {}'.format(test_name, name, e))
            raise e
            return {}
            
        for p, (_, d) in zip(pred, data.iterrows()):
            p.update({
                k : v for k, v in d.items() if k not in ('label_id', 'eeg') and v is not None
            })

        if save: dump_json(pred_file, pred)

        pred_df = pd.DataFrame(pred)
        
        metrics = {'{}_accuracy'.format(test_name) : np.mean(pred_df['label'].values == pred_df['pred'].values)}
        keys    = [k for k in _keys_to_group if len(pred_df[k].unique()) > 1]
        if keys:
            for values, group_data in pred_df.groupby(keys if len(keys) > 1 else keys[0]):
                if isinstance(values, str): values = [values]
                subj = group_data.iloc[0]['id']
                group = '_'.join([
                    '{}-{}'.format(_keys_to_group[k], v.replace(subj + '-', ''))
                    for k, v in zip(keys, values)
                ])
                metrics['{}_{}_accuracy'.format(test_name, group)] = np.mean(group_data['label'].values == group_data['pred'].values)
        
        if save: dump_json(met_file, metrics)
        logger.info('Model `{}` metrics : {}'.format(
            model.nom, metrics if len(metrics) == 2 else json.dumps(to_json(metrics), indent = 4)
        ))
        return metrics if not per_sample else pred
    elif per_sample:
        return load_json(pred_file, default = [])
    elif not os.path.exists(met_file):
        hist = get_model_history(name)
        if hist is None or len(hist.history) == 0: return {}
        metrics = {
            k if 'masked' not in k else k.replace('masked_', '').replace('test_', 'test_masked_') : v
            for k, v in hist.history[-1].items()
        }
        return {k : v for k, v in metrics.items() if k.startswith(test_name)}
    else:
        return load_json(met_file, default = {})

def _get_metrics(model_name, config = None, metric_names = None, metrics = None):
    """
        Return the metrics for the given model, config and metric_names

        Arguments :
            - model_name   : the model name (used for evaluation / metrics file loading)
                - list / dict : iterates over each name to fill-in the resulting dict
                - str         : simple metric name, that can contain a "*" to make regex-based matching
            - config       : `dict` returned by `_get_scenario_config` (used to evaluate the model / infer metric_names)
            - metric_names : list of expected metrics to return
            - metrics      : `dict` of `{metric_name : value}` pairs, typically returned by `evaluate_model`
        Return :
            - metrics      : `dict` of `{metric_name : value}` pairs (value is `None` if the metric has not been evaluated yet)
    """
    if not metric_names and config: metric_names = list(config['metrics'].keys())
    if metrics is None: metrics = evaluate_model(model_name, config)

    if not metric_names:
        return metrics
    elif isinstance(metric_names, (list, tuple, dict)):
        values = {}
        for n in metric_names: values.update(_get_metrics(model_name, config, n, metrics = metrics))
        return values
    elif '*' not in metric_names:
        return {metric_names : metrics.get(metric_names, None)}
    
    matches = {
        met : val for met, val in metrics.items() if re.search(metric_names.replace('*', '.*'), met) is not None
    }
    return matches if matches else {metric_names : None}

def _get_scenario_config(name, ** kwargs):
    """ Simply calls `validate_scenario_config(add_scenario_config(get_model_config(name, ** kwargs), ** kwargs))` """
    config = get_model_config(name, ** kwargs)
    config = add_scenario_config(config, ** kwargs)
    validate_scenario_config(config)
    return config

def _setup_gpu_config(config, lock = None, wait = 0, gpu = None):
    """ Setup the visible devices + available gpu memory for the given process """
    if lock is not None: lock.acquire()
    if wait > 0:         time.sleep(wait)
    
    if config.get('gpu', gpu) is not None:
        try:
            gpu_idx = config['gpu'] if isinstance(config['gpu'], (list, tuple)) else [config['gpu']]
            gpus    = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices([gpus[idx] for idx in gpu_idx], 'GPU')
        except Exception as e:
            if lock is not None: lock.release()
            logger.warning('Error while setting visible devices : {}'.format(e))
            return False

    if config.get('gpu_memory', None) is not None:
        limit_gpu_memory(config['gpu_memory'])

    tf.zeros((1, 1))

    if lock is not None: lock.release()
    
    return True

