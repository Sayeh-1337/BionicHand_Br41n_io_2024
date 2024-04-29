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

import copy
import logging
import numpy as np
import pandas as pd

from utils import plot, plot_multiple

logger = logging.getLogger(__name__)

_default_comparator = lambda n1, n2, m1, m2: m2 - m1 if len(n2) > len(n1) or n2 > n1 else m1 - m2

def group_by_subject(results):
    """
        Groups the results for each model by subject

        Arguments :
            - results : `list` of `dict`, the list of config for each model (the result of `run_experiments`)
        Return :
            - groups   : `dict` of `dict`, where each nested dict corresponds to the results for a given subject
                {subject_id : model_results_for_subject}
                    model_results_for_subject = {model_name : model_config_for_subject}
                    
                    where model_config_for_subject is a copy of the original model config
                    where the `metrics` key only contains metrics specific to the given subject

                    For models trained on a single subject  : `test_accuracy` -> `accuracy`
                    For models trained on multiple subjects : `test_subj-x_accuracy` -> `accuracy`

        Note : models trained on multiple subjects will appear in multiple nested dict
               each time with an adapted config to only keep the metrics related to the specific subject
    """
    groups = {}
    for config in results.values():
        if any(v is None for v in config['metrics'].values()): continue
        for metric_name, metric in config['metrics'].items():
            parts = metric_name.split('_')
            if parts[1].startswith('subj'):
                metric_name, subj = '_'.join(parts[2:]), '-'.join(parts[1].split('-')[1:])
            else:
                metric_name, subj = '_'.join(parts[1:]), str(config['dataset_config']['subjects'])
            
            if config['model_name'] not in groups.get(subj, {}):
                subj_config = copy.deepcopy(config)
                subj_config['metrics'] = {}
                groups.setdefault(subj, {}).update({
                    config['model_name'] : subj_config
                })
            groups[subj][config['model_name']]['metrics'][metric_name] = metric
    
    return groups

def group_models(groups, ignore = 'run', skip = None):
    """
        Combine models by ignoring some parts of their config
        The default behavior (`ignore = 'run'`) will group all models with the same config except for the `run` entry

        The result will have the following structure :
            {
                'Subject {subj_id}' : {
                    model_metric_display_name : values
                    ...
                }
            }
        where `values` is the list of metrics for grouped models
    """
    if not isinstance(ignore, (list, tuple)): ignore = [ignore]
    if isinstance(skip, str): skip = [skip]
    
    should_skip = skip if callable(skip) else lambda name: any(s in name for s in skip) if skip else False
    
    ignore = list(ignore) + ['subj']
    
    results = {}
    for subj, models in groups.items():
        subj_key = 'Subject {}'.format(subj)
        results[subj_key] = {}
        for model, config in models.items():
            parts = [
                p for p in model.split('_')
                if not any(p.startswith(ign) for ign in ignore)
            ]
            for met, val in config['metrics'].items():
                parts_  = parts if '_' not in met else (parts + met.split('_')[:-1])
                display = ' '.join([parts[0]] + ['({})'.format(p_i) for p_i in parts_[1:]])
                if should_skip(display): continue
                results[subj_key].setdefault(display, []).append(val)
    
    return results

def compare_config_effect(model_groups, config_key = None, rename_model = None, comparator = _default_comparator):
    if rename_model is None:
        assert config_key
        if isinstance(config_key, str): config_key = [config_key]
        rename_model = lambda name: ' '.join([p for p in name.split() if not any(c in p for c in config_key)])
    
    # convert each list of metrics to numpy array for element-wise substraction
    model_groups = {
        subj : {m : np.array(v) for m, v in models.items()}
        for subj, models in model_groups.items()
    }
    
    comp = {}
    for subj, models in model_groups.items():
        _models = list(models.keys())
        renamed = [rename_model(model) for model in _models]
        for i, (model1, alias1) in enumerate(zip(_models, renamed)):
            for j, (model2, alias2) in enumerate(zip(_models[i + 1 :], renamed[i + 1:]), start = i):
                if alias1 == alias2:
                    m1, m2 = models[model1], models[model2]
                    if len(m1) != len(m2):
                        logger.warning('Models {} and {} do not have the same number of metrics ({} vs {})'.format(
                            model1, model2, len(m1), len(m2)
                        ))
                        n = min(len(m1), len(m2))
                        m1, m2 = m1[:n], m2[:n]
                    cmp = comparator(model1, model2, m1, m2)
                    comp.setdefault(subj, {}).update({alias1 : cmp})
    return comp

def _flatten_dict(data):
    if not isinstance(data, dict): return data
    flattened = {}
    for k, v in data.items():
        flattened.update(_flatten_dict(v))
    return flattened
