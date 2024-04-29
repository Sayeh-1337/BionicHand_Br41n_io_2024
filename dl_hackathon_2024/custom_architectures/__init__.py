
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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
import json
import tensorflow as tf

from utils import get_object, print_objects, load_json
from custom_layers.custom_activations import _activations

def __load():
    for module_name in os.listdir('custom_architectures'):
        if module_name.startswith(('__', '.')): continue
        module_name = 'custom_architectures.' + module_name.replace('.py', '')

        module = __import__(
            module_name, fromlist = ['custom_objects', 'custom_functions']
        )
        if hasattr(module, 'custom_objects'):
            custom_objects.update(module.custom_objects)
        if hasattr(module, 'custom_functions'):
            _custom_architectures.update(module.custom_functions)

def get_architecture(architecture_name, *args, **kwargs):
    return get_object(architectures, architecture_name, *args, 
                      print_name = 'model architecture', err = True, **kwargs)

def print_architectures():
    print_objects(architectures, 'model architectures')
    
custom_objects = _activations.copy()
_custom_architectures = {}

__load()

_keras_architectures = {}

architectures = {**_keras_architectures, **_custom_architectures}