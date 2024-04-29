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

import tensorflow as tf

from utils.generic_utils import convert_to_str

def build_lookup_table(labels, default = -1):
    """
        Builds a `tf.lookup.StaticHashTable` based on `labels` and `mapping`
        
        Arguments :
            - labels  : list of labels (e.g., the labels of the model)
            - default : the default value if a label does not have any mapping
        Return :
            - table   : `tf.lookup.StaticHashTable` instance
    """
    labels = convert_to_str(labels)
    if not isinstance(labels, (list, tuple)): labels = [labels]
    
    keys, values = [], []
    for i, label in enumerate(labels):
        if not isinstance(label, (list, tuple)): label = [label]
        keys.extend(label)
        values.extend([i] * len(label))

    init  = tf.lookup.KeyValueTensorInitializer(tf.cast(keys, tf.string), tf.cast(values, tf.int32))
    table = tf.lookup.StaticHashTable(init, default_value = default)
    return table
