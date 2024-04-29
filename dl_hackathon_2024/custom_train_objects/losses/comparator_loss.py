
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

import tensorflow as tf

class ComparatorLoss(tf.keras.losses.Loss):
    def __init__(self, metric = 'binary_crossentropy', name = None, reduction = None, ** kwargs):
        super().__init__(name = name or metric, reduction = 'none', ** kwargs)
        
        self.metric = metric
    
    @property
    def metric_names(self):
        return '{}_loss'.format(self.metric)
    
    def call(self, y_true, y_pred):
        if self.metric == 'binary_crossentropy':
            y_true = tf.one_hot(y_true, tf.shape(y_pred)[1])
            loss = tf.keras.losses.binary_crossentropy(tf.reshape(y_true [-1]), tf.reshape(y_pred [-1]))
        else:
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits = False)
        
        return loss
    
    def get_config(self):
        config = super().get_config()
        config['metric'] = self.metric
        return config