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

class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self,
                 labels   = None,
                 n_labels = None,
                 labels_to_keep  = None,
                 indexes_to_keep = None,
                 balanced        = False,
                 name = 'masked_accuracy',
                 ** kwargs
                ):
        assert labels or n_labels
        assert (labels_to_keep and labels) or indexes_to_keep
        
        super().__init__(name = name)

        self.balanced        = balanced
        self.n_labels        = n_labels if n_labels else len(labels)
        self.indexes_to_keep = indexes_to_keep if indexes_to_keep else [
            i for i, label in enumerate(labels) if label in labels_to_keep
        ]

        self.mask = tf.tensor_scatter_nd_update(
            tf.zeros((self.n_labels, ), dtype = tf.float32),
            tf.cast(self.indexes_to_keep, tf.int32)[:, tf.newaxis],
            tf.ones((len(self.indexes_to_keep), ), dtype = tf.float32)
        )
        init  = tf.lookup.KeyValueTensorInitializer(
            tf.cast(self.indexes_to_keep, tf.int32),
            tf.range(len(self.indexes_to_keep), dtype = tf.int32)
        )
        self.table = tf.lookup.StaticHashTable(init, default_value = -1)
        
        self.samples  = self.add_weight('samples', shape = (len(self.indexes_to_keep), ), initializer = 'zeros')
        self.accuracy = self.add_weight('accuracy', shape = (len(self.indexes_to_keep), ), initializer = 'zeros')

    def reset_state(self):
        self.samples.assign(tf.zeros((len(self.indexes_to_keep), ), dtype = tf.float32))
        self.accuracy.assign(tf.zeros((len(self.indexes_to_keep), ), dtype = tf.float32))
        
    def update_state(self, y_true, y_pred):
        y_true = self.table.lookup(y_true)
        y_pred = self.table.lookup(tf.argmax(y_pred * self.mask, axis = -1, output_type = tf.int32))

        self.samples.assign_add(tf.cast(tf.math.bincount(y_true, minlength = len(self.indexes_to_keep)), tf.float32))
        self.accuracy.assign_add(tf.cast(tf.math.bincount(
            tf.boolean_mask(y_true, y_true == y_pred), minlength = len(self.indexes_to_keep)
        ), tf.float32))
            
    def result(self):
        if not self.balanced:
            return tf.reduce_sum(self.accuracy) / tf.reduce_sum(self.samples)
        return tf.reduce_mean(self.accuracy / self.samples)

    def get_config(self):
        return {
            ** super().get_config(),
            'balanced' : self.balanced,
            'n_labels' : self.n_labels,
            'indexes_to_keep' : self.indexes_to_keep
        }
