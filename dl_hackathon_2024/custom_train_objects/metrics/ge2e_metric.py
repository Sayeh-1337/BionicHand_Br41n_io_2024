
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

import numpy as np
import tensorflow as tf

from utils.distance import distance

class GE2EMetric(tf.keras.metrics.Metric):
    def __init__(self,
                 mode = 'softmax',
                 distance_metric    = 'cosine',

                 name = 'ge2e_metric',
                 
                 ** kwargs
                ):
        assert mode in ('softmax', 'contrast')
        
        super().__init__(name = name, ** kwargs)
        self.mode   = mode
        self.distance_metric    = distance_metric
        
        if mode == 'softmax':
            self.metric     = tf.keras.metrics.SparseCategoricalAccuracy()
            self.format_fn  = self.softmax_format
        else:
            from .equal_error_rate import EER
            
            self.metric     = EER()
            self.format_fn  = self.contrast_format
        
        from custom_train_objects.losses.ge2e_loss import GE2ELoss

        self.ge2e = GE2ELoss(distance_metric = distance_metric)
        
        self.w = None
        self.b = None
    
    @property
    def metric_names(self):
        return self.metric.metric_names if hasattr(self.metric, 'metric_names') else self.name

    def set_variables(self, variables):
        self.w, self.b = variables
    
    def reset_state(self, * args, ** kwargs):
        self.metric.reset_state(* args, ** kwargs)
    
    def softmax_format(self, idx, similarity_matrix):
        return idx, tf.nn.softmax(tf.reshape(similarity_matrix, [-1, tf.shape(similarity_matrix)[-1]]), axis = -1)
    
    def contrast_format(self, idx, similarity_matrix):
        target_matrix = tf.one_hot(idx, depth = tf.shape(similarity_matrix)[-1])
        return tf.reshape(target_matrix, [-1]), tf.sigmoid(tf.reshape(similarity_matrix, [-1]))
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        uniques, idx    = tf.unique(tf.reshape(y_true, [-1]))
        nb_speakers     = tf.size(uniques)
        
        # Shape == (nb_speakers, nb_utterances, embedded_dim)
        speaker_embedded = tf.reshape(y_pred, [nb_speakers, -1, tf.shape(y_pred)[-1]])
        
        cos_sim_matrix = self.ge2e.similarity_matrix(speaker_embedded)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        target, pred = self.format_fn(idx, cos_sim_matrix)

        return self.metric.update_state(target, pred)
        
    def result(self):
        return self.metric.result()
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'mode'  : self.mode,
            'distance_metric'   : self.distance_metric
        })
        return config
