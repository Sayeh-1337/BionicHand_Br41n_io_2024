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
import pandas as pd
import tensorflow as tf

from utils import convert_to_str
from utils.eeg_utils import build_lookup_table
from models.interfaces.base_eeg_model import BaseEEGModel

class EEGClassifier(BaseEEGModel):
    input_signature = BaseEEGModel.eeg_signature
    get_input       = BaseEEGModel.get_eeg
    
    def __init__(self, rate, channels, labels, nb_class = None, ** kwargs):
        self._init_eeg(rate, channels, ** kwargs)
        
        self.labels   = convert_to_str(labels)
        if not isinstance(self.labels, list): self.labels = [self.labels]
        self.nb_class = nb_class if nb_class and nb_class >= len(self.labels) else len(self.labels)

        super().__init__(** kwargs)
        
        self._lookup_table = build_lookup_table(self.labels, default = -1)
    
    def _build_model(self, ** config):
        config.update({
            'output_dim'       : self.nb_class if not self.is_binary_classifier else 1,
            'final_activation' : None
        })
        return super()._build_model(** config)

    @property
    def is_binary_classifier(self):
        return self.nb_class <= 2
    
    @property
    def output_signature(self):
        return tf.TensorSpec(shape = (None, ), dtype = tf.int32)
    
    @property
    def encoder(self):
        if not isinstance(self.model, tf.keras.Sequential):
            return tf.keras.Model(self.model.input, self.model.layers[-2].output)
        
        encoder = tf.keras.Sequential(name = 'feature_extractor')
        for l in self.model.layers[:-2]:
            encoder.add(l)
        return encoder
    
    def __str__(self):
        des = super().__str__()
        des += self._str_eeg()
        des += "- Labels ({}) : {}\n".format(len(self.labels), self.labels[:25])
        return des

    def infer(self, inputs, training = False, mask_by_id = False, ids = None, ** _):
        if mask_by_id and self.is_binary_classifier: raise RuntimeError('`mask_by_id` is inconsistent for binary classifier')
        
        out = self(inputs, training = training)
        if mask_by_id:
            assert ids is not None, '`ids` must be provided for masking !'
            mask = tf.cast([
                [l.startswith(id_i) for l in self.labels] for id_i in ids
            ], dtype = tf.bool)
            out = tf.where(mask, out, out.dtype.min)
        
        if self.is_binary_classifier: return tf.sigmoid(out)
        return self.labels, tf.nn.softmax(out, axis = -1)

    def compile(self, loss = None, metrics = None, reduction = 'none', ** kwargs):
        if loss is None:
            loss = 'SparseCategoricalCrossentropy' if not self.is_binary_classifier else 'binary_crossentropy'
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = reduction)
        if not metrics:
            metrics = 'sparse_categorical_accuracy' if not self.is_binary_classifier else 'binary_accuracy'
            metrics = [{'metric' : metrics, 'config' : {'name' : 'accuracy'}}]
        kwargs.setdefault('loss_config', {}).setdefault('name', 'loss')
        kwargs['loss_config']['from_logits'] = True
        super().compile(loss = loss, metrics = metrics, ** kwargs)
    
    def get_output(self, data):
        if isinstance(data, pd.DataFrame):        data = data['label' if 'label' in data.columns else 'id'].values
        elif isinstance(data, (dict, pd.Series)): data = data['label' if 'label' in data else 'id']
        
        label = data if isinstance(data, tf.Tensor) else tf.convert_to_tensor(data)
        
        if label.dtype == tf.string:
            label = self._lookup_table.lookup(label)
        else:
            label = tf.cast(label, tf.int32)
        
        return label
    
    def filter_data(self, inputs, output):
        return output != -1

    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_eeg(),
            'labels'    : self.labels,
            'nb_class'  : self.nb_class
        })
        
        return config

        