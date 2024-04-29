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

""" Code for the Euclidian Alignment (EA) normalization """

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.linalg import sqrtm, inv

from utils import load_json, dump_json

class EuclidianAlignment:
    def __init__(self, ids = None, matrices = None):
        if isinstance(ids, str): ids = load_json(ids) if os.path.exists(ids) else None
        if isinstance(matrices, str): matrices = np.load(matrices) if os.path.exists(matrices) else None
        self.ids = tf.as_string(ids) if ids is not None else None
        self.matrices = tf.cast(matrices, tf.float32) if matrices is not None else None

    def __contains__(self, subject):
        return tf.reduce_any(self.ids == tf.as_string(subject))
    
    def __getitem__(self, subject):
        return tf.gather(self.matrices, tf.argmax(tf.cast(self.ids == tf.as_string(subject), tf.int32)))

    def __call__(self, inputs, subject = None):
        if isinstance(inputs, (dict, pd.Series)): inputs, subject = inputs['eeg'], inputs['id']
        if not tf.executing_eagerly(): return self.transform(inputs, subject)
        return tf.cond(
            subject in self,
            lambda: self.transform(inputs, subject),
            lambda: inputs
        )

    def transform(self, inputs, subject):
        is_batched = len(tf.shape(inputs)) == 3
        if not is_batched: inputs = inputs[tf.newaxis]
        normalized = tf.einsum('fe,bet->bft', self[subject], tf.transpose(inputs, [0, 2, 1]))
        normalized = tf.transpose(normalized, [0, 2, 1])
        return normalized if is_batched else normalized[0]

    def fit(self, inputs, subject = None):
        if isinstance(inputs, dict): return
        if isinstance(inputs, pd.DataFrame):
            assert 'id' in inputs.columns and 'eeg' in inputs.columns
            if subject is None and len(inputs['id'].unique()) > 1:
                for subj, subj_data in inputs.groupby('id'):
                    self.fit(subj_data, subject = subj)
                return
            if subject is None: subject = inputs.iloc[0]['id']
            inputs = np.array(list(inputs['eeg'].values))

        if subject is None:
            raise ValueError('`subject` is None and `inputs` is not a `pd.DataFrame` : {}'.format(inputs))
        if subject in self: return
        subject, r = tf.as_string([subject]), tf.cast(compute_r(inputs), tf.float32)[tf.newaxis]
        if self.ids is None:
            self.ids, self.matrices = subject, r
        else:
            self.ids = tf.concat([self.ids, subject], axis = 0)
            self.matrices = tf.concat([self.matrices, r], axis = 0)
    
    def get_config(self):
        return {}

    def save_to_file(self, filename):
        ids_filename, matrices_filename = filename.replace('.json', '_ids.json'), filename.replace('.json', '_matrices.npy')
        if self.ids is not None:
            dump_json(ids_filename, self.ids)
            np.save(matrices_filename, self.matrices.numpy())
        dump_json(
            filename, {** self.get_config(), 'ids' : ids_filename, 'matrices' : matrices_filename}, indent = 4
        )
    
    @classmethod
    def load_from_file(cls, filename):
        return cls(** load_json(filename, default = {}))

def compute_r(x):
    r = np.mean(np.einsum('bet, tab -> bea', x, np.transpose(x, [2, 1, 0])), axis = 0)
    return inv(sqrtm(r))
