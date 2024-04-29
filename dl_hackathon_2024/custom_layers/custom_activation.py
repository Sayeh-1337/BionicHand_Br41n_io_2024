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

from.custom_activations import get_activation

class CustomActivation(tf.keras.layers.Layer):
    def __init__(self, activation, ** kwargs):
        super().__init__(** kwargs)
        self.activation = activation if isinstance(activation, list) else [activation]

        self.activation_fn = [
            get_activation(act, ** kwargs) for act in self.activation
        ]
        
    def call(self, inputs):
        for activation in self.activation_fn: inputs = activation(inputs)
        return inputs

    def get_config(self):
        return {** super().get_config(), 'activation' : self.activation}