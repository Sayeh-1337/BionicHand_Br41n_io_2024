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

import enum
import tensorflow as tf

_default_layer_config = {
    'kernel_size' : 3, 'padding' : 'same', 'strides' : 1, 'use_bias' : False, 'activation' : None,
    'dilation_rate' : 2
}    
    
class ChannelMergingLayer(tf.keras.layers.Layer):
    def __init__(self, method, layer_config = _default_layer_config, shuffle = False, keepdims = False, ** kwargs):
        super().__init__(** kwargs)
        self.method   = method if not isinstance(method, str) else method.split('-')
        self.shuffle  = shuffle
        self.keepdims = keepdims
        self.layer_config  = layer_config
        
        self.layer = None

    def build(self, input_shape):
        if self.built: return
        self.built = True
        if not isinstance(self.method, list): return
        
        if 'convlstm' in self.method:
            self.return_sequences = len(self.method) == 1 or ('bi' in self.method and len(self.method) == 2)
            if 'filters' not in self.layer_config: self.layer_config['filters'] = input_shape[-1] // (2 if 'bi' in self.method else 1)
            self.layer = tf.keras.layers.ConvLSTM1D(
                ** self.layer_config, return_sequences = self.return_sequences
            )
            if 'bi' in self.method: self.layer = tf.keras.layers.Bidirectional(self.layer)
        
    def call(self, inputs, training = False):
        if isinstance(self.method, int): return inputs[:, :, self.method, :]

        out = inputs
        if 'convlstm' in self.method:
            if training and self.shuffle:
                inputs = tf.transpose(inputs, [2, 0, 1, 3])
                inputs = tf.random.shuffle(inputs)
                inputs = tf.transpose(inputs, [1, 2, 0, 3])
            
            out = self.layer(tf.transpose(inputs, [0, 2, 1, 3]), training = training)

            if not self.return_sequences:
                if self.keepdims: out = tf.expand_dims(out, axis = 2)
                return out
            
            out = tf.transpose(out, [0, 2, 1, 3])

        if 'weighted' in self.method:
            weights = tf.nn.softmax(tf.reduce_sum(out, axis = 1, keepdims = True), axis = -2)
            out     = inputs * weights
        elif 'gated' in self.method:
            weights = tf.sigmoid(tf.reduce_sum(out, axis = 1, keepdims = True), axis = -2)
            out     = inputs * weights
        elif 'softmax' in self.method:
            out     = inputs * tf.nn.softmax(out, axis = -2)
        
        if 'avg' in self.method or 'mean' in self.method:
            return tf.reduce_mean(out, axis = -2, keepdims = self.keepdims)
        elif 'sum' in self.method:
            return tf.reduce_sum(out, axis = -2, keepdims = self.keepdims)
        elif 'max' in self.method:
            return tf.reduce_max(out, axis = -2, keepdims = self.keepdims)
        else:
            return tf.reduce_sum(out, axis = -2, keepdims = self.keepdims)
    
    def get_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[3])
            
    def get_config(self):
        return {
            ** super().get_config(),
            'method' : self.method,
            'shuffle' : self.shuffle,
            'layer_config' : self.layer_config,
            'keepdims' : self.keepdims
        }
    
