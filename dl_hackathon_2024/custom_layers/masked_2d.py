
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

def _get_new_length(mask, kernel_size, strides, padding):
    seq_len = tf.reduce_sum(tf.cast(mask, tf.int32), axis = -1)
    if padding != 'same': seq_len = seq_len - kernel_size + 1
    new_len = tf.maximum(1, tf.cast(tf.math.ceil(seq_len / strides), tf.int32))
    return new_len

def build_mask_2d(inputs, mask, kernel_size, strides, padding, dilation = 1):
    if mask is None: return None

    #tf.print(tf.reduce_sum(tf.cast(mask, tf.int32), -1), kernel_size, strides, padding)
    if all(s == 1 for s in strides):
        if padding == 'same': return mask
        return mask[:, (kernel_size[0] - 1) * dilation[0] :, (kernel_size[1] - 1) * dilation[1] :]
    
    height  = _get_new_length(mask[:, :, 0], kernel_size[0], strides[0], padding)
    width   = _get_new_length(mask[:, 0, :], kernel_size[1], strides[1], padding)
    
    h_mask = tf.sequence_mask(height, tf.shape(inputs)[1], dtype = tf.bool)
    w_mask = tf.sequence_mask(width, tf.shape(inputs)[2], dtype = tf.bool)
    
    return tf.logical_and(
        tf.expand_dims(h_mask, axis = -1), tf.expand_dims(w_mask, axis = 1)
    )

class MaskedMaxPooling2D(tf.keras.layers.MaxPooling2D):
    def compute_mask(self, inputs, mask = None):
        return build_mask_1d(inputs, mask, self.pool_size[0], self.strides[0], self.padding)

    def call(self, inputs, mask = None):
        out = super().call(inputs)

        if mask is not None:
            out = out * tf.expand_dims(tf.cast(self.compute_mask(out, mask), out.dtype), axis = -1)
        
        return out

class MaskedAveragePooling2D(tf.keras.layers.AveragePooling2D):
    def compute_mask(self, inputs, mask = None):
        return build_mask_1d(inputs, mask, self.pool_size[0], self.strides[0], self.padding)

    def call(self, inputs, mask = None):
        out = super().call(inputs)

        if mask is not None:
            out = out * tf.expand_dims(tf.cast(self.compute_mask(out, mask), out.dtype), axis = -1)
        
        return out

class MaskedZeroPadding2D(tf.keras.layers.ZeroPadding2D):
    def compute_mask(self, inputs, mask = None):
        if mask is None: return None
        if len(mask.shape) == 2: mask = tf.expand_dims(mask, axis = 0)
        
        height  = sum(self.padding[0]) + tf.reduce_sum(tf.cast(mask[:, :, 0], tf.int32), -1)
        width   = sum(self.padding[1]) + tf.reduce_sum(tf.cast(mask[:, 0, :], tf.int32), -1)

        h_mask = tf.sequence_mask(height, tf.shape(inputs)[1] + sum(self.padding[0]), dtype = tf.bool)
        w_mask = tf.sequence_mask(width, tf.shape(inputs)[2] + sum(self.padding[1]), dtype = tf.bool)

        return tf.logical_and(
            tf.expand_dims(h_mask, axis = -1), tf.expand_dims(w_mask, axis = 1)
        )
    
    def call(self, inputs, mask = None):
        out = super().call(inputs)
        
        if mask is not None:
            if len(mask.shape) == 2: mask = tf.expand_dims(mask, axis = 0)
            out_mask = tf.pad(
                mask, [(0, 0), self.padding[0], self.padding[1]], constant_values = True
            )
            out      = tf.where(tf.expand_dims(out_mask, axis = -1), out, 0.)
        
        return out
    
class MaskedConv2D(tf.keras.layers.Conv2D):
    def compute_mask(self, inputs, mask = None):
        return build_mask_2d(
            inputs, mask, self.kernel_size, self.strides, self.padding, self.dilation_rate
        )
    
    def call(self, inputs, mask = None):
        if mask is not None:
            inputs = tf.where(tf.expand_dims(mask, axis = -1), inputs, 0.)
        
        out = super().call(inputs)

        if mask is not None:
            out_mask = self.compute_mask(out, mask)
            out = tf.where(tf.expand_dims(out_mask, axis = -1), out, 0.)
            out._keras_mask = out_mask
        
        return out

class Masking2D(tf.keras.layers.Layer):
    def __init__(self, mask_value = 0., axis = -1, ** kwargs):
        super().__init__(** kwargs)
        
        self.axis   = axis
        self.mask_value = mask_value
    
    def compute_mask(self, inputs, mask = None):
        if mask is not None: return mask
        return tf.reduce_any(tf.math.not_equal(inputs, self.mask_value), axis = self.axis)

    def call(self, inputs, mask = None):
        inputs._keras_mask = self.compute_mask(inputs, mask = mask)
        return inputs