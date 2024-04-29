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

from tensorflow.keras import layers
from tensorflow.keras.layers import *

from custom_layers import ChannelMergingLayer
from .sota_eeg_arch import *

class ChannelSplitter(tf.keras.layers.Layer):
    def __init__(self, n_channels, axis = -1, ** kwargs):
        super().__init__(** kwargs)
        self.n_channels = n_channels
        self.axis = axis
        
    def call(self, x):
        return tf.split(x, self.n_channels, axis = self.axis)

    def get_config(self):
        return {** super().get_config(), 'n_channels' : self.n_channels, 'axis' : self.axis}

class SingleChannelCombination(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 output_dim,

                 backbone = 'eegsimpleconv',
                 core_architecture     = None,
                 
                 channel_drop_rate     = 0.25,
                 channel_embedding_dim = 8,

                 final_activation = None,
                 
                 name = None,
                 ** kwargs
                ):
        super().__init__(name = name if name else 'splitted_{}'.format(core_architecture))
        if core_architecture: backbone = core_architecture
        self._input_shape = tuple(input_shape)
        self.output_dim = output_dim
        self.backbone   = backbone
        self.channel_drop_rate = channel_drop_rate
        self.channel_embedding_dim = channel_embedding_dim
        self.final_activation = final_activation
        self.kwargs = kwargs

        from custom_architectures import get_architecture

        self.n_channels = input_shape[1]
        single_chann_input_shape = input_shape[:1] + (1, ) + input_shape[2:]

        self.sub_models = [get_architecture(
            backbone,
            input_shape = single_chann_input_shape,
            output_dim  = channel_embedding_dim,

            name        = '{}_chan_{}'.format(core_architecture, i + 1),
            ** kwargs
        ) for i in range(self.n_channels)]

        self.concat = tf.keras.layers.Concatenate()
        embedded_channels = tf.keras.layers.Input(shape = (channel_embedding_dim * self.n_channels, ))
        self.classifier = tf.keras.Model(
            embedded_channels, build_output_layers(embedded_channels, output_dim, final_activation)
        )
        self.build(self.input_shape)

    @property
    def input_shape(self):
        shape = list(self.sub_models[0].input_shape)
        shape[2] = self.n_channels
        return tuple(shape)

    @property
    def output_shape(self):
        return (None, self.output_dim)
        
    def call(self, inputs, training = False):
        outputs = []
        for signal, model in zip(tf.split(inputs, self.n_channels, axis = 2), self.sub_models):
            out = model(signal, training = training)
            if training and self.channel_drop_rate > 0.:
                out = out * tf.cast(tf.random.uniform(()) < self.channel_drop_rate, out.dtype)
            outputs.append(out)

        return self.classifier(self.concat(outputs))

    def get_config(self):
        return {
            'input_shape' : self._input_shape,
            'output_dim' : self.output_dim,
            'backbone' : self.backbone,
            'channel_drop_rate' : self.channel_drop_rate,
            'channel_embedding_dim' : self.channel_embedding_dim,
            'final_activation' : self.final_activation,
            'kwargs' : self.kwargs
        }
    
def _EEGSimpleConv(input_shape = (None, 22),
                  output_dim  = 4,

                  n_convs     = None,
                  filters     = None,
                  kernel_size = None,
                  use_bias    = False,
                  activation  = 'relu',

                  multi_subjects   = False,
                  
                  final_name       = 'output_layer',
                  final_activation = 'softmax',
                  
                  name = 'EEGSimpleConv',
                  ** kwargs
                 ):
    if n_convs is None:     n_convs = 4 if multi_subjects else 1
    if filters is None:     filters = 109 if multi_subjects else 85
    if kernel_size is None: kernel_size = 8 if multi_subjects else 15
    
    input_shape = (None, ) + tuple(input_shape[1:])
    
    inputs = tf.keras.Input(shape = input_shape, name = 'input_eeg')

    x = tf.keras.layers.Conv1D(filters, kernel_size = kernel_size, use_bias = use_bias, padding = 'same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = get_activation(activation)(x)

    for i in range(n_convs):
        if i > 0: filters = int(1.414 * filters)

        for j in range(2):
            x = tf.keras.layers.Conv1D(filters, kernel_size = kernel_size, use_bias = use_bias, padding = 'same')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            if j == 0: x = tf.keras.layers.MaxPooling1D(2)(x)
            x = get_activation(activation)(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    outputs = build_output_layers(x, output_dim = output_dim, activation = final_activation, name = final_name)
    return tf.keras.Model(inputs, outputs, name = name)

def _get(idx, values):
    if isinstance(values, list): return values[idx]
    elif callable(values):       return values(idx)
    return values

    
def cnn(input_shape,
        output_dim,
        
        n_conv      = 3,
        filters     = lambda i: min(32 * 2 ** i, 128),
        kernel_size = lambda i: max(16 // 2 ** i, 3),
        strides     = (2, 1),
        use_bias    = True,
        padding     = 'same',
        dropout     = 0.25,
        norm_type   = 'batch',
        activation  = 'relu',
        
        eegnet_filters = 64,
        eegnet_kernel  = 8,
        eegnet_dropout = 0.2,
        eegnet_d       = 2,
        
        conv_type   = 'Conv2D',
        channel_merging = 'weighted_sum',
        eegnet_channel_merging = None,
        flatten         = None,
        flatten_units   = 32,
        
        add_tcn_block   = True,
        tcn_depth       = 2,
        tcn_kernel_size = 4,
        tcn_filters     = 32,
        tcn_dropout     = 0.3,
        tcn_activation = 'elu',

        final_name       = 'output_layer',
        final_activation = 'softmax',
        
        name = 'cnn',
        ** kwargs
       ):
    conv_class = getattr(layers, conv_type)
    if len(input_shape) == 2 and '2D' in conv_type: input_shape = tuple(input_shape) + (1, )
    
    if not isinstance(filters, list):
        if isinstance(filters, int): filters = [filters] * n_conv
        elif callable(filters):      filters = [filters(i) for i in range(n_conv)]
    
    inputs = Input(shape = input_shape, name = 'input_eeg')   #     TensorShape([None, 1, 22, 1125])
    
    x = inputs
    for i, filters_i in enumerate(filters):
        x = conv_class(
            filters     = filters_i,
            kernel_size = _get(i, kernel_size),
            strides     = _get(i, strides),
            use_bias    = _get(i, use_bias),
            padding     = _get(i, padding)
        )(x)
        if norm_type == 'layer':   x = LayerNormalization()(x)
        elif norm_type == 'batch': x = BatchNormalization()(x)
        
        x = Activation(_get(i, activation))(x)
        
        if dropout: x = Dropout(dropout)(x)

    if eegnet_filters:
        if eegnet_channel_merging is None: eegnet_channel_merging = channel_merging
        x = EEGNet(
            input_layer   = x,
            channels      = x.shape[-2],
            merge_method  = eegnet_channel_merging,
            kernel_length = eegnet_kernel,
            dropout       = eegnet_dropout,
            F1            = eegnet_filters,
            D             = eegnet_d
        )

    if len(x.shape) == 4:
        x = ChannelMergingLayer(channel_merging if not eegnet_filters else -1)(x)

    if add_tcn_block:
        x = TCN_block(
            input_layer     = x,
            input_dimension = x.shape[-1],
            depth       = tcn_depth,
            kernel_size = tcn_kernel_size,
            filters     = tcn_filters,
            dropout     = tcn_dropout,
            activation  = tcn_activation
        )
        
    output = None
    if len(x.shape) == 3:
        if flatten == 'max':   x = GlobalMaxPooling1D()(x)
        elif flatten == 'avg': x = GlobalAveragePooling1D()(x)
        elif flatten is None:  x = Flatten()(x)
        else:
            if flatten_units is not None:
                x = getattr(layers, flatten.upper())(flatten_units)(x)
            else:
                output = getattr(layers, flatten.upper())(output_dim)(x)
    
    if output is None: output = Dense(output_dim)(x)
    if final_activation is not None: output = Activation(final_activation, name = final_name)(output)
    
    return tf.keras.Model(inputs = inputs, outputs = output, name = name)

custom_objects = {
    'SingleChannelCombination' : SingleChannelCombination,
    'ChannelMergingLayer' : ChannelMergingLayer
}

custom_functions = {
    'SingleChannelCombination' : SingleChannelCombination,
    'cnn' : cnn
}
