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

from tensorflow.keras.layers import *
from tensorflow.keras.constraints import max_norm

from custom_layers import CustomActivation, ChannelMergingLayer
from custom_architectures.sota_eeg_attn_blocks import attention_block

#%% Reproduced EEGTCNet model: https://arxiv.org/abs/2006.00622
def EEGTCNet2(input_shape = (1125, None, 1),
              output_dim  = 4,
              layers      = 2,
              kernel_size = 4,
              filters     = 12,
              dropout     = 0.3,
              activation  = 'elu',

              eegnet_blocks  = 1,
              eegnet_filters = 8,
              eegnet_d       = 2,
              eegnet_kernel  = 32,
              eegnet_dropout = 0.3,
              eegnet_merge_first  = True,
              eegnet_merge_method = 'mean',
              eegnet_intermediate_blocks = 2,
              eegnet_intermediate_type   = 'conv',
             
              final_name       = 'output_layer',
              final_activation = 'softmax',
              
              name = 'EEGTCNet',
              ** kwargs
             ):
    if len(input_shape) == 2: input_shape = tuple(input_shape) + (1, )
    inp_length, inp_channels = input_shape[:2]
    
    inputs = Input(shape = input_shape, name = 'input_eeg')

    EEGNet_sep = EEGNet(
        input_layer = inputs, F1 = eegnet_filters, kernel_length = eegnet_kernel, D = eegnet_d, merge_method = eegnet_merge_method,
        channels = inp_channels, dropout = eegnet_dropout, n_blocks = eegnet_blocks, merge_first = eegnet_merge_first,
        intermediate_blocks = eegnet_intermediate_blocks, intermediate_block = eegnet_intermediate_type
    )
    block2 = ChannelMergingLayer(method = -1)(EEGNet_sep)
    outs = TCN_block(
        input_layer = block2, depth = layers, kernel_size = kernel_size, filters = filters,
        dropout = dropout, activation = activation
    )
    out = Lambda(lambda x: x[:,-1,:])(outs)
    
    output = Dense(
        output_dim, name = final_name if final_activation is None else 'final_dense'
    )(out)
    if final_activation is not None:
        output = CustomActivation(final_activation, name = final_name)(output)
    
    return tf.keras.Model(inputs = inputs, outputs = output, name = name)

#%% Reproduced EEGTCNet model: https://arxiv.org/abs/2006.00622
def EEGTCNet1D(input_shape = (None, 1),
               output_dim  = 4,
               layers      = 2,
               kernel_size = 4,
               filters     = 12,
               dropout     = 0.3,
               activation  = 'elu',

               eegnet_blocks  = 1,
               eegnet_filters = 8,
               eegnet_depth   = 2,
               eegnet_kernel  = 32,
               eegnet_strides = 8,
               eegnet_dropout = 0.3,
               eegnet_activation = 'elu',
             
               final_name       = 'output_layer',
               final_activation = 'softmax',
              
               name = 'EEGTCNet1D',
               ** kwargs
              ):
    inputs = Input(shape = input_shape, name = 'input_eeg')

    eegnet_out = EEGNet1D(
        inputs,
        filters     = eegnet_filters,
        kernel_size = eegnet_kernel,
        strides     = eegnet_strides,
        drop_rate   = eegnet_dropout,
        depth       = eegnet_depth,
        activation  = eegnet_activation
    )
    outs = TCN_block(
        input_layer = eegnet_out,
        depth       = layers,
        kernel_size = kernel_size,
        filters     = filters,
        dropout     = dropout,
        activation  = activation
    )
    out = Lambda(lambda x: x[:,-1,:])(outs)
    
    output = Dense(
        output_dim, name = final_name if final_activation is None else 'final_dense'
    )(out)
    if final_activation is not None:
        output = CustomActivation(final_activation, name = final_name)(output)
    
    return tf.keras.Model(inputs = inputs, outputs = output, name = name)

def EEGNet1D(input_layer, filters = 8, kernel_size = 64, depth = 2, strides = 8, n_blocks = 1, activation = 'elu', drop_rate = 0.25):
    F2 = filters * depth
    
    block1 = Conv1D(filters, kernel_size, padding = 'same', use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)
    block1 = Dropout(drop_rate)(block1)

    block2 = DepthwiseConv1D(1, use_bias = False, depth_multiplier = depth, depthwise_constraint = max_norm(1.))(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation(activation)(block2)
    block2 = AveragePooling1D(strides)(block2)
    block2 = Dropout(drop_rate)(block2)

    block3 = block2
    for i in range(n_blocks):
        block3 = SeparableConv1D(F2, 16, use_bias = False, padding = 'same')(block3)
        block3 = BatchNormalization(axis = -1)(block3)
        block3 = Activation(activation)(block3)
        if i < n_blocks - 1: block3 = Dropout(drop_rate)(block3)

    block3 = AveragePooling1D(strides)(block3)
    block3 = Dropout(drop_rate)(block3)
    return block3


def EEGNet(input_layer, F1 = 8, kernel_length = 64, D = 2, n_blocks = 1, channels = 22, dropout = 0.25, merge_first = True, merge_method = 'gated-convlstm', intermediate_blocks = 2, intermediate_block = 'max'):
    """
        EEGNet model from Lawhern et al 2018
        See details at https://arxiv.org/abs/1611.08024

        The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
    
        Notes
        -----
        The initial values in this model are based on the values identified by the authors
        
        References
        ----------
        .. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
           S. M., Hung, C. P., & Lance, B. J. (2018).
           EEGNet: A Compact Convolutional Network for EEG-based
           Brain-Computer Interfaces.
           arXiv preprint arXiv:1611.08024.
    """
    F2 = F1 * D
    
    block1 = Conv2D(F1, (kernel_length,  1), padding = 'same', use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)
    block1 = Dropout(dropout)(block1)

    if merge_first:
        if channels:
            block2 = DepthwiseConv2D(
                (1, channels), use_bias = False, depth_multiplier = D, depthwise_constraint = max_norm(1.)
            )(block1)
            block2 = BatchNormalization(axis = -1)(block2)
        elif merge_method:
            block2 = block1
            for i in range(intermediate_blocks):
                if intermediate_block == 'conv':
                    kernel = 8# if i == 0 else 3
                    block2 = Conv2D(
                        F2, (3, kernel), strides = (1, 1), padding = 'valid', use_bias = False,
                        dilation_rate = 2 ** i
                    )(block2)
                    block2 = BatchNormalization(axis = -1)(block2)
                    block2 = Activation('elu')(block2)
                    block2 = Dropout(dropout)(block2)
                    #block2 = MaxPooling2D((1, 4), padding = 'same')(block2)
                elif intermediate_block == 'max':
                    kernel = 4
                    block2 = MaxPooling2D((1, kernel), padding = 'same')(block2)
            
            block2 = ChannelMergingLayer(merge_method, keepdims = True)(block2)
        else:
            block2 = Conv2D(F2, (kernel_length,  1), padding = 'same', use_bias = False)(block1)
    else:
        block2 = Conv2D(F2, (kernel_length,  1), padding = 'same', use_bias = False)(block1)
    
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1))(block2)
    block2 = Dropout(dropout)(block2)

    block3 = block2
    for i in range(n_blocks):
        block3 = SeparableConv2D(F2, (16, 1), use_bias = False, padding = 'same')(block3)
        block3 = BatchNormalization(axis = -1)(block3)
        if i < n_blocks - 1:
            block3 = Activation('elu')(block3)
            block3 = Dropout(dropout)(block3)

    if not merge_first:
        if channels:
            block3 = DepthwiseConv2D(
                (1, channels), use_bias = False, depth_multiplier = 1, depthwise_constraint = max_norm(1.)
            )(block3)
            block3 = BatchNormalization(axis = -1)(block3)
        elif merge_method:
            for i in range(intermediate_blocks):
                if intermediate_block == 'conv':
                    kernel = 22 # if i == 0 else 3
                    block3 = Conv2D(
                        F2, (3, kernel), strides = (1, 1), padding = 'valid', use_bias = False,
                        dilation_rate = 2 ** i, activation = 'elu'
                    )(block3)
                    block3 = BatchNormalization(axis = -1)(block3)
                    block3 = Activation('elu')(block3)
                    block3 = Dropout(dropout)(block3)
                    #block3 = MaxPooling2D((1, 4), padding = 'same')(block3)
                elif intermediate_block == 'max':
                    kernel = 4
                    block3 = MaxPooling2D((1, kernel), padding = 'same')(block3)

            block3 = ChannelMergingLayer(merge_method, keepdims = True)(block3)

    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8,1))(block3)
    block3 = Dropout(dropout)(block3)
    return block3

#%% Temporal convolutional (TC) block used in the ATCNet model
def TCN_block(input_layer, depth, kernel_size, filters, dropout, activation = 'relu'):
    """ TCN_block from Bai et al 2018
        Temporal Convolutional Network (TCN)
        
        Notes
        -----
        THe original code available at https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
        This implementation has a slight modification from the original code
        and it is taken from the code by Ingolfsson et al at https://github.com/iis-eth-zurich/eeg-tcnet
        See details at https://arxiv.org/abs/2006.00622

        References
        ----------
        .. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
           An empirical evaluation of generic convolutional and recurrent networks
           for sequence modeling.
           arXiv preprint arXiv:1803.01271.
    """    
    
    block = Conv1D(
        filters, kernel_size = kernel_size, padding = 'causal', kernel_initializer = 'he_uniform'
    )(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(
        filters, kernel_size = kernel_size, padding = 'causal', kernel_initializer = 'he_uniform'
    )(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if input_layer.shape[-1] != filters:
        conv = Conv1D(filters, kernel_size = 1, padding = 'same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)
    
    for i in range(depth - 1):
        block = Conv1D(
            filters, kernel_size = kernel_size, dilation_rate = 2 ** (i + 1),
            padding = 'causal', kernel_initializer='he_uniform'
        )(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(
            filters, kernel_size = kernel_size, dilation_rate = 2 ** (i + 1),
            padding = 'causal', kernel_initializer = 'he_uniform'
        )(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out   = Activation(activation)(added)
        
    return out

custom_functions   = {
    'EEGTCNet2'      : EEGTCNet2
}

custom_objects = {
    'CustomActivation'    : CustomActivation,
    'ChannelMergingLayer' : ChannelMergingLayer
}