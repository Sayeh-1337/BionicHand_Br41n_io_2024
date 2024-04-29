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
from tensorflow.keras.constraints import max_norm as max_norm_constraint
from tensorflow.keras.regularizers import L2

from custom_layers import CustomActivation, get_activation
from custom_architectures.sota_eeg_attn_blocks import attention_block

def build_output_layers(x, output_dim, activation = None, use_bias = True, name = 'classification_layer'):
    if not isinstance(output_dim, (list, tuple)): output_dim = [output_dim]

    outputs = []
    for i, dim in enumerate(output_dim):
        layer_name = name if len(output_dim) == 1 else '{}_{}'.format(name, i + 1)
        output     = Dense(dim, use_bias = use_bias, name = layer_name)(x)
        if activation: output = CustomActivation(activation)(output)
        outputs.append(output)
    
    return outputs[0] if len(output_dim) == 1 else outputs

def EEGSimpleConv(input_shape = (None, 22),
                  output_dim  = 4,

                  n_convs     = None,
                  filters     = None,
                  kernel_size = None,
                  use_bias    = False,
                  activation  = 'relu',
                  drop_rate   = 0.,

                  multi_subjects   = False,
                  
                  final_name       = 'output_layer',
                  final_activation = 'softmax',
                  
                  name = 'EEGSimpleConv',
                  ** kwargs
                 ):
    if n_convs is None:     n_convs = 4 if multi_subjects else 1
    if filters is None:     filters = 109 if multi_subjects else 85
    if kernel_size is None: kernel_size = 8 if multi_subjects else 15

    print(n_convs, filters, kernel_size)
    input_shape = (None, ) + tuple(input_shape[1:])
    
    inputs = tf.keras.Input(shape = input_shape, name = 'input_eeg')

    x = tf.keras.layers.Conv1D(
        filters, kernel_size = kernel_size, use_bias = use_bias, padding = 'same'
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = get_activation(activation)(x)

    for i in range(n_convs):
        if i > 0: filters = int(1.414 * filters)

        for j in range(2):
            x = tf.keras.layers.Conv1D(filters, kernel_size = kernel_size, use_bias = use_bias, padding = 'same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            if j == 0: x = tf.keras.layers.MaxPooling1D(2)(x)
            x = get_activation(activation)(x)
            if drop_rate: x = tf.keras.layers.Dropout(drop_rate)(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    outputs = build_output_layers(
        x, output_dim = output_dim, activation = final_activation, name = final_name
    )
    return tf.keras.Model(inputs, outputs, name = name)

def ATCNetV2(input_shape    = (1125, 22, 1),
             output_dim     = 4,
             n_windows  = 5,
             attention  = 'mha',
             
             eegn_F1     = 16,
             eegn_D      = 2,
             eegn_kernel_size = 64,
             eegn_pool_size   = 7,
             eegn_dropout     = 0.3, 
           
             tcn_depth       = 2,
             tcn_kernel_size = 4,
             tcn_filters     = 32,
             tcn_dropout     = 0.3, 
             tcn_activation = 'elu',
             
             fuse   = 'average',
             dense_weight_decay = 0.5,
             conv_weight_decay  = 0.009,
             conv_max_norm  = 0.6,
           
             final_name       = 'output_layer',
             final_activation = 'softmax',
           
             name = 'ATCNet',
             ** kwargs
            ):
    
    """ ATCNet model from Altaheri et al 2023.
        See details at https://ieeexplore.ieee.org/abstract/document/9852687
    
        Notes
        -----
        The initial values in this model are based on the values identified by
        the authors
        
        References
        ----------
        .. H. Altaheri, G. Muhammad, and M. Alsulaiman. "Physics-informed 
           attention temporal convolutional network for EEG-based motor imagery 
           classification." IEEE Transactions on Industrial Informatics, 
           vol. 19, no. 2, pp. 2249-2258, (2023) 
           https://doi.org/10.1109/TII.2022.3197419
    """
    if len(input_shape) == 2: input_shape = tuple(input_shape) + (1, )
    inp_length, inp_channels = input_shape[:2]

    if inp_channels == 1:
        n_windows    = 2
        eegn_filters = 8
        tcn_filters  = 16
    
    inputs = Input(shape = input_shape, name = 'input_eeg')

    block1 = ConvBlockV2(
        input_layer = inputs,
        D   = eegn_D,
        filters = eegn_F1,
        kernel_size = eegn_kernel_size,
        pool_size   = eegn_pool_size,
        weight_decay    = conv_weight_decay,
        max_norm    = conv_max_norm,
        dropout     = eegn_dropout
    )
    block1 = Lambda(lambda x: x[:,:,-1,:])(block1)
    
    # Sliding window 
    sw_concat = []   # to store concatenated or averaged sliding window outputs
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]
        
        # Attention_model
        if attention is not None:
            if (attention == 'se' or attention == 'cbam'):
                block2 = Permute((2, 1))(block2) # shape=(None, 32, 16)
                block2 = attention_block(block2, attention)
                block2 = Permute((2, 1))(block2) # shape=(None, 16, 32)
            else: block2 = attention_block(block2, attention)

        # Temporal convolutional network (TCN)
        block3 = TCNBlockV2(
            input_layer = block2,
            depth   = tcn_depth,
            filters = tcn_filters,
            kernel_size = tcn_kernel_size,
            weight_decay    = conv_weight_decay,
            max_norm    = conv_max_norm,
            activation  = tcn_activation,
            dropout = tcn_dropout
        )
        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:,-1,:])(block3)
        
        # Outputs of sliding window: Average_after_dense or concatenate_then_dense
        if fuse == 'average':
            sw_concat.append(Dense(
                output_dim, kernel_regularizer = L2(dense_weight_decay)
            )(block3))
        elif fuse == 'concat':
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])
                
    if fuse == 'average':
        if len(sw_concat) > 1: # more than one window
            output = tf.keras.layers.Average()(sw_concat)
        else: # one window (# windows = 1)
            output = sw_concat[0]
    elif(fuse == 'concat'):
        output = Dense(output_dim, kernel_regularizer = L2(dense_weight_decay))(sw_concat)
               
    if final_activation is not None:
        output = CustomActivation(final_activation, name = final_name)(output)
       
    return tf.keras.Model(inputs = inputs, outputs = output, name = name)



#%% The proposed ATCNet model, https://doi.org/10.1109/TII.2022.3197419
def ATCNet(input_shape = (1125, 22, 1),
           output_dim  = 4,
           n_windows   = 5,
           attention   = 'mha', 
           
           eegn_F1     = 16,
           eegn_D      = 2,
           eegn_kernel_size = 64,
           eegn_pool_size   = 7,
           eegn_dropout     = 0.3, 
           
           tcn_depth       = 2,
           tcn_kernel_size = 4,
           tcn_filters     = 32,
           tcn_dropout     = 0.3, 
           tcn_activation = 'elu',
           fuse = 'average',
           
           final_name       = 'output_layer',
           final_activation = 'softmax',
           
           name = 'ATCNet',
           ** kwargs
          ):
    """
        ATCNet model from Altaheri et al 2022.
        See details at https://ieeexplore.ieee.org/abstract/document/9852687
    
        Notes
        -----
        The initial values in this model are based on the values identified by the authors
        
        References
        ----------
        .. H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed 
           attention temporal convolutional network for EEG-based motor imagery 
           classification," in IEEE Transactions on Industrial Informatics, 2022, 
           doi: 10.1109/TII.2022.3197419.
    """
    if len(input_shape) == 2: input_shape = tuple(input_shape) + (1, )
    inp_length, inp_channels = input_shape[:2]
    
    inputs = Input(shape = input_shape, name = 'input_eeg')   #     TensorShape([None, 1, 22, 1125])
    regRate = 0.25
    num_filters = eegn_F1
    F2 = num_filters * eegn_D

    block1 = Conv_block(
        input_layer = inputs, F1 = eegn_F1, D = eegn_D, 
        kernel_length = eegn_kernel_size, pool_size = eegn_pool_size,
        channels = inp_channels, dropout = eegn_dropout
    )
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)
     
    # Sliding window 
    sw_concat = []   # to store concatenated or averaged sliding window outputs
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]
        
        # Attention_model
        if attention is not None:
            block2 = attention_block(block2, attention)

        # Temporal convolutional network (TCN)
        block3 = TCN_block(
            input_layer = block2, input_dimension = F2, depth = tcn_depth,
            kernel_size = tcn_kernel_size, filters = tcn_filters, 
            dropout = tcn_dropout, activation = tcn_activation
        )
        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:, -1, :])(block3)
        
        # Outputs of sliding window: Average_after_dense or concatenate_then_dense
        if fuse == 'average':
            sw_concat.append(Dense(output_dim, kernel_constraint = max_norm_constraint(regRate))(block3))
        elif fuse == 'concat' :
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])
                
    if fuse == 'average' :
        if len(sw_concat) > 1: # more than one window
            output = tf.keras.layers.Average()(sw_concat)
        else: # one window (# windows = 1)
            output = sw_concat[0]
    elif fuse == 'concat':
        output = Dense(output_dim, kernel_constraint = max_norm_constraint(regRate))(sw_concat)
    
    if final_activation is not None:
        output = CustomActivation(final_activation, name = final_name)(output)
    
    return tf.keras.Model(inputs = inputs, outputs = output, name = name)


#%% Temporal convolutional (TC) block used in the ATCNet model
def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout, activation = 'relu'):
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
    if input_dimension != filters:
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


#%% Reproduced TCNet_Fusion model: https://doi.org/10.1016/j.bspc.2021.102826
def TCNet_Fusion(input_shape = (1125, 22, 1),
                 output_dim  = 4,
                 layers      = 2,
                 kernel_size = 4,
                 filters     = 12,
                 dropout     = 0.3,
                 activation  = 'elu',
                 
                 F1  = 24,
                 D   = 2,
                 kernel_length = 32,
                 dropout_eeg = 0.3,
                 
                 final_name       = 'output_layer',
                 final_activation = 'softmax',
                 
                 name = 'TCNet_Fusion',
                 ** kwargs
                ):
    """
        TCNet_Fusion model from Musallam et al 2021.
        See details at https://doi.org/10.1016/j.bspc.2021.102826
    
        Notes
        -----
        The initial values in this model are based on the values identified by the authors
        
        References
        ----------
        .. Musallam, Y.K., AlFassam, N.I., Muhammad, G., Amin, S.U., Alsulaiman,
           M., Abdul, W., Altaheri, H., Bencherif, M.A. and Algabri, M., 2021. 
           Electroencephalography-based motor imagery classification
           using temporal convolutional network fusion. 
           Biomedical Signal Processing and Control, 69, p.102826.
    """
    if len(input_shape) == 2: input_shape = tuple(input_shape) + (1, )
    inp_length, inp_channels = input_shape[:2]
    
    inputs = Input(shape = input_shape, name = 'input_eeg')
    regRate = 0.25

    numFilters = F1
    F2= numFilters*D
    
    EEGNet_sep = EEGNet(
        input_layer = inputs, F1 = F1, kernel_length = kernel_length, D = D, channels = inp_channels, dropout = dropout_eeg
    )
    block2 = Lambda(lambda x: x[:, :, -1, :])(EEGNet_sep)
    block2_flat = Flatten()(block2) 

    outs = TCN_block(
        input_layer = block2, input_dimension = F2, depth = layers, kernel_size = kernel_size,
        filters = filters, dropout = dropout, activation = activation
    )

    conv1 = Concatenate()([block2, outs]) 
    out   = Flatten()(conv1) 
    conv2 = Concatenate()([out, block2_flat]) 
    
    output = Dense(
        output_dim, name = final_name if final_activation is None else 'final_dense', kernel_constraint = max_norm_constraint(regRate)
    )(conv2)
    if final_activation is not None:
        output = CustomActivation(final_activation, name = final_name)(output)
    
    return tf.keras.Model(inputs = inputs, outputs = output, name = name)

#%% Reproduced EEGTCNet model: https://arxiv.org/abs/2006.00622
def EEGTCNet(input_shape = (1125, 22, 1),
             output_dim  = 4,
             layers      = 2,
             kernel_size = 4,
             filters     = 12,
             dropout     = 0.3,
             activation  = 'elu',
             
             F1  = 8,
             D   = 2,
             kernel_length = 32,
             dropout_eeg = 0.2,
             
             final_name       = 'output_layer',
             final_activation = 'softmax',
             
             name = 'EEGTCNet',
             ** kwargs
            ):
    """
        EEGTCNet model from Ingolfsson et al 2020.
        See details at https://arxiv.org/abs/2006.00622
    
        The original code for this model is available at https://github.com/iis-eth-zurich/eeg-tcnet
    
        Notes
        -----
        The initial values in this model are based on the values identified by the authors
        
        References
        ----------
        .. Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N.,
           Cavigelli, L., & Benini, L. (2020, October). 
           Eeg-tcnet: An accurate temporal convolutional network
           for embedded motor-imagery brain–machine interfaces. 
           In 2020 IEEE International Conference on Systems, 
           Man, and Cybernetics (SMC) (pp. 2958-2965). IEEE.
    """
    if len(input_shape) == 2: input_shape = tuple(input_shape) + (1, )
    inp_length, inp_channels = input_shape[:2]
    
    inputs = Input(shape = input_shape, name = 'input_eeg')
    regRate = 0.25

    numFilters = F1
    F2 = numFilters * D

    EEGNet_sep = EEGNet(
        input_layer = inputs, F1 = F1, kernel_length = kernel_length, D = D, channels = inp_channels, dropout = dropout_eeg
    )
    block2 = Lambda(lambda x: x[:, :, -1, :])(EEGNet_sep)
    outs = TCN_block(
        input_layer = block2, input_dimension = F2, depth = layers, kernel_size = kernel_size, filters = filters,
        dropout = dropout, activation = activation
    )
    out = Lambda(lambda x: x[:,-1,:])(outs)
    
    output = Dense(
        output_dim, name = final_name if final_activation is None else 'final_dense', kernel_constraint = max_norm_constraint(regRate)
    )(out)
    if final_activation is not None:
        output = CustomActivation(final_activation, name = final_name)(output)
    
    return tf.keras.Model(inputs = inputs, outputs = output, name = name)

#%% Reproduced EEGNeX model: https://arxiv.org/abs/2207.12369
def EEGNeX_8_32(input_shape = (1125, 22, 1),
                output_dim  = 4,
                
                final_activation = 'softmax',
                final_name       = 'output_layer',
                
                name = 'EEGNeX_8_32',
                ** kwargs
               ):
    """
        EEGNeX model from Chen et al 2022.
        See details at https://arxiv.org/abs/2207.12369

        The original code for this model is available at https://github.com/chenxiachan/EEGNeX
           
        References
        ----------
        .. Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2022).
           Toward reliable signals decoding for electroencephalogram: 
           A benchmark study to EEGNeX. arXiv preprint arXiv:2207.12369.
    """
    if len(input_shape) == 2: input_shape = tuple(input_shape) + (1, )
    inp_length, inp_channels = input_shape[:2]
    
    model = tf.keras.Sequential(name = name)
    model.add(Input(shape = input_shape))

    model.add(Conv2D(filters = 8, kernel_size = (1, 32), use_bias = False, padding='same'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters = 32, kernel_size = (1, 32), use_bias = False, padding='same'))
    model.add(LayerNormalization())
    model.add(Activation(activation = 'elu'))

    model.add(DepthwiseConv2D(
        kernel_size = (inp_channels, 1), depth_multiplier = 2, use_bias = False, depthwise_constraint = max_norm_constraint(1.)
    ))
    model.add(LayerNormalization())
    model.add(Activation(activation = 'elu'))
    model.add(AveragePooling2D(pool_size = (1, 4), padding = 'same'))
    model.add(Dropout(0.5))

    
    model.add(Conv2D(filters = 32, kernel_size = (1, 16), use_bias = False, padding = 'same', dilation_rate = (1, 2)))
    model.add(LayerNormalization())
    model.add(Activation(activation = 'elu'))
    
    model.add(Conv2D(filters = 8, kernel_size = (1, 16), use_bias = False, padding = 'same', dilation_rate = (1, 4)))
    model.add(LayerNormalization())
    model.add(Activation(activation = 'elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(output_dim, name = final_name if final_activation is None else 'final_dense', kernel_constraint = max_norm_constraint(0.25)))
    if final_activation is not None:
        model.add(CustomActivation(final_activation, name = final_name))

    return model

#%% Reproduced EEGNet model: https://arxiv.org/abs/1611.08024
def EEGNet_classifier(input_shape = (1125, 22, 1),
                      output_dim  = 4,
                      
                      F1 = 8,
                      D  = 2,
                      kernel_length = 64,
                      dropout_eeg   = 0.25,
                      
                      final_activation = 'softmax',
                      final_name       = 'output_layer',
                
                      name = 'EEGNet',
                      ** kwargs
                     ):
    if len(input_shape) == 2: input_shape = tuple(input_shape) + (1, )
    inp_length, inp_channels = input_shape[:2]
    
    inputs = Input(shape = input_shape, name = 'input_eeg')
    regRate = 0.25

    eegnet = EEGNet(
        input_layer = inputs, F1 = F1, kernel_length = kernel_length, D = D, channels = inp_channels, dropout = dropout_eeg
    )
    eegnet = Flatten()(eegnet)
    
    output = Dense(
        output_dim, name = final_name if final_activation is None else 'final_dense', kernel_constraint = max_norm_constraint(0.25)
    )(eegnet)
    if final_activation is not None:
        output = CustomActivation(final_activation, name = final_name)(output)

    return tf.keras.Model(inputs = inputs, outputs = output, name = name)

def EEGNet(input_layer, F1 = 8, kernel_length = 64, D = 2, channels = 22, dropout = 0.25, merge_method = 'gated_convlstm'):
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
    F2= F1*D
    block1 = Conv2D(F1, (kernel_length,  1), padding = 'same', use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)
    if channels:
        block2 = DepthwiseConv2D(
            (1, channels), use_bias = False, depth_multiplier = D, depthwise_constraint = max_norm_constraint(1.)
        )(block1)
        block2 = BatchNormalization(axis = -1)(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((8, 1))(block2)
        block2 = Dropout(dropout)(block2)
    elif merge_method:
        from.eeg_arch import ChannelMergingLayer
        block2 = block1
        for i in range(2):
            filters = 8 if i == 0 else 3
            block2 = Conv2D(
                F2, (1, filters), padding = 'same', use_bias = False,
                strides = (1, filters)
            )(block2)
            block2 = BatchNormalization(axis = -1)(block2)
            block2 = Activation('elu')(block2)
            block2 = Dropout(dropout)(block2)
        
        block2 = ChannelMergingLayer(merge_method, keepdims = True)(block2)
        if merge_method == 'convlstm':
            block2 = BatchNormalization(axis = -1)(block2)
            block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((8, 1))(block2)
        block2 = Dropout(dropout)(block2)
    else:
        block2 = block1
        block2 = AveragePooling2D((8, 1))(block2)
        block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(
        F2, (16, 1), use_bias = False, padding = 'same'
    )(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8,1))(block3)
    block3 = Dropout(dropout)(block3)
    return block3

#%% Reproduced DeepConvNet model: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
def DeepConvNet(input_shape = (128, 64, 1),
                output_dim  = 4,
                dropout     = 0.5,
                
                final_activation = 'softmax',
                final_name       = 'output_layer',

                name = 'DeepConvNet',
                ** kwargs
               ):
    """
        Tensorflow implementation of the Deep Convolutional Network as described in Schirrmeister et. al. (2017), Human Brain Mapping.
        See details at https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730

        The original code for this model is available at : https://github.com/braindecode/braindecode
    
        This implementation is taken from code by the Army Research Laboratory (ARL) 
        at https://github.com/vlawhern/arl-eegmodels

        This implementation assumes the input is a 2-second EEG signal sampled at 
        128Hz, as opposed to signals sampled at 250Hz as described in the original
        paper. We also perform temporal convolutions of length (1, 5) as opposed
        to (1, 10) due to this sampling rate difference. 

        Note that we use the max_norm constraint on all convolutional layers, as 
        well as the classification layer. We also change the defaults for the
        BatchNormalization layer. We used this based on a personal communication 
        with the original authors.

                          ours        original paper
        pool_size        1, 2        1, 3
        strides          1, 2        1, 3
        conv filters     1, 5        1, 10

        Note that this implementation has not been verified by the original authors. 
    """
    if len(input_shape) == 2: input_shape = tuple(input_shape) + (1, )
    inp_length, inp_channels = input_shape[:2]
    
    inputs = Input(shape = input_shape, name = 'input_eeg')
    
    block1       = Conv2D(25, (1, 5), kernel_constraint = max_norm_constraint(2., axis = (0, 1, 2)))(inputs)
    block1       = Conv2D(25, (inp_channels, 1), kernel_constraint = max_norm_constraint(2., axis=(0, 1, 2)))(block1)
    block1       = BatchNormalization(epsilon = 1e-05, momentum = 0.9)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size = (1, 2), strides = (1, 2))(block1)
    block1       = Dropout(dropout)(block1)
  
    block2       = Conv2D(50, (1, 5), kernel_constraint = max_norm_constraint(2., axis=(0, 1, 2)))(block1)
    block2       = BatchNormalization(epsilon = 1e-05, momentum = 0.9)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size = (1, 2), strides = (1, 2))(block2)
    block2       = Dropout(dropout)(block2)
    
    block3       = Conv2D(100, (1, 5), kernel_constraint = max_norm_constraint(2., axis=(0, 1, 2)))(block2)
    block3       = BatchNormalization(epsilon = 1e-05, momentum = 0.9)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size = (1, 2), strides = (1, 2))(block3)
    block3       = Dropout(dropout)(block3)
    
    block4       = Conv2D(200, (1, 5), kernel_constraint = max_norm_constraint(2., axis=(0, 1, 2)))(block3)
    block4       = BatchNormalization(epsilon = 1e-05, momentum = 0.9)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size = (1, 2), strides = (1, 2))(block4)
    block4       = Dropout(dropout)(block4)
    
    flatten      = Flatten()(block4)
    
    output       = Dense(
        output_dim, name = final_name if final_activation is None else 'final_dense', kernel_constraint = max_norm_constraint(0.25)
    )(flatten)
    if final_activation is not None:
        output = CustomActivation(final_activation, name = final_name)(output)

    return tf.keras.Model(inputs = inputs, outputs = output, name = name)

#%% need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   

#%% Reproduced ShallowConvNet model: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
def ShallowConvNet(input_shape = (128, 64, 1),
                   output_dim  = 4,
                   dropout     = 0.5,
                
                   final_activation = 'softmax',
                   final_name       = 'output_layer',

                   name = 'ShallowConvNet',
                   ** kwargs
                  ):
    """
        Tensorflow implementation of the Shallow Convolutional Network as described
        in Schirrmeister et. al. (2017), Human Brain Mapping.
        See details at https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730

        The original code for this model is available at:
            https://github.com/braindecode/braindecode

        This implementation is taken from code by the Army Research Laboratory (ARL) 
        at https://github.com/vlawhern/arl-eegmodels

        Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
        the original paper, they do temporal convolutions of length 25 for EEG
        data sampled at 250Hz. We instead use length 13 since the sampling rate is 
        roughly half of the 250Hz which the paper used. The pool_size and stride
        in later layers is also approximately half of what is used in the paper.

        Note that we use the max_norm constraint on all convolutional layers, as 
        well as the classification layer. We also change the defaults for the
        BatchNormalization layer. We used this based on a personal communication 
        with the original authors.

                         ours        original paper
        pool_size        1, 35       1, 75
        strides          1, 7        1, 15
        conv filters     1, 13       1, 25    

        Note that this implementation has not been verified by the original 
        authors. We do note that this implementation reproduces the results in the
        original paper with minor deviations. 
    """

    if len(input_shape) == 2: input_shape = tuple(input_shape) + (1, )
    inp_length, inp_channels = input_shape[:2]
    inputs = Input(shape = input_shape, name = 'input_eeg')

    block1       = Conv2D(40, (1, 13), kernel_constraint = max_norm_constraint(2., axis=(0, 1, 2)))(inputs)
    block1       = Conv2D(
        40, (inp_channels, 1), use_bias = False, kernel_constraint = max_norm_constraint(2., axis=(0, 1 ,2))
    )(block1)
    block1       = BatchNormalization(epsilon = 1e-05, momentum = 0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size = (1, 35), strides = (1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropout)(block1)
    flatten      = Flatten()(block1)
    
    output       = Dense(
        output_dim, name = final_name if final_activation is None else 'final_dense', kernel_constraint = max_norm_constraint(0.5)
    )(flatten)
    if final_activation is not None:
        output = CustomActivation(final_activation, name = final_name)(output)

    return tf.keras.Model(inputs = inputs, outputs = output, name = name)


def ConvBlockV2(input_layer,
                filters = 4,
                kernel_size = 64,
                pool_size   = 8,
                D   = 2,
                weight_decay    = 0.009,
                max_norm    = 0.6,
                dropout = 0.25
               ):
    """ Conv_block
    
        Notes
        -----
        using  different regularization methods.
    """
    
    block1 = Conv2D(
        filters,
        (kernel_size, 1),
        padding = 'same',
        kernel_regularizer = L2(weight_decay),
                    
        # In a Conv2D layer with data_format="channels_last", the weight tensor has shape 
        # (rows, cols, input_depth, output_depth), set axis to [0, 1, 2] to constrain 
        # the weights of each filter tensor of size (rows, cols, input_depth).
        kernel_constraint = max_norm_constraint(max_norm, axis = [0,1,2]),
        use_bias = False
    )(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)  # bn_axis = -1 if data_format() == 'channels_last' else 1
    
    block2 = DepthwiseConv2D(
        (1, block1.shape[-2]),  
        depth_multiplier = D,
        depthwise_regularizer = L2(weight_decay),
        depthwise_constraint  = max_norm_constraint(max_norm, axis=[0,1,2]),
        use_bias = False
    )(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1))(block2)
    block2 = Dropout(dropout)(block2)
    
    block3 = Conv2D(
        filters * D,
        kernel_size = (16, 1),
        kernel_regularizer  = L2(weight_decay),
        kernel_constraint   = max_norm_constraint(max_norm, axis=[0,1,2]),
        use_bias    = False,
        padding = 'same'
    )(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    
    block3 = AveragePooling2D((pool_size, 1))(block3)
    block3 = Dropout(dropout)(block3)
    return block3

def TCNBlockV2(input_layer,
               depth,
               kernel_size,
               filters,
               dropout,
               
               weight_decay = 0.009,
               max_norm = 0.6,
               activation = 'relu'
              ):
    """ TCN_block from Bai et al 2018
        Temporal Convolutional Network (TCN)
        
        Notes
        -----
        using different regularization methods
    """    
    
    block = Conv1D(
        filters,
        kernel_size = kernel_size,
        dilation_rate   = 1,
        kernel_initializer  = 'he_uniform',
        kernel_regularizer  = L2(weight_decay),
        kernel_constraint   = max_norm_constraint(max_norm, axis=[0,1]),
        padding = 'causal'
    )(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(
        filters,
        kernel_size = kernel_size,
        dilation_rate   = 1,
        kernel_initializer  = 'he_uniform',
        kernel_regularizer  = L2(weight_decay),
        kernel_constraint   = max_norm_constraint(max_norm, axis=[0,1]),
        padding = 'causal'
    )(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_layer.shape[-1] != filters):
        conv = Conv1D(
            filters,
            padding = 'same',
            kernel_size = 1,
            kernel_regularizer  = L2(weight_decay),
            kernel_constraint   = max_norm_constraint(max_norm, axis=[0,1]),
        )(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)
    
    for i in range(depth - 1):
        block = Conv1D(
            filters,
            padding = 'causal',
            kernel_size = kernel_size,
            dilation_rate   = 2 ** (i + 1),
            kernel_initializer  = 'he_uniform',
            kernel_regularizer  = L2(weight_decay),
            kernel_constraint   = max_norm_constraint(max_norm, axis=[0,1])
        )(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(
            filters,
            padding = 'causal',
            kernel_size = kernel_size,
            dilation_rate   = 2 ** (i + 1),
            kernel_initializer  = 'he_uniform',
            kernel_regularizer  = L2(weight_decay),
            kernel_constraint   = max_norm_constraint(max_norm, axis=[0,1])
        )(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
        
    return out

#%% Convolutional (CV) block used in the ATCNet model
def Conv_block(input_layer, F1 = 4, kernel_length = 64, pool_size = 8, D = 2, channels = 22, dropout = 0.1):
    """ Conv_block
    
        Notes
        -----
        This block is the same as EEGNet with SeparableConv2D replaced by Conv2D 
        The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
        See details at https://arxiv.org/abs/1611.08024
    """
    F2= F1*D
    block1 = Conv2D(F1, (kernel_length, 1), padding = 'same', use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)
    block2 = DepthwiseConv2D(
        (1, channels), use_bias = False, depth_multiplier = D, depthwise_constraint = max_norm_constraint(1.)
    )(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8,1))(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1), use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    
    block3 = AveragePooling2D((pool_size, 1))(block3)
    block3 = Dropout(dropout)(block3)
    return block3

custom_functions   = {
    'ATCNet'        : ATCNet,
    'ATCNetV2'      : ATCNetV2,
    'EEGNet'        : EEGNet_classifier,
    'EEGTCNet'      : EEGTCNet,
    'EEGNeX'        : EEGNeX_8_32,
    'TCNetFusion'   : TCNet_Fusion,
    'TCNet_Fusion'  : TCNet_Fusion,
    'DeepConvNet'   : DeepConvNet,
    'ShallowConvNet'    : ShallowConvNet,
    'EEGSimpleConv'     : EEGSimpleConv
}

custom_objects = {
    'CustomActivation'   : CustomActivation
}