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

import math
import tensorflow as tf

from tensorflow.keras.layers import *

#%% Create and apply the attention model
def attention_block(net, attention_model):
    shape = net.shape

    if attention_model == 'mha':   # Multi-head self attention layer 
        if len(shape) > 3: net = Reshape((shape[1], -1))(net)
        net = mha_block(net, vanilla = True)
    elif attention_model == 'mhla':  # Multi-head local self-attention layer 
        if len(shape) > 3: net = Reshape((in_sh[1], -1))(net)
        net = mha_block(net, vanilla = False)
    elif attention_model == 'se':   # Squeeze-and-excitation layer
        if len(shape) < 4: net = tf.expand_dims(net, axis = 3)
        net = se_block(net, ratio = 8)
    elif attention_model == 'cbam': # Convolutional block attention module
        if len(shape) < 4: net = tf.expand_dims(net, axis = 3)
        net = cbam_block(net, ratio = 8)
    else:
        raise ValueError("'{}' is not supported attention module!".format(attention_model))
    
    if len(shape) == 3 and len(net.shape) == 4:
        net = tf.squeeze(net, axis = 3)
    elif len(shape) == 4 and len(net.shape) == 3:
        net = Reshape((shape[1], shape[2], shape[3]))(net)
    
    return net

#%% Multi-head self Attention (MHA) block
def mha_block(input_feature, key_dim = 8, num_heads = 2, dropout = 0.5, vanilla = True):
    """
        Multi Head self Attention (MHA) block.     

        Here we include two types of MHA blocks: 
        The original multi-head self-attention as described in https://arxiv.org/abs/1706.03762
        The multi-head local self attention as described in https://arxiv.org/abs/2112.13492v1
    """    
    # Layer normalization
    x = LayerNormalization(epsilon = 1e-6)(input_feature)
    
    if vanilla:
        # Create a multi-head attention layer as described in 
        # 'Attention Is All You Need' https://arxiv.org/abs/1706.03762
        x = MultiHeadAttention(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(x, x)
    else:
        # Create a multi-head local self-attention layer as described in 
        # 'Vision Transformer for Small-Size Datasets' https://arxiv.org/abs/2112.13492v1
        
        # Build the diagonal attention mask
        NUM_PATCHES = input_feature.shape[1]
        diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
        diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)
        
        # Create a multi-head local self attention layer.
        x = MultiHeadAttention_LSA(
            key_dim = key_dim, num_heads = num_heads, dropout = dropout
        )(x, x, attention_mask = diag_attn_mask)
        
    x = Dropout(0.3)(x)
    # Skip connection
    mha_feature = Add()([input_feature, x])
    
    return mha_feature


#%% Multi head self Attention (MHA) block: Locality Self Attention (LSA)
class MultiHeadAttention_LSA(tf.keras.layers.MultiHeadAttention):
    """
        local multi-head self attention block

        Locality Self Attention as described in https://arxiv.org/abs/2112.13492v1
        This implementation is taken from  https://keras.io/examples/vision/vit_small_ds/ 
    """    
    def __init__(self, ** kwargs):
        super().__init__(** kwargs)
        # The trainable temperature term. The initial value is the square 
        # root of the key dimension.
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable = True)

    def _compute_attention(self, query, key, value, attention_mask = None, training = None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


#%% Squeeze-and-excitation block
def se_block(input_feature, ratio = 8):
    """
        Squeeze-and-Excitation(SE) block.

        As described in https://arxiv.org/abs/1709.01507
        The implementation is taken from https://github.com/kobiso/CBAM-keras
    """
    channel = input_feature.shape[-1]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    
    se_feature = Dense(
        channel // ratio,
        activation = 'relu',
        kernel_initializer = 'he_normal',
        use_bias = True,
        bias_initializer = 'zeros'
    )(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel//ratio)
    
    se_feature = Dense(
        channel,
        activation = 'sigmoid',
        kernel_initializer = 'he_normal',
        use_bias = True,
        bias_initializer = 'zeros'
    )(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


#%% Convolutional block attention module
def cbam_block(cbam_feature, ratio=8):
    """
        Convolutional Block Attention Module(CBAM) block.

        As described in https://arxiv.org/abs/1807.06521
        The implementation is taken from https://github.com/kobiso/CBAM-keras
    """
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio = 8):
    # 	channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[-1]
    
    shared_layer_one = Dense(
        channel // ratio,
        activation = 'relu',
        kernel_initializer = 'he_normal',
        use_bias = True,
        bias_initializer = 'zeros'
    )
    shared_layer_two = Dense(
        channel,
        kernel_initializer = 'he_normal',
        use_bias = True,
        bias_initializer = 'zeros'
    )
	
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)

    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel//ratio)

    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
	
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, kernel_size = 7):
    channel = input_feature.shape[-1]
    cbam_feature = input_feature
	
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis = 3, keepdims = True))(cbam_feature)
    assert avg_pool.shape[-1] == 1

    max_pool = Lambda(lambda x: tf.reduce_max(x, axis = 3, keepdims = True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    
    concat = Concatenate(axis = 3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    
    cbam_feature = Conv2D(
        filters = 1,
        kernel_size = kernel_size,
        strides = 1,
        padding = 'same',
        activation = 'sigmoid',
        kernel_initializer = 'he_normal',
        use_bias = False
    )(concat)	
    assert cbam_feature.shape[-1] == 1
	
    return multiply([input_feature, cbam_feature])
