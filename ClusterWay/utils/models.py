# Copyright 2022 Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
from utils.train import Trainer
from utils.tools import deepWayLoss

class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape

    
    
def channel_attention(x, filters, ratio=8):
    
    shared_layer_one = tf.keras.layers.Dense(filters//ratio,
                             activation='relu',
                            kernel_initializer='he_normal',
                             use_bias=True,
                                bias_initializer='zeros')
    shared_layer_two = tf.keras.layers.Dense(filters,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)    
    avg_pool = tf.keras.layers.Reshape((1,1,filters))(avg_pool)

    avg_pool = shared_layer_one(avg_pool)

    avg_pool = shared_layer_two(avg_pool)


    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    max_pool = tf.keras.layers.Reshape((1,1,filters))(max_pool)

    max_pool = shared_layer_one(max_pool)

    max_pool = shared_layer_two(max_pool)


    cbam_feature = tf.keras.layers.Add()([avg_pool,max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return tf.keras.layers.Multiply()([x, cbam_feature])



def spatial_attention(x, kernel_size = 7):

    avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(x)

    max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(x)

    attention = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])

    attention = tf.keras.layers.Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(attention)

    return tf.keras.layers.multiply([x, attention])



def red_module(x, filters, kernel_size):
    
    x_res = x
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same', activation=None)(x)
    x = Mish()(x)
    x_res = tf.keras.layers.Conv2D(filters, 1, strides=2, padding='same', activation=None)(x_res)
    x_res = Mish()(x_res)
    
    return x_res + x
 
    
    
def resa_red_module(x, filters,  kernel_size):
    
    x_res = x
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same', activation=None)(x)
    x = Mish()(x)
    x = channel_attention(x, filters)
    x = spatial_attention(x, kernel_size = 7)
    x = x + x_res
    return red_module(x, filters, kernel_size)



def resa_module(x, filters,  kernel_size):
    
    x_res = x
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same', activation=None)(x)
    x = Mish()(x)
    x = channel_attention(x, filters)
    x = spatial_attention(x, kernel_size = 7)
    x = x + x_res
    return x


class padToShape(tf.keras.layers.Layer):
    ''' Works only with H and W dims, not channel and batch'''
    def call(self, inputs, shape, **kwargs):
        to_pad = (shape[1:-1] - tf.shape(inputs)[1:-1])/2
        to_pad_pre = tf.math.ceil(to_pad)
        to_pad_post = tf.math.floor(to_pad)
        paddings = [[0,0],[to_pad_pre[0],to_pad_post[0]],[to_pad_pre[1],to_pad_post[1]],[0,0]]
        return tf.pad(inputs, paddings)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]



####################################################################################
#                                    CLASSIC
####################################################################################

def pred_module(x):
    
    x = tf.keras.layers.Conv2D(3, 1, strides=1, padding='same', activation= None)(x)
    
    x_0 = tf.keras.layers.Lambda(lambda x: x[...,0:1])(x)
    x_1 = tf.keras.layers.Lambda(lambda x: x[...,1:])(x)
    
    x_0 = tf.keras.layers.Activation('sigmoid')(x_0)
    x_1 = tf.keras.layers.Activation('tanh')(x_1)
    
    return tf.keras.layers.Concatenate()([x_0, x_1])


def build_deepway(name_model, filters = 32, kernel_size = 3, N = 2, MASK_DIM = 800,
                          high_level_features_as_output = False):
    
    input_tensor = tf.keras.layers.Input(shape=(MASK_DIM,MASK_DIM))
    x = input_tensor
    
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(x)
    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=(-1)))(x)
    
    # first compression
    x = tf.keras.layers.Conv2D(filters, 7, strides=2, padding='same', activation=None)(x)
    x = Mish()(x)
    
    # main corpus
    for i in range(N):
        x = resa_red_module(x, filters=filters, kernel_size=kernel_size)
        
    # downsampling
    x_down = resa_red_module(x, filters=filters, kernel_size=kernel_size)  
    x_up = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=(2,2), padding='same', activation=None)(x_down)
    x_up = Mish()(x_up)
    x = tf.keras.layers.Concatenate()([x_up, x])
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same', activation=None)(x)
    x = Mish()(x) #these are the high level features
    
    #output
    output_tensors = {'mask': pred_module(x)}
    
    if high_level_features_as_output:
        output_tensors['hl_features'] = x
    return tf.keras.Model(input_tensor, output_tensors, name=name_model)


####################################################################################
#                                   CLUSTER
####################################################################################


def build_clusterway(name_model, model_classic, filters = 32, kernel_size = 3,
                          MASK_DIM = 800, out_feats = None, high_level_features_as_output = False):
    
    for l in model_classic.layers:
        l.trainable = False

    # second head
    x = model_classic.output['hl_features']
    h2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    h2 = Mish()(h2)
    h2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(h2)
    h2 = Mish()(h2)
    x_feats = tf.keras.layers.Conv2D(out_feats, 1, padding='same', name='features')(h2)

    output_tensors = {'features': x_feats, 'mask': model_classic.output['mask']}
    if high_level_features_as_output:
        output_tensors['hl_features'] = model_classic.output['hl_features']
    return tf.keras.Model(model_classic.input, output_tensors, name=name_model)


