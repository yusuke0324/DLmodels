# -*- coding: utf-8 -*-

# Xception
#paper:
#https://arxiv.org/abs/1610.02357

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers import Input, GlobalMaxPool2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras import layers
import keras.backend as K


def xception_model(input_shape=(256, 256, 3), cls_num=5, use_bias=False, middle_flow_iter=8):
    '''
     xception model. This model follows the original paper and has entry, middle and exit flow.
    Arguments
    --------------
        inpute_shape: input shape for the images, (h, w, channel)
        cls_num: class number
        use_bias: if the each cnn use bias
        middle_flow_iter: how many times middle flow is iterated. Use this parameter for adjusting based on your resource.
    Return
    ---------------
        Xception model.
    '''

#     input_block
    input_l = Input(shape=input_shape, name='input')
    x = Conv2D(32, (3, 3), strides=(2, 2), name='conv_1', use_bias=use_bias)(input_l)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), name='conv_2', use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

#     entry_flow
    x = entry_flow_resblock(x, first_activation=False, filter_num=128, name='entry_flow_resblock1_')
    x = entry_flow_resblock(x, first_activation=True, filter_num=256, name='entry_flow_resblock2_')
    x = entry_flow_resblock(x, first_activation=True, filter_num=728, name='entry_flow_resblock3_')

#     middle_flow
    for i in range(middle_flow_iter):
        x = middle_flow_resblock(x, name='middle_flow_resblock'+str(i+1)+'_')

#     exit flow
    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False,name='exit_flow_sepconv1')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='exit_flow_sepconv2')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='exit_flow_maxpool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='exit_flow_sepconv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='exit_flow_sepconv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(cls_num, activation='softmax', name='predictions')(x)

    model = Model(input_l, x, name='xception')
    return model

def entry_flow_resblock(x, first_activation=True, filter_num=128, kernel_size=(3, 3), use_bias=False, name='entry_flow_resblock_'):

    residual = Conv2D(filter_num, (1, 1), strides=(2, 2), use_bias=use_bias)(x)
    residual = BatchNormalization()(residual)

    if first_activation:
        x = Activation('relu')(x)

    x = SeparableConv2D(filter_num, kernel_size, padding='same', use_bias=use_bias, name=name + 'sepconv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filter_num, kernel_size, padding='same', use_bias=use_bias, name=name + 'sepconv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(kernel_size, strides=(2, 2), padding='same', name=name+'maxpool')(x)
    x = layers.add([x, residual])

    return x

def middle_flow_resblock(x, filter_num=728, kernel_size=(3, 3), name='middle_flow_resblock_', conv_num=3, use_bias=False):

    residual = x

    for i in range(conv_num):
        x = Activation('relu')(x)
        x = SeparableConv2D(filter_num, kernel_size,padding='same', name=name+'sepconv'+str(i+1), use_bias=use_bias)(x)
        x = BatchNormalization()(x)

    x = layers.add([x, residual])

    return x