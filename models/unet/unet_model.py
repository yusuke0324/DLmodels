# simple U-net model
# paper: https://arxiv.org/abs/1505.04597

from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, Concatenate
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np


def unet_model(input_shape=(128, 128, 3), min_filter_num=16, kernel_size=(3, 3), up_down_size=(2, 2), strides=(1, 1), activation='elu', offset=2, kernel_initializer='he_normal'):
    '''
    U-net model. Encoder - Decoder with skipped connections.
    Arguments
    ------------------
        input_shape: input shape of image (h, w, channel)
        min_filter: the minimum number of filters in each convolution layer
        kernel_size: kernel size of filters
        up_down_size: up and down sampling size
        strides: strides size of each conv
        activation: activation function of each conv
        offset: the number of encoder layers should be conv num(log2_shape) - offset
    Returns
    ------------------
        unet_model, keras model
    '''

    # a num of conv = log_2(im_height or width)
    # log_a(X) = log_e(X) / log_e(a)
    # use 'e' to compute
    min_shape = min(input_shape[:-1])
    conv_num = int(np.floor(np.log(min_shape)/np.log(2))) - offset

    # filter number list; ex) [64, 128, 256, 512, 512, 512....]
    filter_nums = [min_filter_num*min(2**i, 16) for i in range(conv_num)]

    # Input layer
    input_l = Input(shape=input_shape, name='input_layer')

    # first encoder
    first_encoder = _encoder_block(input_l, filter_nums[0], strides=strides, kernel_size=kernel_size, dropout=0.1, name='encoder_block_1', activation=activation)
    x = MaxPooling2D(up_down_size)(first_encoder)

    # make the rest of encoders
    encoders = [first_encoder]

    for i, filter_num in enumerate(filter_nums[1:]):
        x = _encoder_block(x, filter_num, name='encoder_block_'+str(i+2), strides=strides, kernel_size=kernel_size, dropout=0.1, activation=activation)
        encoders.append(x)
        if not i == (len(filter_nums) - 2):
            x = MaxPooling2D(up_down_size)(x)

    # Decoders
    # revers filter nums for decoders
    # do not use the last filter num
    decoder_filter_nums = filter_nums[::-1][1:]

    for i, filter_num in enumerate(decoder_filter_nums):
        # [NOTE] first decoder is concated with the second last encoders!
        x = _decoder_block(x, encoders[-(i+2)], decoder_filter_nums[i], name='decoder_block_'+str(i+1), strides=strides, kernel_size=kernel_size, dropout=0.1, activation=activation)

    # for segmentation, apply sigmoid to every pixel
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=[input_l], outputs=[outputs])

    return model





def _encoder_block(x, filter_num, name='encoder_block', strides=(1, 1), kernel_size=(3, 3), dropout=0.1, activation='elu', down_size=(2, 2), kernel_initializer='he_normal'):
    '''
    U-net encoder cnn block
    Conv -> activation -> dropout -> conv -> activation
    '''

    x = Conv2D(filter_num, kernel_size, strides=strides, name=name, activation=activation, padding='same', kernel_initializer=kernel_initializer)(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filter_num, kernel_size, strides=strides, activation=activation, padding='same', kernel_initializer=kernel_initializer)(x)
    # out = MaxPooling2D(down_size)(x)

    return x

def _decoder_block(x, skip_connect, filter_num, name='decoder_block', strides=(1, 1), kernel_size=(3, 3), dropout=0.1, activation='elu', up_size=(2, 2), kernel_initializer='he_normal'):
    '''
    U-net decoder cnn block
    transpose conv -> concat skip_connect -> conv -> dropout -> conv
    '''

    x = Conv2DTranspose(filter_num, up_size, strides=up_size, padding='same', activation=activation, name=name)(x)
    x = Concatenate()([x, skip_connect])
    x = Conv2D(filter_num, kernel_size, strides=strides, padding='same', activation=activation, kernel_initializer=kernel_initializer)(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filter_num, kernel_size, strides=strides, padding='same', activation=activation, kernel_initializer=kernel_initializer)(x)

    return x