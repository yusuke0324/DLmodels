from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np

def crop_patch_cnn(input_shape=(512, 512, 3), patch_size=32, cls_num=5, **kwargs):

    '''
    crop patch-based CNN
    Arguments
    ---------------
        input_shape: shape of input image, (h, w, channel)
        patch_size: pix size of patches, integer
        cls_num: class number
        **kwargs:
            min_filter_num -> 64
            kernel_size -> (3, 3)
            strides -> (2, 2)
            a -> 0.2
            dropout -> 0.5
    Returns
    ----------------
        keras model, crop_patch_cnn
    '''

    min_filter_num = kwargs.get('min_filter_num', 64)
    kernel_size = kwargs.get('kernel_size', (3, 3))
    strides = kwargs.get('strides', (2, 2))
    a = kwargs.get('a', 0.2)
    dropout = kwargs.get('dropout', 0.5)

    patch_shape = (patch_size, patch_size, input_shape[-1])
    input_layer = Input(shape=input_shape, name='input')
    patch_num = (input_shape[0]//patch_size) * (input_shape[1]//patch_size)

    y_list = [(i*patch_size, (i+1)*patch_size) for i in range(input_shape[0]//patch_size)]
    x_list = [(i*patch_size, (i+1)*patch_size) for i in range(input_shape[1]//patch_size)]

    crop_layers = []
    for y in y_list:
        for x in x_list:
            crop_layer = Lambda(lambda z: z[:, y[0]:y[1], x[0]:x[1], :])(input_layer)
            crop_layers.append(crop_layer)

    min_shape = min(input_shape[:-1])
    conv_num = int(np.floor(np.log(patch_size)/np.log(2)))
    filter_nums = [min_filter_num*min(2**i, 8) for i in range(conv_num)]
#     first patch conv
    patch_input = Input(shape=patch_shape, name='patch_input')
    x = Conv2D(filter_nums[0], kernel_size, strides=strides, name='conv_1')(patch_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(a)(x)

    for i, filter_num in enumerate(filter_nums[1:]):
        name = 'conv_' + str(i+2)
        x = Conv2D(filter_num, kernel_size, strides=strides, name=name, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(a)(x)

    x_flat = Flatten()(x)
    x_flat = Dropout(dropout)(x_flat)
    x = Dense(12, activation='tanh', name='dense')(x_flat)

    patch_net = Model(inputs=patch_input, outputs=x, name='patch_net')
    print('patch net summary')
    patch_net.summary()

#     patch input layers from crop layers

    patch_input_layers = [Input(shape=patch_shape, name='patch_input_'+str(i)) for i in range(patch_num)]

    all_patch_nets = [patch_net([patch_input_layers[i]]) for i in range(patch_num)]

    if len(all_patch_nets) > 1:
        x = Concatenate()(all_patch_nets)
    else:
        x = all_patch_nets[0]


    patch_cnn = Model(inputs=patch_input_layers, outputs=x, name='patch_cnn')
    crop_patch_out = patch_cnn(crop_layers)
    crop_patch_out = Dropout(dropout)(crop_patch_out)

    x_out = Dense(cls_num, activation='softmax', name='output')(crop_patch_out)
    # x_out = x_out(patch_input_layers)
    # x_out = x_out(crop_layers)

    # crop_patch_cnn = Model(inputs=input_layer, outputs=crop_patch_out, name='crop_patch_cnn')
    crop_patch_cnn = Model(inputs=input_layer, outputs=x_out, name='crop_patch_cnn')

    return crop_patch_cnn