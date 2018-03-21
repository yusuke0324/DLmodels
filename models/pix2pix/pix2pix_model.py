# This scripts are for components of pix2pix network architectures
# paper: https://phillipi.github.io/pix2pix/
# this implementation is for channel_last (h, w, channel_num) like tensorflow supports

from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np

def make_dcgan(generator, discriminator, img_shape, patch_size, name='dcgan'):
    '''
    DCGAN; generator -> crop layers -> discriminator
    Arguments
    --------------
        generator: generator model
        discriminator: discriminator model, which accepets patches
        patch_size: size of a patch, integer
        name: model name
    Returns
    --------------
        DCGAN model
    '''

    # in learning DCGAN, do not update discriminator
    discriminator.trainable = False

    # this tensor's shape should be (batchnum, h, w, channel)
    raw_input = Input(shape=img_shape, name='dcgan_input')
    # it doesn't mean that the image IS actually generated here!!
    # this is just adding layers
    generated_layer = generator(raw_input)

    img_h, img_w = img_shape[:-1]
    patch_h, patch_w = patch_size, patch_size

    y_list = [(i*patch_h, (i+1)*patch_h) for i in range(img_h//patch_h)]
    x_list = [(i*patch_w, (i+1)*patch_w) for i in range(img_w//patch_w)]

    raw_crop_layers = []
    gen_crop_layers = []

    for y in y_list:
        for x in x_list:
            # use Lambda layer to make them INPUT for discriminator
            # this layer is crop layer
            raw_crop_l = Lambda(lambda z: z[:, y[0]:y[1], x[0]:x[1], :])(raw_input)
            generated_crop_l = Lambda(lambda z: z[:, y[0]:y[1], x[0]:x[1], :])(generated_layer)
            # append list
            raw_crop_layers.append(raw_crop_l)
            gen_crop_layers.append(generated_crop_l)

    dcgan_output = discriminator(raw_crop_layers+gen_crop_layers)
    # in order to enable generated image visible, output generated_layer
    dcgan = Model(inputs=[raw_input], outputs=[generated_layer, dcgan_output], name=name)

    return dcgan


def make_discriminator(gen_input_patch_shape, raw_input_patch_shape, patch_num, min_filter_num=64, kernel_size=(3,3), strides=(2,2), a=0.2, model_name='patch_discriminator'):
    '''
    discriminator based on patch GAN
    Arguments
    --------------
        gen_input_patch_shape: shape of patch input from cropped generator output
        raw_input_patch_shape: shape of raw image patches
        patch_num: the number of patches
        min_filter_num: minimum number of filers of a convolution layer
        kernel_size: kernel size of filters in a conv
        strides: stride size of filters in a conv
        a: a in LeakyReLU, y = max(ax, x)
        model_name: model name
    Returns
    ---------------
        discriminator model
    '''

    # make input layer for discriminator and raw * patch_num
    disc_input_layers = [Input(shape=gen_input_patch_shape, name='gen_input_l_'+str(i)) for i in range(patch_num)]
    raw_input_layers = [Input(shape=raw_input_patch_shape, name='raw_input_l_'+str(i)) for i in range(patch_num)]

    # a num of conv = log_2(im_height or width)
    # log_a(X) = log_e(X) / log_e(a)
    # use 'e' to compute
    min_shape = min(gen_input_patch_shape[:-1])
    conv_num = int(np.floor(np.log(min_shape)/np.log(2)))

    # filter number list; ex) [64, 128, 256, 512, 512, 512....]
    filter_nums = [min_filter_num*min(2**i, 8) for i in range(conv_num)]

    # First Conv
    # for generated patches
    gen_patch_input = Input(shape=gen_input_patch_shape, name='generated_patch_input')
    x_g = Conv2D(filter_nums[0], kernel_size, strides=strides, padding='same', name='gen_conv_1')(gen_patch_input)
    x_g = BatchNormalization()(x_g)
    x_g = LeakyReLU(a)(x_g)

    # for raw input
    raw_patch_input = Input(shape=raw_input_patch_shape, name='raw_patch_input')
    x_r = Conv2D(filter_nums[0], kernel_size, strides=strides, padding='same', name='raw_conv_1')(raw_patch_input)
    x_r = BatchNormalization()(x_r)
    x_r = LeakyReLU(a)(x_r)

    x = Concatenate()([x_g, x_r])

    # Conv
    # add all convolution layers by the number of filters
    for i, filter_num in enumerate(filter_nums):
        name = 'conv_' + str(i+2)
        x = Conv2D(filter_num, kernel_size, strides=strides, name=name, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(a)(x)

    x_flat = Flatten()(x)
    # [TBC] the number of units, 2? or 1?
    x = Dense(2, activation='softmax', name='dense')(x_flat)

    patch_gan = Model(inputs=[gen_patch_input, raw_patch_input], outputs=[x], name='PatchGAN')
    print('PatchGAN summary:')
    patch_gan.summary()

    #list of patch gan for all patches
    all_patch_gans = [patch_gan([disc_input_layers[i], raw_input_layers[i]]) for i in range(patch_num)]

    # concat all patch gans
    if len(all_patch_gans)>1:
        x = Concatenate()(all_patch_gans)
    else:
        x = all_patch_gans[0]

    x_out = Dense(2, activation='softmax', name='disc_output')(x)
    discriminator = Model(inputs=disc_input_layers+raw_input_layers, outputs=[x_out], name='discriminator')

    return discriminator

def _generator_encoder_block(x, filter_num, name, strides=(2,2), kernel_size=(3,3), a=0.2):
    '''
    This is a encoder block of a generator.
    Basic architechture is the following:
        LeakyReLU -> Convolution(3,3) -> batchnormalization
    Because the block is supposed to follow a convlolution layer, it starts with activation function.
    Arguments
    --------------
        x: keras tensor
        filter_num: filter number of the convolution layer
        name: block name
        strides: size of strides for the convolution layer
        kernel_size: kernel size of the filters in the convlution layer
        a: a in LeakyReLU, y = max(ax, x)
        a: a in LeakyReLU, y = max(ax, x)
    Returns
    --------------
        x: constructed block as a keras tensor
    '''
    x = LeakyReLU(a)(x)
    x = Conv2D(filter_num, kernel_size, strides=strides, name=name, padding='same')(x)
    x = BatchNormalization()(x)
    return x

def _generator_decoder_block(x, skip_connect, filter_num, name, upsample_size=(2,2), kernel_size=(3,3), activation='relu', dropout=0.5):
    '''
    This is a decoder of a generator.
    Basic architechture is the following u-net architecture:
        Activation -> Upsampling -> Convolution -> Batch Normalization -> dropout -> concat to skip connection
    Arguments
    -------------
        x: keras tensor
        skip_connect: keras tensor, this should be the layer of same lavel encoder block.
        filter_num: filter number of the convolution layer
        name: block name
        upsample_size: because of decoder. the layer need to be 'fractionally-strided convolutions'. in this block, just using Upsample2D in keras.
        kernel_size: kernel size of the filters in the convlution layer
        activation: string of activation function
        dropout: ratio of dropout
    Returns
    -------------
        x: constructed block as a keras tensor
    '''
    x = Activation(activation)(x)
    x = UpSampling2D(size=upsample_size)(x)
    x = Conv2D(filter_num, kernel_size, name=name, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    # skip connection
    x = Concatenate()([x, skip_connect])
    return x

def make_generator(in_shape, out_shape, model_name='unet_generator', min_filter_num=64, kernel_size=(3,3), strides=(2,2), a=0.2, dropout=0.5, decoder_act='relu'):
    '''
    Make U-net architecture generator.
    Basic architecture is the following;
        Input -> Convolution -> encoder_block * n -> decoder_block * n (with skip connections) -> tanh activation
    Arguments
    ------------
        in_shape: input image shape as tupple (h, w, c)
        out_shape: output image shape as tupple (h, w, c)
        model_name: name of this model
        min_filter_num: minimum filter number of each convolution layer
        kernel_size: kernel size of filters in convlutions
        a: a in LeakyReLU, y = max(ax, x)
    Returns
    ------------
        generator_model: generator model
    '''

    # a num of conv = log_2(im_height or width)
    # log_a(X) = log_e(X) / log_e(a)
    # use 'e' to compute
    min_shape = min(in_shape[:-1])
    conv_num = int(np.floor(np.log(min_shape)/np.log(2)))

    # filter number list; ex) [64, 128, 256, 512, 512, 512....]
    filter_nums = [min_filter_num*min(2**i, 8) for i in range(conv_num)]

    # Input layer
    input_l = Input(shape=in_shape, name='input_layer')

    # first convlution layer
    first_encoder = Conv2D(filter_nums[0], kernel_size, strides=strides, padding='same', name='encoder_1')(input_l)
    # make list of encoders to be used for skip connections
    encoders = [first_encoder]

    # Encoder
    for i, filter_num in enumerate(filter_nums[1:]):
        name = 'encode_' + str(i+2)
        encoder = _generator_encoder_block(encoders[-1], filter_num, name=name, kernel_size=kernel_size, strides=strides, a=a)
        encoders.append(encoder)

    # Decoder
    # reverse filter nums for decoders
    # [TBC]the last filter num is not used -> how many filters are used for first encoder??
    decoder_filter_nums = filter_nums[::-1][1:]

    # first decoder
    x = _generator_decoder_block(encoders[-1], encoders[-2], decoder_filter_nums[0], name='decoder_1',upsample_size=strides, kernel_size=kernel_size, dropout=dropout, activation=decoder_act)

    for i, filter_num in enumerate(decoder_filter_nums[1:]):
        name = 'decode_' + str(i+2)
        # first 2+1 decoders have drop out
        if i>=2:
            dropout = 0
        # down sampled by strides, so upsample by strides here
        x = _generator_decoder_block(x, encoders[-(i+3)], decoder_filter_nums[i+1], name=name, upsample_size=strides, kernel_size=kernel_size, dropout=dropout, activation=decoder_act)

    # the last of decoder
    x = Activation(decoder_act)(x)
    x = UpSampling2D(size=strides)(x)
    x = Conv2D(out_shape[-1], kernel_size, name='last_decoder', padding='same')(x)
    x = Activation('tanh')(x)

    generator = Model(inputs=[input_l], outputs=[x], name='generator')

    return generator
