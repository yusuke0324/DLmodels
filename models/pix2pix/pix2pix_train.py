import models
import math
import time
import numpy as np
import os
from datetime import datetime as dt
import keras.backend as K
from keras.utils import progbar
from keras.optimizers import Adam, SGD

def _get_date_str():
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y_%m%d_%H%M')
    return tstr

def combine_images(images):
    total = images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height = images.shape[1:3]
    combined_image = np.zeros((height*rows, width*cols),dtype=images.dtype)
    for index, image in enumerate(images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image

def _make_dir(dir_name):
    if not os.path.exists('dir_name'):
        os.makedirs(dir_name)

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred-y_true), axis=-1)

def pix2pix_train(ground_truth_img, raw_img, groud_truth_img_val, raw_img_val, save_dir='./log',  batch_size=32, epochs=10 **kwargs):
# def train_main(**kwargs):

    '''
    main train DCGAN  for function
    Argments
    ---------------
    ground_truth_img: numpy array of ground truth images whose shape is [number, h, w, channel]
    raw_img: image which will be transformed by generator whose shape is [number, h, w, channel]
    ground_truth_img_val: validation ground truth
    raw_img_val: validation data raw
    batch_size: mini-batch size
    epochs; the total number of epochs
    **kwargs;
    'kernel_size'-> (3,3)
    'patch_size'-> 30
    'a'-> 0.2
    'strides'-> (2,2)
    'min_filter_num'-> 64
    'drop_out'-> 0.5
    'decoder_act'-> 'relu'
    'opt_dcgan' -> Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    'opt_discriminator' -> Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    '''

    # default optimizers
    opt_dcgan = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # paramters
    kernel_size = kwargs.get('kernel_size', (3,3))
    patch_size = kwargs.get('patch_size', 30)
    a = kwargs.get('a', 0.2)
    strides = kwargs.get('strides', (2,2))
    min_filter_num = kwargs.get('min_filter_num', 64)
    dropout = kwargs.get('drop_out', 0.5)
    decoder_act = kwargs.get('decoder_act', 'relu')
    opt_dcgan = kwargs.get('opt_dcgan', opt_dcgan)
    opt_discriminator = kwargs.get('opt_discriminator', opt_discriminator)

    img_shape = ground_truth_img.shape[1:]
    channel_num = ground_truth_img.shape[-1]
    patch_shape = (patch_size, patch_size, channel_num)
    patch_num = img_shape[0]//patch_size * img_shape[1]//patch_size


    # load models
    generator = models.make_generator(img_shape, img_shape, min_filter_num=min_filter_num, kernel_size=kernel_size, strides=strides, a=a, dropout=dropout, decoder_act=decoder_act)
    generator.compile(loss='mae', optimizer=opt_generator)

    discriminator = models.make_discriminator(patch_shape, patch_shape, patch_num, min_filter_num=min_filter_num, kernel_size=kernel_size, strides=strides, a=a)
    # [TBC] not need to compile discriminator here??

    dcgan = models.make_dcgan(generator, discriminator, img_shape, patch_size)

    # display data
    print('ground_truth_img.shape: ', ground_truth_img.shape)
    print('raw_img.shape: ', raw_img.shape)
    print('ground_truth_img_val.shape: ', ground_truth_img_val.shape)
    print('raw_img_val.shape: ', raw_img_val.shape)
    print('patch_size: ', patch_size)
    print('patch_num: ', patch_num)

    # DCGAN is supposed to have TWO outputs, which are 'generated_layer' and 'dcgan_output'
    # generated_layer is output from only generator
    # dcgan output is output from DCGAN
    loss = [l1_loss, 'binary_crossentropy']
    loss_weights = [10, 1]
    dcgan.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    save_dir = save_dir + '/' + _get_date_str() + '/'

    # train
    print('start train')
    for epoch in epochs:

        starttime = time.time()
        random_indices = np.random.permutation(ground_truth_img.shape[0])
        gt_img = ground_truth_img[random_indices]
        r_img = raw_img[random_indices]

        # devide by batches
        gt_img_batch_sets = [gt_img[i:i+batch_size] for i in range(0, gt_img.shape[0], batch_size)]
        r_img_batch_sets = [r_img_batch_sets[i:i+batch_size] for i in range(0, r_img.shape[0], batch_size)]

        batch_i = 0
        progbar = generic_utils.Progbar(ground_truth_img.shape[0])

        # mini batch train
        for (gt_img_batch_set, r_img_batch_set) in zip(gt_img_batch_sets, r_img_batch_sets):
            batch_i += 1

            # generate image
            generated_imgs= generator.predict(gt_img_batch_set, verbose=0)

            # make train data for discriminator
            X = np.concatenate(gt_img_batch_set, generated_imgs)
            y = [1] * batch_size + [0] * batch_size
            # update discriminator
            d_loss = discriminator.train_on_batch(X, y)

            # update generator
            y_gen =  [1]*batch_size
            # [TBC] as gen_y, should raw image or gt_img input??
            g_loss = dcgan.train_on_batch(r_img_batch_set, [r_img_batch_set, y_gen])
            # print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, batch_i, g_loss, d_loss))
            progbar.add(batch_size, value={
                ("d_loss", d_loss),
                ("g total", g_loss[0]),
                ("g l1", g_loss[1]),
                ("g logloss", g_loss[2])
                })

            # [ToDo] save images and visualize
            # twice a batch, save images
            if batch_i % (gt_img.shape[0]//batch_size//2) == 0:
                # gt_ims = gt_img_batch_set[:batch_size//3]
                gen_ims = generated_imgs[:batch_size//3]
                combined_img = combine_images(gen_ims)
                Image.fromarray(combined_img.astype(np.uint8))\
                            .save(save_dir + "%04d_%04d.png" % (epoch, batch_i))

        print('Epoch %s/%s, Time: %s' % (epoch, epochs, time.time()-starttime))

