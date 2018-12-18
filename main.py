#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
from random import shuffle

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d
from utils import *
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
num_gpus = 6
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size/num_gpus))


def gen_input_feed_map(num_gpus, t_image_s, t_target_image_s, b_imgs_96, b_imgs_384):
    input_map = dict()
    batch_size_per_gpu = batch_size/num_gpus
    for i in range(num_gpus):
        input_map[t_image_s[i]] = b_imgs_96[batch_size_per_gpu*i:batch_size_per_gpu*(i+1)]
        input_map[t_target_image_s[i]] = b_imgs_384[batch_size_per_gpu*i:batch_size_per_gpu*(i+1)]
    return input_map


def gen_input_feed_map_test(num_gpus, t_image_s, b_imgs_96):
    input_map = dict()
    batch_size_per_gpu = batch_size/num_gpus
    for i in range(num_gpus):
        input_map[t_image_s[i]] = b_imgs_96[batch_size_per_gpu*i:batch_size_per_gpu*(i+1)]
    return input_map


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        if grads[0] is None:
            grad = None
        else:
            grad = tf.stack(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    #train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    #valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    #valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    total_train_imgs = len(train_hr_imgs)
    # Batch size is fixed in this model.
    train_hr_imgs = train_hr_imgs[:total_train_imgs-total_train_imgs%batch_size]
    for j, img in enumerate(train_hr_imgs):
        train_hr_imgs[j] = img[:, :, np.newaxis] #np.repeat(img[:, :, np.newaxis], 3, axis=2)

    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    model_gpus = list()
    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device('/cpu:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr_init, trainable=False)
        
        opt = tf.train.AdamOptimizer(lr_v, beta1=beta1)

        for gpu_ind in range(0, num_gpus):
            reuse = (gpu_ind > 0)

            with tf.device("/gpu:{}".format(gpu_ind)):
                t_image = tf.placeholder('float32', [batch_size/num_gpus, 96, 96, 1],
                    name='t_image_input_to_SRGAN_generator')
                t_target_image = tf.placeholder('float32', [batch_size/num_gpus, 384, 384, 1],
                    name='t_target_image')

                # net_g and net_d are overwritten.
                net_g = SRGAN_g(t_image, is_train=True, reuse=reuse)
                net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=reuse)
                _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

                ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
                """
                t_target_image_224 = tf.image.resize_images(
                    t_target_image, size=[224, 224], method=0,
                    align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
                t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

                net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
                _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)
                """

                ## test inference
                if gpu_ind == 0:
                    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

                # ###========================== DEFINE TRAIN OPS ==========================###
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
                d_loss = d_loss1 + d_loss2

                g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
                mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
                #vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
                g_loss = mse_loss + g_gan_loss # mse_loss + vgg_loss + g_gan_loss

                d_grad = opt.compute_gradients(d_loss)
                mse_grad = opt.compute_gradients(mse_loss)
                g_grad = opt.compute_gradients(g_loss)

                #tf.get_variable_scope().reuse_variables()
                model_gpus.append((t_image, t_target_image, d_loss, mse_loss, g_gan_loss, g_loss,
                    d_grad, mse_grad, g_grad))

        #g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
        #d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

        with tf.device('/cpu:0'):
            t_image_s, t_target_image_s, d_loss_s, mse_loss_s, g_gan_loss_s, g_loss_s, \
                d_grad_s, mse_grad_s, g_grad_s = zip(*model_gpus)
            
            # Take average over all GPUs.
            d_loss = tf.reduce_mean(d_loss_s)
            mse_loss = tf.reduce_mean(mse_loss_s)
            g_gan_loss = tf.reduce_mean(g_gan_loss_s)
            g_loss = tf.reduce_mean(g_loss_s)

            d_grad_op = opt.apply_gradients(average_gradients(d_grad_s))
            mse_grad_op = opt.apply_gradients(average_gradients(mse_grad_s))
            g_grad_op = opt.apply_gradients(average_gradients(g_grad_s))


    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g)
    #tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    #tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    """
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()
    """

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    tl.vis.save_images(sample_imgs_96[0:batch_size/num_gpus], [ni, ni], save_dir_ginit + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384[0:batch_size/num_gpus], [ni, ni], save_dir_ginit + '/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_96[0:batch_size/num_gpus], [ni, ni], save_dir_gan + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384[0:batch_size/num_gpus], [ni, ni], save_dir_gan + '/_train_sample_384.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    """
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        shuffle(train_hr_imgs)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update G
            errM, _ = sess.run([mse_loss, mse_grad_op],
                gen_input_feed_map(num_gpus, t_image_s, t_target_image_s, b_imgs_96, b_imgs_384))
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        print("[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
            epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter))

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 50 == 0):
            out = sess.run(net_g_test.outputs, gen_input_feed_map_test(num_gpus, t_image_s, sample_imgs_96))
            print("[*] save images")
            tl.vis.save_images(out[0:batch_size/num_gpus], [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 50 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
    """

    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            print(" ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay))
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            print(" ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (
                lr_init, decay_every, lr_decay))

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0
        shuffle(train_hr_imgs)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update D
            errD, _ = sess.run([d_loss, d_grad_op], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            ## update G
            errG, errM, errA, _ = sess.run([g_loss, mse_loss, g_gan_loss, g_grad_op],
                gen_input_feed_map(num_gpus, t_image_s, t_target_image_s, b_imgs_96, b_imgs_384))
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        print("[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
            epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter))

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 50 == 0):
            out = sess.run(net_g_test.outputs, gen_input_feed_map_test(num_gpus, t_image_s, sample_imgs_96))
            print("[*] save images")
            tl.vis.save_images(out[0:batch_size/num_gpus], [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 50 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = 'gens' # "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    config.VALID.lr_img_path = 'lrs'
    config.VALID.hr_img_path = 'hrs'


    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    imid = 0  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]

    valid_lr_img = np.stack([valid_lr_img, valid_lr_img, valid_lr_img], axis=2)
    valid_hr_img = np.stack([valid_hr_img, valid_hr_img, valid_hr_img], axis=2)


    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 1], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 1) /  gen HR size: (1, 1356, 2040, 1)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
