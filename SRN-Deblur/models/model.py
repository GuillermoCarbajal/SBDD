from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
from util.BasicConvLSTMCell import *


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = args.nScales
        # self.scale = 0.5
        self.scale = self.args.scaleFactor if self.args.scaleFactor != '' else 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = args.crop_size
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)

        self.validation_available = False
        if self.args.validation_list != '':
            self.validation_list = open(args.validation_list, 'rt').read().splitlines()
            self.validation_list = list(map(lambda x: x.split(' '), self.validation_list))
            random.shuffle(self.validation_list)
            self.validation_available = True


        if self.args.training_dir == '':  # this was the original code
            self.train_dir = os.path.join('./checkpoints', args.model)
        else:
            self.train_dir = self.args.training_dir

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate
        self.epochsToSave = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, 3000, 4000]
        self.std_dev_gauss_noise = args.std_dev_gauss_noise
        self.gamma_factor = args.gamma_factor

    def validation_producer(self, batch_size=10):

        def read_data():
            print(self.validation_queue[0], self.validation_queue[1])
            img_a = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.validation_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.validation_queue[1]])),
                                          channels=3)
            img_a, img_b = preprocessing([img_a, img_b])
            return img_a, img_b

        def preprocessing(imgs):
            imgs = [tf.pow((tf.cast(img, tf.float32) / 255.0), self.gamma_factor) for img in imgs]
            if self.args.model != 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]

            img_crop = tf.unstack(
                tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns]),
                axis=0)

            return img_crop

        with tf.variable_scope('validation'):
            List_all = tf.convert_to_tensor(self.validation_list, dtype=tf.string)
            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]

            self.validation_queue = tf.train.slice_input_producer([in_list, gt_list], capacity=20)
            image_in, image_gt = read_data()
            batch_in, batch_gt = tf.train.batch([image_in, image_gt], batch_size=batch_size, num_threads=8, capacity=20)

        return batch_in, batch_gt

    def input_producer(self, batch_size=10):

        def read_data():
            print(self.data_queue[0], self.data_queue[1])
            img_a = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[1]])),
                                          channels=3)
            img_a, img_b = preprocessing([img_a, img_b])
            return img_a, img_b

        def preprocessing(imgs):

            imgs = [tf.pow(tf.cast(img, tf.float32) / 255.0, self.gamma_factor) for img in imgs]
            if self.args.model != 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]

            img_crop = tf.unstack(tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns]),
                                  axis=0)
            print('img crop shape', img_crop[0].get_shape())
            noise = tf.random_normal(shape=img_crop[0].get_shape(), mean=0.0, stddev=self.std_dev_gauss_noise, dtype=tf.float32)

            do_flip = tf.random_uniform([]) > 0.5
            img_crop[0] = tf.cond(do_flip, lambda: tf.image.flip_left_right(img_crop[0]), lambda: img_crop[0])
            img_crop[1] = tf.cond(do_flip, lambda: tf.image.flip_left_right(img_crop[1]), lambda: img_crop[1])

            img_crop[0] = img_crop[0] + noise

            return img_crop

        with tf.variable_scope('input'):
            List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]

            self.data_queue = tf.train.slice_input_producer([in_list, gt_list], capacity=20)
            image_in, image_gt = read_data()
            batch_in, batch_gt = tf.train.batch([image_in, image_gt], batch_size=batch_size, num_threads=8, capacity=20)

        return batch_in, batch_gt


    def generator(self, inputs, reuse=False, scope='g_net' ):
        isVerbose = self.args.generator_verbose
        n, h, w, c = inputs.get_shape().as_list()

        if isVerbose:
            print('Generating default architecture')
            print('Input parameters: n= %d, h= %d, w= %d, c=%d' % (n, h, w, c))

        if self.args.model == 'lstm':
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            if isVerbose:
                print('Model is lstm, cell shape: h/4= %d, w/4= %d, 128 filters 3x3' % (h/4, w/4))

        x_unwrap = []
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                if isVerbose:
                    print('El numero de escalas es %d' % self.n_levels)

                for i in xrange(self.n_levels):

                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    if isVerbose:
                        print('Image size in scale %d is %d x %d' % (i, hi, wi))
                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                    inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))

                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')

                    if self.args.model == 'lstm':
                        rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)
                        if isVerbose:
                            print('rnn state shape in scale %d is %d x %d' % (i, hi//4, wi//4))

                    # encoder
                    conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                    conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                    conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                    conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
                    conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                    conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                    conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                    conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                    conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                    conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                    conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                    conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

                    if self.args.model == 'lstm':
                        deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                    else:
                        deconv3_4 = conv3_4

                    # decoder
                    deconv3_3 = ResnetBlock(deconv3_4, 128, 5, scope='dec3_3')
                    deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                    deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')
                    deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                    cat2 = deconv2_4 + conv2_4

                    deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                    deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                    deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')
                    deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
                    cat1 = deconv1_4 + conv1_4
                    deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                    deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                    deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')
                    inp_pred = slim.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')

                    if isVerbose:
                        print('deconv2_4, conv2_4, cat2 ', deconv2_4.get_shape(), conv2_4.get_shape(), cat2.get_shape())
                        print('deconv1_4, conv1_4, cat1 ', deconv1_4.get_shape(), conv1_4.get_shape(), cat1.get_shape())

                    if i >= 0:
                        x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.get_variable_scope().reuse_variables()


            return x_unwrap

    def build_model(self):


        img_in, img_gt = self.input_producer(self.batch_size)

        if self.validation_available:
            val_in, val_gt = self. validation_producer(batch_size=self.batch_size)
            print('Validation list is available')

        self.psnrs = tf.image.psnr(img_in, img_gt, max_val=1.0)

        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))
        print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())


        # generator

        x_unwrap = self.generator(img_in, reuse=False, scope='g_net')
        if self.validation_available:
            xval_unwrap = self.generator(val_in, reuse=True, scope='g_net')


        # calculate multi-scale loss
        self.loss_total = 0
        self.val_loss_batch = 0
        #self.val_loss_total =  tf.placeholder(tf.float32, [])
        for i in xrange(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)

            if self.validation_available:
                valgt_i = tf.image.resize_images(val_gt, [hi, wi], method=0)
                val_loss = tf.reduce_mean((valgt_i - xval_unwrap[i]) ** 2)
                self.val_loss_batch += val_loss


            loss = tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)
            self.loss_total += loss

            tf.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.summary.scalar('loss_' + str(i), loss)

        # losses
        tf.summary.scalar('loss_total', self.loss_total)
        if self.validation_available:
            tf.summary.scalar('validation_loss_batch', self.val_loss_batch)
            #self.val_summary = tf.summary.scalar('validation_loss', self.val_loss_total)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]

        # all the varaiables are optimized (default behaviour)
        optim_vars = all_vars
        self.optim_vars = optim_vars

        print('Optimization variables: %d parameters' % len(self.optim_vars))
        for var in self.optim_vars:
            print(var.name)


    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        if self.args.step != -1:
            # Resume training
            global_step = tf.Variable(initial_value=self.args.step, dtype=tf.int32, trainable=False)
        else:
            global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)

        self.global_step = global_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0.0,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.optim_vars) # by default optim_vars is all_vars

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=4)

        if self.args.step != -1:
            self.load(sess, self.train_dir, step=self.args.step)


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)
        print('Initializing training ...')
        for step in xrange(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update G network
            _, loss_total_val = sess.run([train_gnet, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f; %.5f, %.5f)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, 0.0,
                                    0.0, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 20000 == 0 or step == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)
            if step % (self.max_steps / self.epoch ) == 0:
                currentEpoch = int(step / self.data_size)
                if currentEpoch in self.epochsToSave:
                    checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                    self.save(sess, checkpoint_path, currentEpoch, model_name='deblur.model.epoch')

    def save(self, sess, checkpoint_dir, step, model_name="deblur.model"):
        #model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None, model_name="deblur.model"):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print('Model %s loaded' % ckpt_name)
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print('Model %s loaded' % ckpt_name)
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width, input_path, output_path):

        import collections

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3

        network_dict = collections.defaultdict(list)

        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=False)

        network_dict['%dx%d' % (H,W)] = [inputs, outputs]

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, self.args.step)
        data_list = sorted(os.listdir(input_path))

        for i, img_name in enumerate(data_list):
            print('%d/%d: %s' % (i+1, len(data_list), img_name))
            _inp_data = scipy.misc.imread(os.path.join(input_path, img_name))  # (h, w, c)
            inp_data = (_inp_data.astype('float32') / 255)**self.args.gamma_factor

            h = int(inp_data.shape[0])
            w = int(inp_data.shape[1])

            if (h % 16) == 0:
                new_h = h
            else:
                new_h = h - (h % 16) + 16
            if (w % 16) == 0:
                new_w = w
            else:
                new_w = w - (w % 16) + 16

            if network_dict['%dx%d' % (new_h,new_w)] == []:
                print('add network to dict', new_h, new_w)
                inputs = tf.placeholder(shape=[self.batch_size, new_h, new_w, inp_chns], dtype=tf.float32)
                outputs = self.generator(inputs, reuse=True)

                network_dict['%dx%d' % (new_h, new_w)] = [inputs, outputs]

            if (new_h - h) > 0 or (new_w - w) > 0:
                inp_data = np.pad(inp_data, ((0, new_h - h), (0, new_w - w), (0, 0)), 'edge')

            inp_data = np.expand_dims(inp_data, 0)
            if self.args.model == 'color':
                val_x_unwrap = sess.run(network_dict['%dx%d' % (new_h, new_w)][1], feed_dict={network_dict['%dx%d' % (new_h, new_w)][0]: inp_data})
                out = val_x_unwrap[-1]
            else:
                inp_data = np.transpose(inp_data, (3, 1, 2, 0))  # (c, h, w, 1)
                val_x_unwrap = sess.run(network_dict['%dx%d' % (new_h, new_w)][1], feed_dict={network_dict['%dx%d' % (new_h, new_w)][0]: inp_data})
                out = val_x_unwrap[-1]
                out = np.transpose(out, (3, 1, 2, 0))  # (1, h, w, c)

            if (new_h - h) > 0 or (new_w - w) > 0:
                out = out[:, :h, :w, :]


            out = out**(1.0/self.args.gamma_factor)
            out = np.clip(out*255, 0, 255) + 0.5
            out = out.astype('uint8')
            out = out[0]

            out_img_name = img_name #+ ".png"
            scipy.misc.imsave(os.path.join(output_path, out_img_name), out)
