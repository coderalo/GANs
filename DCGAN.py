"""
The deep convolutional generative adversarial network (Ref(1)).
For the conditional version, it used the method from Ref(2).
Ref(1): https://arxiv.org/pdf/1406.2661.pdf
Ref(2): https://arxiv.org/pdf/1605.05396.pdf
Codera Lo, 2017.07.18
"""
import os
import sys
import glob
import json
import numpy as np
from math import ceil, floor, log
from utils import *
from model_utils import *
from data_utils import *

class DCGAN:
        
    ########################################################
    #            initialize and main training              #
    ########################################################    
    
    def __init__(self, sess, FLAGS, Engine):
        self.sess = sess
        self.input_height, self.input_width, self.output_height, self.output_width = \
                valid_input_and_output((FLAGS.input_height, FLAGS.input_width), 
                        (FLAGS.output_height, FLAGS.output_width))
        self.aggregate_size = (FLAGS.aggregate_height, FLAGS.aggregate_width)
        self.channels = FLAGS.channels
        self.z_dim = FLAGS.z_dim
        self.fc_dim = FLAGS.fc_dim
        self.fd_dim = FLAGS.fd_dim
        self.batch_size = FLAGS.batch_size
        self.sample_num = FLAGS.sample_num
        self.save_step = FLAGS.save_step
        self.sample_step = FLAGS.sample_step
        self.verbose_step = FLAGS.verbose_step
        self.d_round = FLAGS.d_round
        self.g_round = FLAGS.g_round
        self.checkpoint_dir = check_dir(FLAGS.checkpoint_dir)
        self.save_dir = check_dir(FLAGS.save_dir)
        self.images_dir = check_dir(FLAGS.images_dir)
        if FLAGS.is_train == True:
            self.training_log = check_log(FLAGS.training_log)
            self.testing_log = check_log(FLAGS.testing_log, training=False)
        else:
            self.test_file = FLAGS.test_file
        self.is_conditional = FLAGS.is_conditional
        if self.is_conditional:
            self.y_dim = FLAGS.y_dim
            self.yl_dim = FLAGS.yl_dim

        self.data_engine = Engine

        self.layers_count, self.conv_size = count_layers(self.output_height, self.output_width)
        self.dbn, self.gbn = [None] * (self.layers_count + 1), [None] * (self.layers_count+1)
        for idx in range(self.layers_count): self.dbn[idx+1] = batch_norm(name='batch_d_{}'.format(idx+1))
        for idx in range(self.layers_count): self.gbn[idx] = batch_norm(name='batch_g_{}'.format(idx))

        self.build_model()

    def train(self, config):
        self.D_optimizer = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1) \
                .minimize(self.D_loss, var_list=self.d_vars)
        self.G_optimizer = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1) \
                .minimize(self.G_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.G_sum = tf.summary.merge([
            self.z_sum,
            self.D_fake_sum, self.G_sum,
            self.D_fake_loss_sum, self.G_loss_sum
            ])
        
        if self.is_conditional:
            self.D_sum = tf.summary.merge([
                self.z_sum,
                self.D_real_sum, self.D_wrong_sum, 
                self.D_real_loss_sum, self.D_wrong_loss_sum, self.D_loss_sum
                ])
        else:
            self.D_sum = tf.summary.merge([
                self.z_sum,
                self.D_real_sum,
                self.D_real_loss_sum, self.D_loss_sum
                ])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter = 1
        print_time_info("Start training...")
        checker, before_counter = self.load_model()
        if checker: counter = before_counter
        
        self.errD_list, self.errG_list = [], []
        for __ in range(config.iterations):
            self.train_batch(counter)
            if counter % self.sample_step == 0: self.sample_test(counter)
            if counter % self.save_step == 0: self.save_model(counter)
            counter += 1

    ########################################################
    #                    model structure                   #
    ########################################################    
    
    def build_model(self):
        # inputs (images, noises and tags)
        output_shape = [self.output_height, self.output_width, self.channels]
        self.I = tf.placeholder(tf.float32, [self.batch_size] + output_shape, name="image_input")
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="noise_input")
        if self.is_conditional:
            self.y_real = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name="real_tag_input")
            self.y_fake = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name="fake_tag_input")
        ## summary
        self.z_sum = tf.summary.histogram('z', self.z)
        
        # generator and sampler
        if self.is_conditional:
            self.G = self.generator(self.z, self.y_real)
            self.S = self.sampler(self.z, self.y_real)
        else:
            self.G = self.generator(self.z, None)
            self.S = self.sampler(self.z, None)
        ## summary
        self.G_sum = tf.summary.image('G', self.G)
        
        # discriminator (for real images and generator's images)
        if self.is_conditional:
            self.D_real, self.D_real_logits = self.discriminator(self.I, self.y_real, reuse=False)
            self.D_fake, self.D_fake_logits = self.discriminator(self.G, self.y_real, reuse=True)
            self.D_wrong, self.D_wrong_logits = self.discriminator(self.I, self.y_fake, reuse=True)
        else:
            self.D_real, self.D_real_logits = self.discriminator(self.I, None, reuse=False)
            self.D_fake, self.D_fake_logits = self.discriminator(self.G, None, reuse=True)
        ## summary
        self.D_real_sum = tf.summary.histogram('D_real', self.D_real)
        self.D_fake_sum = tf.summary.histogram('D_fake', self.D_fake)
        if self.is_conditional:
            self.D_wrong_sum = tf.summary.histogram('D_wrong', self.D_wrong)
        
        # loss of model
        ## discriminator
        self.D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
        self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))
        if self.is_conditional:
            self.D_wrong_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_wrong_logits, labels=tf.zeros_like(self.D_wrong))) 
            self.D_loss = self.D_real_loss + self.D_fake_loss + self.D_wrong_loss
        else:
            self.D_loss = self.D_real_loss + self.D_fake_loss
        ## generator
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake)))
        ## summary
        self.D_real_loss_sum = tf.summary.scalar('D_real_loss', self.D_real_loss)
        self.D_fake_loss_sum = tf.summary.scalar('D_fake_loss', self.D_fake_loss)
        self.D_loss_sum = tf.summary.scalar('D_loss', self.D_loss)
        self.G_loss_sum = tf.summary.scalar('G_loss', self.G_loss)
        if self.is_conditional:
            self.D_wrong_loss_sum = tf.summary.scalar('D_wrong_loss', self.D_wrong_loss)
    
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def discriminator(self, input_tensor, label_tensor, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse: scope.reuse_variables()
            hidden = []
            for idx in range(self.layers_count):
                if idx == 0:
                    hidden.append(
                            LeakyReLU(conv2d(input_tensor, self.fc_dim, name="d_h0_conv"))
                            )
                else:
                    hidden.append(
                            LeakyReLU(self.dbn[idx](conv2d(hidden[-1], 
                                self.fc_dim * (2 ** idx), 
                                name="d_h{}_conv".format(idx))))
                            )
            if self.is_conditional:      
                yl = linear(label_tensor, self.yl_dim, name="d_yl")
                yl = tf.expand_dims(yl, 1)
                yl = tf.expand_dims(yl, 2)
                yl = tf.tile(yl, [1, self.conv_size, self.conv_size, 1], name="d_yl_tile")
                h_concat = tf.concat([hidden[-1], yl], 3, name="d_h{}_concat".format(self.layers_count - 1))
                hidden.append(
                        LeakyReLU(self.dbn[-1](conv2d(h_concat, self.fc_dim * (2 ** self.layers_count), 
                            1, name="d_h{}_conv".format(self.layers_count))))
                        )
                logits = linear(tf.reshape(hidden[-1], [self.batch_size, -1]), 1)
            else:
                logits = linear(tf.reshape(hidden[-1], [self.batch_size, -1]), 1)
            
            return tf.nn.sigmoid(logits), logits

    def generator(self, noise_tensor, label_tensor):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = [self.output_height], [self.output_width]
            for i in range(self.layers_count):
                s_h.insert(0, conv_out_size_same(s_h[0], 2))
                s_w.insert(0, conv_out_size_same(s_w[0], 2))
           
            hidden = []
            if self.is_conditional:
                yl = linear(label_tensor, self.yl_dim, name="g_yl")
                noise_tensor = tf.concat([noise_tensor, yl], 1)
                for idx in range(self.layers_count+1):
                    if idx == 0: 
                        hidden.append(tf.nn.relu(self.gbn[0](tf.reshape(linear(noise_tensor, self.fd_dim*(2 ** self.layers_count)*s_h[0]*s_w[0], name='g_h0_lin'), [-1, s_h[0], s_w[0], self.fd_dim*(2 ** self.layers_count)]))))
                    else:
                        hidden.append(tf.nn.relu(self.gbn[idx](deconv2d(hidden[-1], [self.batch_size, s_h[idx], s_w[idx], self.fd_dim*(2 ** (self.layers_count-idx))], name='g_h{}_deconv'.format(idx)))))
                hidden.append(deconv2d(hidden[-1], [self.batch_size, s_h[-1], s_w[-1], self.channels], name='g_h{}_deconv'.format(self.layers_count+1)))
            else:
                for idx in range(self.layers_count):
                    if idx == 0:
                        hidden.append(tf.nn.relu(self.gbn[0](tf.reshape(linear(noise_tensor, self.fd_dim*(2 ** self.layers_count)*s_h[0]*s_w[0], name='g_h0_lin'), [-1, s_h[0], s_w[0], self.fd_dim*(2 ** self.layers_count)]))))
                    else:
                        h = self.gbn[idx](deconv2d(hidden[-1], [self.batch_size, s_h[idx], s_w[idx], self.fd_dim*(2 ** (self.layers_count-idx))], name='g_h{}_deconv'.format(idx)))
                        hidden.append(tf.nn.relu(h))
                hidden.append(deconv2d(hidden[-1], [self.batch_size, s_h[-1], s_w[-1], self.channels], name='g_h{}_deconv'.format(self.layers_count)))
            return (tf.nn.tanh(hidden[-1])/2. + 0.5)
    
    def sampler(self, noise_tensor, label_tensor):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w = [self.output_height], [self.output_width]
            for i in range(self.layers_count):
                s_h.insert(0, conv_out_size_same(s_h[0], 2))
                s_w.insert(0, conv_out_size_same(s_w[0], 2))
           
            hidden = []
            if self.is_conditional:
                yl = linear(label_tensor, self.yl_dim, name="g_yl")
                noise_tensor = tf.concat([noise_tensor, yl], 1)
                for idx in range(self.layers_count+1):
                    if idx == 0: 
                        hidden.append(tf.nn.relu(self.gbn[0](tf.reshape(linear(noise_tensor, self.fd_dim*(2 ** self.layers_count)*s_h[0]*s_w[0], name='g_h0_lin'), [-1, s_h[0], s_w[0], self.fd_dim*(2 ** self.layers_count)]), train=False)))
                    else:
                        hidden.append(tf.nn.relu(self.gbn[idx](deconv2d(hidden[-1], [self.batch_size, s_h[idx], s_w[idx], self.fd_dim*(2 ** (self.layers_count-idx))], name='g_h{}_deconv'.format(idx)), train=False)))
                hidden.append(deconv2d(hidden[-1], [self.batch_size, s_h[-1], s_w[-1], self.channels], name='g_h{}_deconv'.format(self.layers_count+1)))
            else:
                for idx in range(self.layers_count):
                    if idx == 0:
                        hidden.append(tf.nn.relu(self.gbn[0](tf.reshape(linear(noise_tensor, self.fd_dim*(2 ** self.layers_count)*s_h[0]*s_w[0], name='g_h0_lin'), [-1, s_h[0], s_w[0], self.fd_dim*(2 ** self.layers_count)]), train=False)))
                    else:
                        h = self.gbn[idx](deconv2d(hidden[-1], [self.batch_size, s_h[idx], s_w[idx], self.fd_dim*(2 ** (self.layers_count-idx))], name='g_h{}_deconv'.format(idx)), train=False)
                        hidden.append(tf.nn.relu(h))
                hidden.append(deconv2d(hidden[-1], [self.batch_size, s_h[-1], s_w[-1], self.channels], name='g_h{}_deconv'.format(self.layers_count)))
            return (tf.nn.tanh(hidden[-1])/2. + 0.5)

    ########################################################
    #                   train and sample                   #
    ########################################################    

    def train_batch(self, counter):
        if self.is_conditional:
            batch = self.data_engine.get_batch(self.batch_size, with_labels=True, with_wrong_labels=True)
            batch_I = batch["images"]
            batch_y_real = batch["labels"]
            batch_y_fake = batch["fake_labels"]
        else:
            batch = self.data_engine.get_batch(self.batch_size)
            batch_I = batch["images"]

        for _ in range(self.d_round):
            batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            if self.is_conditional:
                _, summary_str = self.sess.run([self.D_optimizer, self.D_sum],
                        feed_dict={
                            self.I: batch_I,
                            self.z: batch_z,
                            self.y_real: batch_y_real,
                            self.y_fake: batch_y_fake
                            })
            else:
                _, summary_str = self.sess.run([self.D_optimizer, self.D_sum],
                        feed_dict={
                            self.I: batch_I,
                            self.z: batch_z
                            })

            self.writer.add_summary(summary_str, counter)

        for _ in range(self.g_round): 
            batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            if self.is_conditional:
                _, summary_str = self.sess.run([self.G_optimizer, self.G_sum],
                        feed_dict={
                            self.z: batch_z,
                            self.y_real: batch_y_real
                            })
            else:
                _, summary_str = self.sess.run([self.G_optimizer, self.G_sum],
                        feed_dict={
                            self.z: batch_z
                            })

            self.writer.add_summary(summary_str, counter)

        if self.is_conditional:
            errD_real = self.D_real_loss.eval({
                self.I: batch_I,
                self.y_real: batch_y_real
                })
            
            errD_fake = self.D_fake_loss.eval({
                self.z: batch_z,
                self.y_real: batch_y_real
                })

            errD_wrong = self.D_wrong_loss.eval({
                self.I: batch_I,
                self.y_fake: batch_y_fake
                })
            errG = self.G_loss.eval({
                self.z: batch_z,
                self.y_real: batch_y_real
                })

            errD = errD_real + errD_fake + errD_wrong
        
        else:
            errD_real = self.D_real_loss.eval({
                self.I: batch_I
                })
            errD_fake = self.D_fake_loss.eval({
                self.z: batch_z
                })
            errG = self.G_loss.eval({
                self.z: batch_z
                })
            
            errD = errD_real + errD_fake
            
        if counter % self.verbose_step == 0:
            print_time_info("Iteration {:0>7} errD: {}, errG: {}".format(counter, errD, errG))
            with open(self.training_log, 'a') as file:
                file.write("{},{},{}\n".format(counter, errD, errG))

            self.errD_list.append(errD)
            self.errG_list.append(errG)
    
    def sample_test(self, counter):
        if self.is_conditional:
            sample = self.data_engine.get_batch(self.sample_num, with_labels=True, with_wrong_labels=True, is_random=True)
            sample_I = sample["images"]
            sample_y_real = sample["labels"]
            sample_y_fake = sample["fake_labels"]
        else:
            sample = self.data_engine.get_batch(self.sample_num, is_random=True)
            sample_I = sample["images"]

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        
        if self.is_conditional:
            samples, d_loss, g_loss = self.sess.run(
                    [self.S, self.D_loss, self.G_loss],
                    feed_dict={
                        self.z: sample_z,
                        self.I: sample_I,
                        self.y_real: sample_y_real,
                        self.y_fake: sample_y_fake
                        })
        else:
            samples, d_loss, g_loss = self.sess.run(
                    [self.S, self.D_loss, self.G_loss],
                    feed_dict={
                        self.z: sample_z,
                        self.I: sample_I
                        }
                    )
        save_images(samples, counter, self.aggregate_size, self.channels, self.images_dir, True)
        print_time_info("Iteration {} validation errD: {}, errG: {}".format(counter, d_loss, g_loss))
        with open(self.testing_log, 'a') as file:
            file.write("{},{},{}\n".format(counter, d_loss, g_loss))
  
    ########################################################
    #                       testing                        #
    ########################################################   
    
    def test(self):
        checker, before_counter = self.load_model()
        if not checker:
            print_time_info("There isn't any ready model, quit.")
            sys.quit()
        
        sample_z = np.random_uniform(-1, 1, size=(self.batch_size, self.z_dim))
        if self.is_conditional:
            sample = self.data_engine.get_batch(self.batch_size, with_labels=True, is_random=True)
            sample_y = sample['labels']
            samples = self.sess.run(self.S, feed_dict={self.z: sample_z, self.y_real: sample_y})
        else:
            samples = self.sess.run(self.S, feed_dict={self.z: sample_z})

        save_images(samples, 2, self.aggregate_size, self.channels, self.images_dir, False)
        print_time_info("Testing end!")
    
    ########################################################
    #                load and save model                   #
    ########################################################   
    
    def load_model(self):
        import re
        print_time_info("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print_time_info("Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print_time_info("Failed to find a checkpoint")
            return False, 0

    def save_model(self, counter):
        model_name = os.path.join(self.save_dir, "{}.ckpt".format(counter))
        print_time_info("Saving checkpoint...")
        self.saver.save(self.sess, model_name, global_step=counter)
