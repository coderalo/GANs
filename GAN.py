"""
The original generative adversarial network.
Ref: https://arxiv.org/pdf/1406.2661.pdf

Codera Lo, 2017.07.18
"""

import os
import sys
import numpy as np
from math import ceil
from utils import *
from model_utils import *
from data_utils import *

class GAN:
        
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
        self.h_dim = FLAGS.h_dim
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
        if FLAGS.is_train:
            self.training_log = check_log(FLAGS.training_log)
            self.testing_log = check_log(FLAGS.testing_log, training=False)
        self.is_conditional = FLAGS.is_conditional
        if self.is_conditional: self.y_dim = FLAGS.y_dim

        self.data_engine = Engine

        self.build_model()

    def train(self, config):
        self.D_optimizer = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1) \
                .minimize(self.D_loss, var_list=self.d_vars)
        self.G_optimizer = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1) \
                .minimize(self.G_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.G_sum = tf.summary.merge([
            self.z_sum, 
            self.D2_sum, self.G_sum, 
            self.D2_loss_sum, self.G_loss_sum
            ])

        self.D_sum = tf.summary.merge([
            self.z_sum,
            self.D1_sum, self.D1_loss_sum,
            self.D_loss_sum
            ])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter = 1
        print_time_info("Start training...")
        checker, before_counter = self.load_model()
        if checker: counter = before_counter
        
        self.errD_list, self.errG_list = [], []
        for _ in range(config.iterations):
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
        self.I = tf.placeholder(tf.float32, [None] + output_shape, name="image_input")
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name="noise_input")
        if self.is_conditional: self.y = tf.placeholder(tf.float32, [None, self.y_dim], name="label_input")
        ## summary
        self.z_sum = tf.summary.histogram('z', self.z)
        # generator and sampler
        if self.is_conditional:
            self.G = self.generator(self.z, self.y)
            self.S = self.generator(self.z, self.y)
        else:
            self.G = self.generator(None, self.z)
            self.S = self.sampler(None, self.z)
        ## summary
        self.G_sum = tf.summary.image('G', self.G)
        # discriminator (for real images and generator's images)
        if self.is_conditional:
            self.D1, self.D1_logits = self.discriminator(self.I, self.y, reuse=False)
            self.D2, self.D2_logits = self.discriminator(self.G, self.y, reuse=True)
        else:
            self.D1, self.D1_logits = self.discriminator(self.I, None, reuse=False)
            self.D2, self.D2_logits = self.discriminator(self.G, None, reuse=True)
        ## summary
        self.D1_sum = tf.summary.histogram('D1', self.D1)
        self.D2_sum = tf.summary.histogram('D2', self.D2)
        # loss of model
        ## discriminator
        self.D1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits, labels=tf.ones_like(self.D1)))
        self.D2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.zeros_like(self.D2)))
        self.D_loss = self.D1_loss + self.D2_loss
        ## generator
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.ones_like(self.D2)))
        ## summary
        self.D1_loss_sum = tf.summary.scalar('D1_loss', self.D1_loss)
        self.D2_loss_sum = tf.summary.scalar('D2_loss', self.D2_loss)
        self.D_loss_sum = tf.summary.scalar('D_loss', self.D_loss)
        self.G_loss_sum = tf.summary.scalar('G_loss', self.G_loss)
    
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def discriminator(self, input_tensor, label_tensor=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse: scope.reuse_variables()
            I = tf.reshape(input_tensor, (-1, self.input_height * self.input_width))
            if label_tensor != None: 
                h = tf.nn.relu(linear(tf.concat([I, label_tensor], 1), self.h_dim, name="d_hid_lin"))
            else:
                h = tf.nn.relu(linear(I, self.h_dim, name="d_hid_lin"))
                
            logits = linear(h, 1, name="d_logits")
            return tf.nn.sigmoid(logits), logits

    def generator(self, label_tensor, noise_tensor):
        with tf.variable_scope("generator") as scope:
            if label_tensor != None:
                h = tf.nn.relu(linear(tf.concat([noise_tensor, label_tensor], 1), self.h_dim, name="g_hid_lin"))
            else:
                h = tf.nn.relu(linear(noise_tensor, self.h_dim, name="g_hid_lin"))
            output = tf.reshape(tf.nn.sigmoid(linear(h, 784, name="g_output")), (self.batch_size, 28, 28, self.channels))
            
            return output
    
    def sampler(self, label_tensor, noise_tensor):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            if label_tensor != None:
                h = tf.nn.relu(linear(tf.concat([noise_tensor, label_tensor], 1), self.h_dim, name="g_hid_lin"))
            else:
                h = tf.nn.relu(linear(noise_tensor, self.h_dim, name="g_hid_lin"))
            output = tf.reshape(tf.nn.sigmoid(linear(h, 784, name="g_output")), (self.batch_size, 28, 28, self.channels))
            
            return output

    ########################################################
    #                   train and sample                   #
    ########################################################    

    def train_batch(self, counter):
        if self.is_conditional: 
            batch = self.data_engine.get_batch(self.batch_size, with_labels=True)
            batch_I = batch['images']
            batch_y = batch['labels']
        else:
            batch = self.data_engine.get_batch(self.batch_size, with_labels=False) 
            batch_I = batch['images']

        for _ in range(self.d_round): 
            batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            if self.is_conditional:
                _, summary_str = self.sess.run([self.D_optimizer, self.D_sum],
                        feed_dict={
                            self.I: batch_I,
                            self.y: batch_y,
                            self.z: batch_z
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
                            self.I: batch_I,
                            self.y: batch_y,
                            self.z: batch_z
                            })
            else:
                _, summary_str = self.sess.run([self.G_optimizer, self.G_sum],
                        feed_dict={
                            self.I: batch_I,
                            self.z: batch_z
                            })
            
            self.writer.add_summary(summary_str, counter)
        
        if self.is_conditional:
            errD1 = self.D1_loss.eval({self.I: batch_I, self.y: batch_y})
            errD2 = self.D2_loss.eval({self.z: batch_z, self.y: batch_y})
            errG = self.G_loss.eval({self.z: batch_z, self.y: batch_y})
        else:
            errD1 = self.D1_loss.eval({self.I: batch_I})
            errD2 = self.D2_loss.eval({self.z: batch_z})
            errG = self.G_loss.eval({self.z: batch_z})

        errD = errD1 + errD2

        if counter % self.verbose_step == 0:
            print_time_info("Iteration {:0>7} errD: {}, errG: {}".format(counter, errD, errG))
            with open(self.training_log, 'a') as file:
                file.write("{},{},{}\n".format(counter, errD, errG))

            self.errD_list.append(errD)
            self.errG_list.append(errG)
    
    def sample_test(self, counter):
        if self.is_conditional: 
            sample = self.data_engine.get_batch(self.sample_num, with_labels=True, is_random=True)
            sample_I = sample['images']
            sample_y = sample['labels']
        else:
            sample = self.data_engine.get_batch(self.sample_num, with_labels=False, is_random=True) 
            sample_I = sample['images']
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        
        if self.is_conditional:
            samples, d_loss, g_loss = self.sess.run(
                    [self.S, self.D_loss, self.G_loss],
                    feed_dict={
                        self.z: sample_z,
                        self.y: sample_y,
                        self.I: sample_I
                        })
        else:
            samples, d_loss, g_loss = self.sess.run(
                    [self.S, self.D_loss, self.G_loss],
                    feed_dict={
                        self.z: sample_z,
                        self.I: sample_I
                        })

        save_images(samples, counter, self.aggregate_size, self.channels, self.images_dir, True)
        print_time_info("Counter {} errD: {}, errG: {}".format(counter, d_loss, g_loss))
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
        samples = self.sess_run(self.S, feed_dict={self.z: sample_z})
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
