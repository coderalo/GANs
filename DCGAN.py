import numpy as np
import random
import argparse
import sys
import os
import json
import glob
from scipy.misc import imread
from utils import *
from model_utils import *
from utils import *
from misc_utils import *

class DCGAN:
        
    ########################################################
    #            initialize and main training              #
    ########################################################    
    
    def __init__(
        self,
        sess, # tensorflow session
        input_image_size=96, 
        output_image_size=64, 
        aggregate_size=36,
        channels=3,
        z_dim=100, # dimension of random noise
        y_dim=4800, # sent2vec dimension
        yl_dim=128, # latent dimension for word embeddings 
        fc_dim=64, # the count of filters of first layer of convolution network
        fd_dim=64, # the count of filters of first layer of deconvolution network
        batch_size=64,
        sample_num=64,
        save_step=100,
        sample_step=50,
        d_round=1, g_round=2, # training round for discriminator / generator for each training cycle
        checkpoint_dir="./model",
        save_dir="./model",
        data_dir="./data",
        images_dir="./images",
        training_log="./train.log",
        testing_log="./test.log"
        ):
        
        self.sess = sess
        self.input_height, self.input_width, self.output_height, self.output_width = \
                valid_input_and_output(input_image_size, output_image_size)
        self.aggregate_size = aggregate_size
        self.channels = channels
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.yl_dim = yl_dim
        self.fc_dim = fc_dim
        self.fd_dim = fd_dim
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.save_step = save_step
        self.sample_step = sample_step
        self.d_round = d_round
        self.g_round = g_round
        self.checkpoint_dir = check_dir(checkpoint_dir)
        self.save_dir = check_dir(save_dir)
        self.images_dir = check_dir(images_dir)
        self.training_log = check_log(training_log)
        self.testing_log = check_log(testing_log, training=False)

        images_dir = os.path.join(data_dir, "faces/")
        tags_list = os.path.join(data_dir, "tags_list.json")
        tags_csv = os.path.join(data_dir, "tags_clean.csv")
        embeddings = os.path.join(data_dir, "tags_embeddings.json")
        self.images_tags, self.data, self.tag_embeddings = prepare_data(images_dir, tags_list, tags_csv, embeddings)

        self.dbn1 = batch_norm(name='batch_d_1')
        self.dbn2 = batch_norm(name='batch_d_2')
        self.dbn3 = batch_norm(name='batch_d_3')
        self.dbn4 = batch_norm(name='batch_d_4')
        self.gbn0 = batch_norm(name='batch_g_0')
        self.gbn1 = batch_norm(name='batch_g_1')
        self.gbn2 = batch_norm(name='batch_g_2')
        self.gbn3 = batch_norm(name='batch_g_3')

        self.build_model()

    def train(self, config):
        self.D_optimizer = tf.train.AdamOptimizer(config['d_learning_rate'], beta1=config['beta1']) \
                .minimize(self.D_loss, var_list=self.d_vars)
        self.G_optimizer = tf.train.AdamOptimizer(config['g_learning_rate'], beta1=config['beta1']) \
                .minimize(self.G_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.G_sum = tf.summary.merge([
            self.z_sum,
            self.D_fake_sum, self.G_sum,
            self.D_fake_loss_sum, self.G_loss_sum
            ])

        self.D_sum = tf.summary.merge([
            self.z_sum,
            self.D_real_sum, self.D_wrong_sum, 
            self.D_real_loss_sum, self.D_wrong_loss_sum, self.D_loss_sum
            ])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter = 1
        print_time_info("Start training...")
        checker, before_counter = self.load_model()
        if checker: counter = before_counter
        
        self.errD_list, self.errG_list = [], []
        for epoch_idx in range(config['nb_epoch']):
            nb_batch = len(self.data) // self.batch_size
            np.random.shuffle(self.data)
            for batch_idx in range(nb_batch):
                self.train_batch(epoch_idx, batch_idx, counter)
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
        self.y_real = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name="real_tag_input")
        self.y_fake = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name="fake_tag_input")
        ## summary
        self.z_sum = tf.summary.histogram('z', self.z)
        # generator and sampler
        self.G = self.generator(self.z, self.y_real)
        self.S = self.sampler(self.z, self.y_real)
        ## summary
        self.G_sum = tf.summary.image('G', self.G)
        # discriminator (for real images and generator's images)
        self.D_real, self.D_real_logits = self.discriminator(self.I, self.y_real, reuse=False)
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, self.y_real, reuse=True)
        self.D_wrong, self.D_wrong_logits = self.discriminator(self.I, self.y_fake, reuse=True)
        ## summary
        self.D_real_sum = tf.summary.histogram('D_real', self.D_real)
        self.D_fake_sum = tf.summary.histogram('D_fake', self.D_fake)
        self.D_wrong_sum = tf.summary.histogram('D_wrong', self.D_wrong)
        # loss of model
        ## discriminator
        self.D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
        self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))
        self.D_wrong_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_wrong_logits, labels=tf.zeros_like(self.D_wrong)))
        self.D_loss = self.D_real_loss + self.D_fake_loss + self.D_wrong_loss
        ## generator
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake)))
        ## summary
        self.D_real_loss_sum = tf.summary.scalar('D_real_loss', self.D_real_loss)
        self.D_fake_loss_sum = tf.summary.scalar('D_fake_loss', self.D_fake_loss)
        self.D_wrong_loss_sum = tf.summary.scalar('D_wrong_loss', self.D_wrong_loss)
        self.D_loss_sum = tf.summary.scalar('D_loss', self.D_loss)
        self.G_loss_sum = tf.summary.scalar('G_loss', self.G_loss)
    
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def discriminator(self, input_tensor, label_tensor, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse: scope.reuse_variables()
            h0 = LeakyReLU(conv2d(input_tensor, self.fc_dim, name="d_h0_conv")) 
            h1 = LeakyReLU(self.dbn1(conv2d(h0, self.fc_dim * 2, name="d_h1_conv")))
            h2 = LeakyReLU(self.dbn2(conv2d(h1, self.fc_dim * 4, name="d_h2_conv")))
            h3 = LeakyReLU(self.dbn3(conv2d(h2, self.fc_dim * 8, name="d_h3_conv")))
            yl = linear(label_tensor, self.yl_dim, name="d_yl")
            yl = tf.expand_dims(yl, 1)
            yl = tf.expand_dims(yl, 2)
            yl = tf.tile(yl, [1, 4, 4, 1], name="d_yl_tile")
            h3_concat = tf.concat([h3, yl], 3, name="h3_concat")
            h4 = LeakyReLU(self.dbn4(conv2d(h3_concat, self.fc_dim * 8, 1, name="d_h4_conv")))
            logits = linear(tf.reshape(h4, [self.batch_size, -1]), 1)

            return tf.nn.sigmoid(logits), logits

    def generator(self, noise_tensor, label_tensor):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            yl = linear(label_tensor, self.yl_dim, name="g_yl")
            noise_tensor = tf.concat([noise_tensor, yl], 1)
            h0 = tf.reshape(linear(noise_tensor, self.fd_dim*8*s_h16*s_w16, name='g_h0_lin'), [-1, s_h16, s_w16, self.fd_dim*8])
            h0 = tf.nn.relu(self.gbn0(h0))
            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.fd_dim*4], name='g_h1_deconv')
            h1 = tf.nn.relu(self.gbn1(h1))
            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.fd_dim*2], name='g_h2_deconv')
            h2 = tf.nn.relu(self.gbn2(h2))
            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.fd_dim], name='g_h3_deconv')
            h3 = tf.nn.relu(self.gbn3(h3))
            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.channels], name='g_h4_deconv')
            
            return (tf.nn.tanh(h4)/2. + 0.5)
    
    def sampler(self, noise_tensor, label_tensor):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            yl = linear(label_tensor, self.yl_dim, name="g_yl")
            noise_tensor = tf.concat([noise_tensor, yl], 1)
            h0 = tf.reshape(linear(noise_tensor, self.fd_dim*8*s_h16*s_w16, name='g_h0_lin'), [-1, s_h16, s_w16, self.fd_dim*8])
            h0 = tf.nn.relu(self.gbn0(h0, train=False))
            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.fd_dim*4], name='g_h1_deconv')
            h1 = tf.nn.relu(self.gbn1(h1, train=False))
            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.fd_dim*2], name='g_h2_deconv')
            h2 = tf.nn.relu(self.gbn2(h2, train=False))
            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.fd_dim], name='g_h3_deconv')
            h3 = tf.nn.relu(self.gbn3(h3, train=False))
            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.channels], name='g_h4_deconv')
            
            return (tf.nn.tanh(h4)/2. + 0.5)

    ########################################################
    #                   train and sample                   #
    ########################################################    

    def train_batch(self, epoch_idx, batch_idx, counter):
        batch_z, batch_I, batch_y_real, batch_y_fake = \
                self.get_data(self.data[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size])
        
        for _ in range(self.d_round): 
            batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            _, summary_str = self.sess.run([self.D_optimizer, self.D_sum],
                    feed_dict={
                        self.I: batch_I,
                        self.z: batch_z,
                        self.y_real: batch_y_real,
                        self.y_fake: batch_y_fake
                        })

            self.writer.add_summary(summary_str, counter)

        for _ in range(self.g_round): 
            batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            _, summary_str = self.sess.run([self.G_optimizer, self.G_sum],
                    feed_dict={
                        self.z: batch_z,
                        self.y_real: batch_y_real
                        })

            self.writer.add_summary(summary_str, counter)

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

        errD = errD_real + errD_fake + errD_wrong

        errG = self.G_loss.eval({
            self.z: batch_z,
            self.y_real: batch_y_real
            })

        print_time_info("Epoch {:0>3} batch {:0>5} errD: {}, errG: {}".format(epoch_idx, batch_idx, errD, errG))
        with open(self.training_log, 'a') as file:
            file.write("{},{},{},{}\n".format(epoch_idx, batch_idx, errD, errG))

        self.errD_list.append(errD)
        self.errG_list.append(errG)
    
    def sample_test(self, counter):
        sample_z, sample_I, sample_y_real, sample_y_fake = self.get_data(self.data[:self.sample_num])
        samples, d_loss, g_loss = self.sess.run(
                [self.S, self.D_loss, self.G_loss],
                feed_dict={
                    self.z: sample_z,
                    self.I: sample_I,
                    self.y_real: sample_y_real,
                    self.y_fake: sample_y_fake
                    })
        # TODO: revise the data processing code
        save_images(samples, counter, self.aggregate_size, self.channels, self.images_dir)
        print_time_info("Counter {} errD: {}, errG: {}".format(counter, d_loss, g_loss))
        with open(self.testing_log, 'a') as file:
            file.write("{},{},{}\n".format(counter, d_loss, g_loss))
    
    ########################################################
    #                   data processing                    #
    ########################################################   

    def get_tags(self, images_path):
        tags = []
        for p in images_path:
            name = os.path.basename(p).split('.')[0]
            tag = self.images_tags[int(name)]
            tags.append(self.tag_embeddings[tag])
        return np.array(tags)

    def get_data(self, images_path):
        batch_z = np.random.uniform(-1, 1, size=(len(images_path), self.z_dim))
        input_shape, output_shape = (self.input_height, self.input_width), (self.output_height, self.output_width)
        batch_I = get_images(images_path, input_shape, output_shape)
        batch_y_real = np.squeeze(self.get_tags(images_path))
        batch_y_fake = np.squeeze(self.get_tags(np.random.choice(self.data, len(images_path))))
        return batch_z, batch_I, batch_y_real, batch_y_fake

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