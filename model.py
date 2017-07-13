import numpy as np
import random
import argparse
import sys
import os
import json
import glob
from utils import *
from model_utils import *
from utils import *


def valid_input_and_output(input_image_size, output_image_size):
    def valid_size(size):
        if type(size) == list or type(size) == tuple:
            assert len(size) <= 2, "ImageShapeError"
            if len(size) == 2:
                height, width = size
            else:
                height = width = size[0]
        elif type(size) == int:
            height = width = size[0]
        else: assert False, "ImageShapeTypeError"
        return height, width
    
    input_height, input_width = valid_size(input_image_size)
    if output_image_size == None:
        output_height, output_width = input_height, input_width
    else:
        output_height, output_width = valid_size(output_image_size)

    return input_height, input_width, output_height, output_width
     

class DCGAN:
        
    def __init__(
        self,
        sess, # tensorflow session
        input_image_size=64, output_image_size=None, channels=3,
        z_dim=100, # dimension of random noise
        y_dim=1200*4, # word embeddings for 4 words
        yl_dim=128, # latent dimension for word embeddings 
        fc_dim=64, # the count of filters of first layer of convolution network
        fd_dim=64, # the count of filters of first layer of deconvolution network
        batch_size=64,
        d_round=2, g_round=1, # training round for discriminator / generator for each training cycle
        ):
        
        self.sess = sess
        self.input_height, self.input_width, self.output_height, self.output_width = \
                valid_input_and_output(input_image_size, output_image_size)
        self.channels = channels
        self.y_dim = y_dim
        self.yl_dim = yl_dim
        self.fc_dim = fc_dim
        self.fd_dim = fd_dim
        self.batch_size = batch_size
        self.d_round = d_round
        self.g_round = g_round

        self.build_model()

    def build_model():
        # inputs (images, noises and tags)
        input_shape = [self.input_height, self.input_width, self.channels]
        self.I = tf.placeholder(tf.float32, [self.batch_size] + input_shape, name="image_input")
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
        self.D_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real))
        self.D_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake))
        self.D_wrong_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_wrong_logits, labels=tf.zeros_like(self.D_wrong))
        self.D_loss = self.D_real_loss + self.D_fake_loss + self.D_wrong_loss
        ## generator
        self.G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake))
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

    def get_data(images_dir, label_path):
        print_time_info("Read images data from {}".format(images_dir))
        print_time_info("Read tags data from {}".format(label_path))
        self.images = glob.glob(images_dir + "/*.jpg")
        labels = np.loadtxt(label_path)
        self.labels = {label[0]: (label[1], label[2]) for label in labels}

    def train(config):
        D_optimizer = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1) \
                .minimize(self.D_loss, var_list=self.d_vars)
        G_optimizer = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1) \
                .minimize(self.G_loss, var_list=self.g_vars)
