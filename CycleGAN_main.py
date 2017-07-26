import tensorflow as tf
import numpy as np
from CycleGAN import CycleGAN

flags = tf.app.flags
# DATA SETTINGS (INPUT & OUTPUT)
flags.DEFINE_string("data_dir", "./data", "Directory of data [./data]")
flags.DEFINE_integer("input_height", 28, "Height of input image [28]")
flags.DEFINE_integer("input_width", None, "(Optional) Width of input image")
flags.DEFINE_integer("output_height", 28, "Height of output image [28]")
flags.DEFINE_integer("output_width", None, "(Optional) Width of output image")
flags.DEFINE_integer("aggregate_height", 28, "Height of aggregate output image [28]")
flags.DEFINE_integer("aggregate_width", None, "(Optional) Width of aggregate output image")
flags.DEFINE_integer("channels", 1, "Count of image channels [1]")
flags.DEFINE_boolean("is_crop", True, "Crop the images (if input_size > output_size) or not [True]")
# MODEL STRUCTURE SETTINGS
## GLOBAL
flags.DEFINE_integer("fc_dim", 64, "The count of filters of first layer of convolution network [64]")
flags.DEFINE_integer("fd_dim", 64, "The count of filters of first layer of deconvolution network [64]")
# TRAINING SETTINGS
flags.DEFINE_integer("iterations", 500000, "Number iterations to train [500000]")
flags.DEFINE_integer("batch_size", 1, "Batch size for training [1]")
flags.DEFINE_integer("sample_num", 1, "Number of sampling [1]")
flags.DEFINE_float("d_learning_rate", 0.0002, "Learning rate for discriminator [0.0002]")
flags.DEFINE_float("g_learning_rate", 0.0002, "Learning rate for generator [0.0002]")
flags.DEFINE_float("L1_lambda", 10, "Scale of consistency loss [10]")
## ADAM (FOR NOT W-GAN)
flags.DEFINE_float("beta1", 0.9, "Momentum parameter for Adam [0.9]")
# LOG AND MODEL
flags.DEFINE_string("checkpoint_dir", "./model", "Directory for pre-loading model [./model]")
flags.DEFINE_string("save_dir", "./model", "Directory for saving model [./model]")
flags.DEFINE_string("images_dir", "./images", "Directory for sampled images [./images]")
flags.DEFINE_string("training_log", "./train.log", "Path of training log [./train.log]")
flags.DEFINE_string("testing_log", "./test.log", "Path of testing log [./test.log]")
flags.DEFINE_integer("save_step", 50000, "save the model every N step [50000]")
flags.DEFINE_integer("sample_step", 10000, "sample every N step [10000]")
flags.DEFINE_integer("verbose_step", 1000, "output log every N step [1000]")
# OTHER
flags.DEFINE_boolean("is_train", True, "Training or testing [True]")
FLAGS = flags.FLAGS

if FLAGS.input_width == None: FLAGS.input_width = FLAGS.input_height
if FLAGS.output_width == None: FLAGS.output_width = FLAGS.output_height
if FLAGS.aggregate_width == None: FLAGS.aggregate_width = FLAGS.aggregate_height

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

with tf.Session(config=run_config) as sess:
    model = CycleGAN(sess, FLAGS)
    if FLAGS.is_train:
        model.train(FLAGS)
    else:
        model.test()
