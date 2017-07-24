import tensorflow as tf
import numpy as np
from DCGAN import DCGAN
from GAN import GAN
from DataEngine import DataEngine

flags = tf.app.flags
# DATA SETTINGS (INPUT & OUTPUT)
flags.DEFINE_string("data_dir", "./data", "Directory of data [./data]")
flags.DEFINE_string("dataset", "MNIST", "The dataset going to be used [MNIST] / COMIC / WIKIART")
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
flags.DEFINE_string("type", "GAN", "The type of GAN going to be used [GAN]")
flags.DEFINE_boolean("is_conditional", False, "Conditional GAN or not [False]")
flags.DEFINE_integer("z_dim", 100, "Dimension of random noise [100]")
flags.DEFINE_string("y_dim", 10, "Dimension of label [10]")
## CONVOLUTIONAL GAN
flags.DEFINE_integer("fc_dim", 64, "The count of filters of first layer of convolution network [64]")
flags.DEFINE_integer("fd_dim", 64, "The count of filters of first layer of deconvolution network [64]")
## MLP GAN
flags.DEFINE_integer("h_dim", 128, "The dimension of hidden MLP layer [128]")
## TEXT2IMAGE GAN
flags.DEFINE_integer("w_dim", 4800, "Sent2vec dimension [4800]")
flags.DEFINE_integer("yl_dim", 128, "Latent dimension for word embeddings [128]")
# TRAINING SETTINGS
flags.DEFINE_integer("iterations", 500000, "Number iterations to train [500000]")
flags.DEFINE_integer("batch_size", 64, "Batch size for training [64]")
flags.DEFINE_integer("sample_num", 64, "Number of sampling [64]")
flags.DEFINE_float("d_learning_rate", 0.001, "Learning rate for discriminator [0.001]")
flags.DEFINE_float("g_learning_rate", 0.001, "Learning rate for generator [0.001]")
flags.DEFINE_integer("d_round", 1, "Number of iteration of discriminator for each cycle [1]")
flags.DEFINE_integer("g_round", 1, "Number of iteration of generator for each cycle [1]")
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
## CONDITIONAL GAN
flags.DEFINE_string("test_file", "./test.txt", "The testing data, only used for testing [./test.txt]")
FLAGS = flags.FLAGS

if FLAGS.input_width == None: FLAGS.input_width = FLAGS.input_height
if FLAGS.output_width == None: FLAGS.output_width = FLAGS.output_height
if FLAGS.aggregate_width == None: FLAGS.aggregate_width = FLAGS.aggregate_height

input_shape = (FLAGS.input_height, FLAGS.input_width)
output_shape = (FLAGS.output_height, FLAGS.output_width)

Engine = DataEngine(FLAGS.data_dir, FLAGS.dataset, FLAGS.is_conditional, input_shape, output_shape, FLAGS.channels, FLAGS.is_crop)

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

with tf.Session(config=run_config) as sess:
    
    if FLAGS.type == "GAN":
        model = GAN(sess, FLAGS, Engine)
        if FLAGS.is_train:
            model.train(FLAGS)
        else:
            model.test()

    elif FLAGS.type == "T2I-GAN":
        model = DCGAN(sess, FLAGS, Engine)
        if FLAGS.is_train:
            model.train(FLAGS)
        else:
            sample_y = np.loadtxt(FLAGS.test_file)
            model.test(sample_y)
