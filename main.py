import tensorflow as tf
import numpy as np
from DCGAN import DCGAN

flags = tf.app.flags
# DATA SETTINGS (INPUT & OUTPUT)
flags.DEFINE_string("data_dir", "./data", "Directory of data [./data]")
flags.DEFINE_integer("input_height", 96, "Height of input image [96]")
flags.DEFINE_integer("input_width", None, "(Optional) Width of input image")
flags.DEFINE_integer("output_height", 64, "Height of output image [96]")
flags.DEFINE_integer("output_width", None, "(Optional) Width of output image")
flags.DEFINE_integer("aggregate_height", 36, "Height of aggregate output image [36]")
flags.DEFINE_integer("aggregate_width", None, "(Optional) Width of aggregate output image")
flags.DEFINE_integer("channels", 3, "Count of image channels [3]")
# MODEL STRUCTURE SETTINGS
flags.DEFINE_integer("z_dim", 100, "Dimension of random noise [100]")
flags.DEFINE_integer("y_dim", 4800, "Sent2vec dimension [4800]")
flags.DEFINE_integer("yl_dim", 128, "Latent dimension for word embeddings [128]")
flags.DEFINE_integer("fc_dim", 64, "The count of filters of first layer of convolution network [64]")
flags.DEFINE_integer("fd_dim", 64, "The count of filters of first layer of deconvolution network [64]")
# TRAINING SETTINGS
flags.DEFINE_integer("nb_epoch", 300, "Number of epochs to train [300]")
flags.DEFINE_integer("batch_size", 64, "Batch size for training [64]")
flags.DEFINE_integer("sample_num", 64, "Number of sampling [64]")
flags.DEFINE_float("d_learning_rate", 0.0002, "Learning rate for discriminator [0.0002]")
flags.DEFINE_float("g_learning_rate", 0.0002, "Learning rate for generator [0.0002]")
flags.DEFINE_integer("d_round", 1, "Number of iteration of discriminator for each cycle [1]")
flags.DEFINE_integer("g_round", 2, "Number of iteration of generator for each cycle [2]")
flags.DEFINE_float("beta1", 0.5, "Momentum parameter for Adam [0.5]")
# LOG AND MODEL
flags.DEFINE_string("checkpoint_dir", "./model", "Directory for pre-loading model [./model]")
flags.DEFINE_string("save_dir", "./model", "Directory for saving model [./model]")
flags.DEFINE_string("images_dir", "./images", "Directory for sampled images [./images]")
flags.DEFINE_string("training_log", "./train.log", "Path of training log [./train.log]")
flags.DEFINE_string("testing_log", "./test.log", "Path of testing log [./test.log]")
flags.DEFINE_integer("save_step", 500, "save the model every N step [500]")
flags.DEFINE_integer("sample_step", 100, "sample every N step [100]")
# OTHER
flags.DEFINE_boolean("is_train", True, "Training or testing [True]")
flags.DEFINE_string("test_file", "./test.txt", "The testing data, only used for testing [./test.txt]")
FLAGS = flags.FLAGS

if FLAGS.input_width == None: FLAGS.input_width = FLAGS.input_height
if FLAGS.output_width == None: FLAGS.output_width = FLAGS.output_height
if FLAGS.aggregate_width == None: FLAGS.aggregate_width = FLAGS.aggregate_height

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

with tf.Session(config=run_config) as sess:
    model = DCGAN(
        sess=sess,
        input_image_size=(FLAGS.input_height, FLAGS.input_width), 
        output_image_size=(FLAGS.output_height, FLAGS.output_width), 
        aggregate_size=(FLAGS.aggregate_height, FLAGS.aggregate_width),
        channels=FLAGS.channels,
        z_dim=FLAGS.z_dim,
        y_dim=FLAGS.y_dim,
        yl_dim=FLAGS.yl_dim, 
        fc_dim=FLAGS.fc_dim,
        fd_dim=FLAGS.fd_dim,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.sample_num,
        save_step=FLAGS.save_step,
        sample_step=FLAGS.sample_step,
        d_round=FLAGS.d_round,
        g_round=FLAGS.g_round,
        checkpoint_dir=FLAGS.checkpoint_dir,
        save_dir=FLAGS.save_dir,
        data_dir=FLAGS.data_dir,
        images_dir=FLAGS.images_dir,
        training_log=FLAGS.training_log,
        testing_log=FLAGS.testing_log)

    if FLAGS.is_train:
        model.train(FLAGS)
    else:
        sample_y = np.loadtxt(FLAGS.test_file)
        model.test(sample_y)
