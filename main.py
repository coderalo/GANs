import tensorflow as tf
from DCGAN import DCGAN

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

with tf.Session(config=run_config) as sess:
    model = DCGAN(sess)
    config = {}
    config['d_learning_rate'] = config['g_learning_rate'] = 0.0002
    config['beta1'] = 0.5
    config['nb_epoch'] = 300 
    model.train(config)
