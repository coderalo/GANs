from tensorflow.examples.tutorials.mnist import input_data
from skimage.io import imread
from scipy.misc import imsave
import numpy as np
import sys
import os
import shutil

data_dir = sys.argv[1]
if not os.path.exists(data_dir): os.makedirs(data_dir)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
images, labels = mnist.train.images, mnist.train.labels

for idx, (image, label) in enumerate(zip(images, labels)):
    path = os.path.join(data_dir, "{}.jpg".format(idx))
    labels = os.path.join(data_dir, "labels.csv")
    imsave(path, np.array(image).reshape((28, 28)))
    with open(labels, 'a') as file:
        file.write("{},{}\n".format(idx, label))

shutil.rmtree("MNIST_data/")
