import numpy as np
import random
import argparse
import sys
import os
import time
import datetime
import json
import glob
from termcolor import cprint
from skimage.io import imread
from skimage.transform import resize
from scipy.misc import imsave
from math import ceil, sqrt

def print_time_info(string):
    T = time.gmtime()
    Y, M, D = T.tm_year, T.tm_mon, T.tm_mday
    h, m, s = T.tm_hour, T.tm_min, T.tm_sec
    print("[{}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}] {}".format(Y, M, D, h, m, s, string))

def center_crop(image, output_shape):
    image_shape = image.shape
    assert image_shape[0] >= output_shape[0] and image_shape[1] >= output_shape[1], "ImageCropError"
    upper = (image_shape[0] - output_shape[0]) // 2
    lower = upper + output_shape[0]
    left = (image_shape[1] - output_shape[1]) // 2
    right = left + output_shape[1]
    image = image[upper:lower, left:right]
    return image

def get_images(images_path, input_shape, output_shape, crop=True, grayscale=False):
    images = []
    for p in images_path:
        if grayscale: image = imread(p, flatten=True)
        else: image = imread(p)
        if crop: image = center_crop(image, output_shape).astype(np.float)
        else: image = resize(image, output_shape).astype(np.float)
        image /= 255.0
        images.append(image)
    return np.array(images).astype(np.float32)

def get_images_path(count, counter, images_dir):
    offset = len(str(count))
    images_path = []
    for i in range(count):
        images_path.append(os.path.join(images_dir, "{:0>5}_{:0>{}}.jpg".format(counter, i, offset)))
    return images_path

def merge_image(images, aggregate_size, channels):
    if type(aggregate_size) == int: size = (aggregate_size, aggregate_size)
    elif type(aggregate_size) in [list, tuple]:size = aggregate_size
    length_cnt = ceil(sqrt(len(images)))
    merged_image = np.zeros((length_cnt*size[0], length_cnt*size[1], channels), dtype=np.float)
    for idx, image in enumerate(images):
        lup = ((idx // length_cnt)*size[0], (idx % length_cnt)*size[1])
        rlp = (lup[0] + size[0], lup[1] + size[1])
        merged_image[lup[0]:rlp[0], lup[1]:rlp[1]] = resize(image / 255.0, size)
    return merged_image * 255.0

def save_images(images, counter, aggregate_size, channels, images_dir):
    images *= 255.0 # IMPORTANT!!!
    images_path = get_images_path(len(images), counter, images_dir)
    for image, path in zip(images, images_path): imsave(path, image)
    aggregate_path = os.path.join(images_dir, "{:0>5}_all.jpg".format(counter))
    merged_image = merge_image(images, aggregate_size, channels)
    imsave(aggregate_path, merged_image)

def prepare_data(images_dir, tags_list, tags_csv, embeddings_file):
    with open(tags_list, 'r') as file: tags = json.load(file)
    images_tags, images_path = [], []
    images_path = glob.glob(os.path.join(images_dir, "*.jpg"))
    with open(tags_csv, 'r') as file:
        for line in file:
            images_tags.append((-1, -1))
            data = line.strip().replace(',', '\t').split('\t')[1:]
            for d in data:
                tag = d.split(':')[:-1]
                if tag in tags['hair']: 
                    images_tags[-1][0] = tags['hair'].index(tag)
                elif tag in tags['eyes']:
                    images_tags[-1][1] = tags['eyes'].index(tag)
            images_tags[-1] = tags['hair'][images_tags[-1][0]] + \
                    " " + tags['eyes'][images_tags[-1][1]]

    good_idx = []
    for idx, tags in enumerate(images_tags):
        if tags[0] != -1 and tags[1] != -1: good_idx.append(idx)
    
    images_tags = [images_tags[idx] for idx in good_idx]
    images_path = [images_path[idx] for idx in good_idx]

    with open(embeddings_file, 'r') as file: tag_embeddings = json.load(file)
    
    return images_tags, images_path, tag_embeddings
