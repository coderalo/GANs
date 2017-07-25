from data_utils import *
from utils import *
import sys
import os
import glob
import random
import numpy as np

class DataEngine:
    def __init__(self, data_dir, dataset, with_labels, input_shape, output_shape, channels, crop=True):
        """
            Get the images (path) and labels for the training.
            images: ["xxx.jpg", "ooo.jpg", ...]
            labels:
                MNIST: [image_1, image_2, ...]
                COMIC: 
                    labels: ["xxx hair xx eyes", "ooo hair oo eyes", ...]
                    label_embeddings:
                        {
                            "xxx hair xx eyes": [ ... ]    
                            "ooo hair oo eyes": [ ... ]
                            ...
                        }
        """
        print_time_info("Start reading data from dataset {}...".format(dataset))
        images_dir = os.path.join(data_dir, dataset)
        self.images = glob.glob(os.path.join(images_dir, "*.jpg"))
        self.with_labels = with_labels
        if with_labels:
            if dataset == "MNIST":
                    self.labels = np.reshape(np.loadtxt(os.path.join(images_dir, "labels.csv"))[:, 1], (-1))
                    self.num_classes = 10
            if dataset == "COMIC":  
                tags_list = os.path.join(images_dir, "tags_list.txt")
                tags_csv = os.path.join(images_dir, "tags_clean.csv")
                embeddings_file = os.path.join(images_dir, "embeddings.json")
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
                
                self.labels = [images_tags[idx] for idx in good_idx]
                self.images = [images_path[idx] for idx in good_idx]

                with open(embeddings_file, 'r') as file: self.label_embeddings = json.load(file)
                self.embedding_dim = len(self.label_embeddings(list(self.label_embeddings.keys())[0]))
            if dataset == "WIKIART":
                pass
        print_time_info("Data reading complete.")
        if with_labels:
            assert len(self.images) == len(self.labels), "DataIntegrityError"
        self.counter = 0
        self.data_size = len(self.images)
        self.input_shape, self.output_shape, self.channels, self.crop \
                = input_shape, output_shape, channels, crop
        self.shuffle_data()

    def shuffle_data(self):
        if self.with_labels:
            combined = list(zip(self.images, self.labels))
            random.shuffle(combined)
            self.images[:], self.labels[:] = zip(*combined)
        else:
            random.shuffle(self.images)

    def get_labels(images_path):
        if self.dataset == "MNIST":
            labels = np.zeros(len(images_path), self.num_classes) # class: 0 to 9
            for idx, path in enumerate(images_path):
                num = os.path.basename(path).split('.')[0]
                labels[idx] = self.labels[int(num)]
        elif self.dataset == "COMIC":
            labels = np.zeros(len(images_path), self.embedding_dim)
            for idx, path in enumerate(images_path):
                num = os.path.basename(path).split('.')[0]
                labels[idx] = self.label_embeddings[self.labels[num]]
        elif self.dataset == "WIKIART":
            pass

        return labels

    def get_random_labels(batch_size):
        images_path = np.random.choice(self.images, batch_size)
        labels = get_labels(images_path)
        return labels

    def get_batch(self, batch_size, with_labels=False, with_wrong_labels=False, is_random=False):
        if (with_labels or with_wrong_labels) and not self.with_labels:
            print_time_info("Error! The engine hasn't initialized with the labels, quit.")
            sys.quit()
        if self.counter + batch_size >= self.data_size and not is_random:
            self.counter = 0
            self.shuffle_data()
        if self.channels == 1: grayscale = True
        else: grayscale = False
        if is_random:
            images_path = np.random.choice(self.images, batch_size)
        else:
            images_path = self.images[self.counter: self.counter + batch_size]
        images = get_images(
            images_path = images_path,
            input_shape = self.input_shape, 
            output_shape = self.output_shape,
            crop = self.crop, 
            grayscale = grayscale
            )
        if with_labels:
            labels = self.get_labels(images_path)
        if with_wrong_labels:
            wrong_labels = self.get_random_labels(batch_size)

        if not is_random: self.counter += batch_size
        
        batch = {"images": images}
        if with_labels: batch["labels"] = labels
        if with_wrong_labels: batch["wrong_labels"] = wrong_labels
        
        return batch

    def conditional_test(self, batch_size):
        if not self.with_labels:
            print_time_info("Error! The engine hasn't initialized with the labels, quit.")
            sys.quit()
        if self.dataset == "MNIST":
            label_batch_size = batch_size // self.num_classes
            offset = batch_size % self.num_classes
            labels = np.zeros(len(images_path), self.num_classes) 
            for idx in range(self.num_classes):
                labels[label_batch_size*idx: label_batch_size*(idx+1), idx] = 1.0
        elif self.dataset == "COMIC":
            labels_list = []
            for key, embedding in self.label_embeddings:
                labels_list.append([key, embedding])
            label_batch_size = batch_size // len(labels_list)
            offset = batch_size % len(labels_list)
            labels = np.zeros(len(images_path), self.embedding_dim)
            for idx, data in enumerate(label_list):
                labels[label_batch_size*idx: label_batch_size*(idx+1), :] = data[1]
        elif self.dataset == "WIKIART":
            labels = None

        return labels, offset

    def interpolation_test(self, labels, batch_size):
        if not self.with_labels:
            print_time_info("Error! The engine hasn't initialized with the labels, quit.")
            sys.quit()
        if self.dataset == "MNIST":
            label_1 = label_2 = np.zeros((self.num_classes))
            label_1[int(labels[0])], label_2[int(labels[1])] = 1, 1
        elif self.dataset == "COMIC":
            label_1, label_2 = self.label_embeddings[labels[0]], self.label_embeddings[labels[1]]
        elif self.dataset == "WIKIART":
            label_1 = label_2 = None
        labels = np.zeros((batch_size, len(label_1)))
        for idx in range(len(label_1)):
            labels[:, idx] = np.linspace(label_1[idx], label_2[idx], batch_size)

        return labels
