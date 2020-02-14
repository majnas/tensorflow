#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/
# date: 2020-February-14
#------------------------------------------------------------------------------------------#

""" 
    functions to read images and labels for train and test portion of dataset. 
    These function uses tf.data api from tensorflow 1.x in order to read, prefetch
     and batch data for model. 
"""

import tensorflow as tf 
import os
from config import Config as cfg

def get_filenames_list(is_training):
    if is_training:
        fh = open(os.path.join(cfg.DATA_DIR, "train_images_list.txt"), "r")
        image_list = [line.rstrip() for line in fh.readlines()]
        fh = open(os.path.join(cfg.DATA_DIR, "train_labels_list.txt"), "r")
        label_list = [int(line.rstrip()) for line in fh.readlines()]
    else:
        fh = open(os.path.join(cfg.DATA_DIR, "test_images_list.txt"), "r")
        image_list = [line.rstrip() for line in fh.readlines()]
        fh = open(os.path.join(cfg.DATA_DIR, "test_labels_list.txt"), "r")
        label_list = [int(line.rstrip()) for line in fh.readlines()]
    return image_list, label_list

def _parse_function(image_paths, labels, is_training):
    image_string = tf.io.read_file(image_paths)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_decoded = image_decoded / 255
    image_resized = tf.image.resize(image_decoded, size=(cfg.IMG_HIGHT, cfg.IMG_WIDTH))
    labels = tf.one_hot(labels, depth=cfg.N_CLASSES)
    return image_resized, labels

def input_fn(is_training, batch_size):
    image_list, label_list = get_filenames_list(is_training=is_training)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
    if is_training:
        dataset = dataset.shuffle(buffer_size=cfg.N_TRAIN)
    dataset = dataset.map(lambda image,label: _parse_function(image, label, is_training))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(None) # repeat forever
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels

