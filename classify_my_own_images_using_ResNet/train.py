#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/
# date: 2020-February-14
#------------------------------------------------------------------------------------------#

"""
    Classify our own images using Resnet50. 
    In fact this code is to show how to use pretrained popular architectures as feature 
    extractor and place multiple layers on top of them for own application. In this code we 
    used Resnet50 as backend, obviously it can be replaced with others like (ResNet, ...).
    
    * The whole pipline of the project
    >>> creating train and test splits of our dataset.
        here we create an synthatic image dataset of seven symbols. Symbol names are, "spadesuit",
        "clubsuit", "Join", "Omega", "Phi", "Psi", "heartsuit". this dataset includes images 
        with diffrent sizes. obviously this dataset can be replaced with other datasets if the 
        dataset structure remain inact. Dataset structure is as follows.

        ---- dataset
                '
                '---- train
                '       '
                '       '---- category_1
                '       '       '
                '       '       '---- image_1.png
                '       '       '---- image_2.png
                '       '       '---- ....
                '       '       '---- image_n.png
                '       '
                '       '---- category_2
                '       '       '
                '       '       '---- image_1.png
                '       '       '---- image_2.png
                '       '       '---- ....
                '       '       '---- image_n.png
                '       '
                '       '----  ...
                '       '       
                '       '       
                '       '
                '       '---- category_m
                '               '
                '               '---- image_1.png
                '               '---- image_2.png
                '               '---- ....
                '               '---- image_n.png
                '       
                ' --- test 
                        '
                        '---- category_1
                        '       '
                        '       '---- image_1.png
                        '       '---- image_2.png
                        '       '---- ....
                        '       '---- image_n.png
                        '
                        '---- category_2
                        '       '
                        '       '---- image_1.png
                        '       '---- image_2.png
                        '       '---- ....
                        '       '---- image_n.png
                        '
                        '----  ...
                        '       
                        '       
                        '
                        '---- category_m
                                '
                                '---- image_1.png
                                '---- image_2.png
                                '---- ....
                                '---- image_n.png

    >>> Making a list of images and labels of train and test split.
        by running 

    >>> Loading pretrained architecture weights and placeing some layers on top of it. then train
        the model in two diffrent scenario. Finetuning backend weight or use the weight as is. these
        two scenarios will lead two different results. it's clear that tuning backend will result 
        better performance.


    In this code we used tf.data api to read images and labels from folders.
    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf 
import dataset_utils
import numpy as np 
import matplotlib.pyplot as plt
from config import Config as cfg
import tensorflow.contrib.layers as layers 
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim

# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if cfg.BASE_ARCHITECTURE == "resnet_v2_50":
    base_model = resnet_v2.resnet_v2_50    
else:
    base_model = resnet_v2.resnet_v2_101

tf.reset_default_graph()

# train dataset queue using tf.data
train_datset = dataset_utils.input_fn(is_training=True, batch_size=cfg.BATCH_SIZE)

# test dataset queue using tf.data
test_dataset = dataset_utils.input_fn(is_training=False, batch_size=cfg.TEST_BATCH_SIZE)

images = tf.placeholder(tf.float32, shape=[None, cfg.IMG_HIGHT, cfg.IMG_WIDTH, 3])
labels = tf.placeholder(tf.float32, shape=[None, cfg.N_CLASSES])
keep_prob = tf.placeholder(tf.float32, shape=None)
is_training = True
# keep_prob = None

with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=cfg.BATCH_NORM_DECAY)):
    fratures, end_points = base_model(images,
                                    num_classes=None,
                                    is_training=is_training,
                                    global_pool=False)

exclude = [cfg.BASE_ARCHITECTURE + "/logits", cfg.BASE_ARCHITECTURE + "/AuxLogits"]
variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude)
variables_to_restore = tf.contrib.framework.get_variables_to_restore()
init_fn = tf.contrib.framework.assign_from_checkpoint_fn(cfg.RESNET_CKPT_DIR, variables_to_restore)

# Head on top of backed network
flatten = layers.flatten(fratures, scope="head")
flatten = tf.nn.dropout(flatten, keep_prob=keep_prob)
fc1 = layers.fully_connected(flatten, num_outputs=120, activation_fn=tf.nn.relu, scope="head/fc1")
fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
logits = layers.fully_connected(fc1, num_outputs=cfg.N_CLASSES, activation_fn=None, scope="head/fc2")
pred = tf.argmax(tf.nn.softmax(logits), axis=1)
# print(logits)

head_variables = [t for t in tf.trainable_variables() if "head" in t.name]
head_init = tf.variables_initializer(head_variables)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
cross_entropy = tf.reduce_mean(cross_entropy)

if cfg.FREEZE_BATCH_NORM:
    # in the case of smaller batch_size not to change batch normalization parameters
    train_var_list = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
else:
    train_var_list = [v for v in tf.trainable_variables()]

if not cfg.FINETUNE:
    # tuning only new trainable variables
    train_var_list = head_variables

# l2 regularization
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])

# Add weight decay to the loss.
with tf.variable_scope("total_loss"):
    loss = cross_entropy + cfg.WEIGHT_DECAY * l2_loss

opt = tf.train.AdamOptimizer(learning_rate=cfg.LEARNING_RATE).minimize(loss, var_list=train_var_list)

acc = tf.equal(pred, tf.argmax(labels, axis=1))
acc = tf.cast(acc, tf.float32)
acc = tf.reduce_mean(acc)

# print(train_var_list)
graph = tf.get_default_graph()
fc1_w = graph.get_tensor_by_name("head/fc2/weights:0")
conv1_w = graph.get_tensor_by_name("resnet_v2_50/conv1/weights:0")

with tf.Session() as sess:
    #Initializations
    sess.run(tf.global_variables_initializer())
    init_fn(sess)
    sess.run(head_init)

    conv1_w_val = sess.run(conv1_w)
    print("resnet conv1_w before training:", conv1_w_val[0,0,0,0:3])

    n_iters = cfg.N_EPOCHS * cfg.N_TRAIN // cfg.BATCH_SIZE
    for itr in range(n_iters):
        x_train, y_train = sess.run(train_datset)
        feed_dict = {images: x_train, labels: y_train, keep_prob: 0.5}
        sess.run(opt, feed_dict=feed_dict)

        if (itr%10 == 0):
            train_loss_value, train_acc_value = sess.run([loss, acc], feed_dict=feed_dict)
            print("ITER", itr, "Train Loss:", train_loss_value, "Train ACC:", train_acc_value)


        # After 50 iterations we evaluate model on test samples
        if (itr%100 == 0):
            n_test_iters = cfg.N_TEST // cfg.TEST_BATCH_SIZE
            loss_array = []
            acc_array = []
            for _ in range(n_test_iters):
                x_test, y_test = sess.run(test_dataset)
                feed_dict = {images: x_test, labels:y_test, keep_prob: 1.0}
                test_loss_value, test_acc_value = sess.run([loss, acc], feed_dict=feed_dict)
                p = sess.run(pred, feed_dict=feed_dict)
                loss_array.append(test_loss_value)
                acc_array.append(test_acc_value)
            
            mean_loss_value = np.mean(loss_array)
            mean_acc_value = np.mean(acc_array)
            print("ITER", itr, "Test Loss:", mean_loss_value, "Test ACC:", mean_acc_value)


    conv1_w_val = sess.run(conv1_w)
    print("resnet conv1_w after training:", conv1_w_val[0,0,0,0:3])

