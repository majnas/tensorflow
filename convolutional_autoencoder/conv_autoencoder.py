#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/blob/master/convolutional_autoencoder
# base code: TensorFlow Github
# date: 2017-December-03
#-----------------------------------------------------------------------------#

"""
    In this code we want to use a convolutinal autoencoder to transfer image 
    from pixel space to dense features representation with smaller spatial 
    sizes then reconstruct image from dense features.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

downloaded_mnist_dir = '/mnt/Document/AI/Dataset/MNIST'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(downloaded_mnist_dir, validation_size=0)

log_dir = './logs/model'


n_train = mnist.train.num_examples
n_test = mnist.test.num_examples

# Network Parameters
kf1, nf1 = 3, 24
kf2, nf2 = 3, 20
kf3, nf3 = 3, 16

# Training Parameters
starter_learning_rate = 0.01
batch_size = 32
display_step = 10
n_epochs = 40
n_itrs = n_train//batch_size

# turn off interactive mode
plt.ioff()

graph = tf.Graph()
with graph.as_default():
    
    # define polynomial decay function 
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.polynomial_decay(learning_rate=starter_learning_rate,
                                              global_step=global_step,
                                              decay_steps=(n_train/batch_size)*n_epochs,
                                              end_learning_rate=starter_learning_rate/100,
                                              power=3.0)
    tf.summary.scalar("learning_rate", learning_rate)
    
    # define placeholder for input images
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    
    # Encoder - Transforming image space to feature space
    with tf.name_scope('encoder') as scope:
        # Convolution Layer 1
        with tf.name_scope('conv1') as scope:
            weights = tf.Variable(tf.truncated_normal([kf1, kf1, 1, nf1], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[nf1]), name='biases')
            conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(conv , name=scope)
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
        # Convolution Layer 2
        with tf.name_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal([kf2, kf2, nf1, nf2], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[nf2]), name='biases')
            conv = tf.nn.conv2d(pool, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(conv , name=scope)
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolution Layer 3
        with tf.name_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal([kf3, kf3, nf2, nf3], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[nf3]), name='biases')
            conv = tf.nn.conv2d(pool, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(conv , name=scope)
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Decoder - Transforming feature space to image space
    with tf.name_scope('decoder') as scope:
        # Unpooling & Convolution Layer 1
        with tf.name_scope('conv1') as scope:
            weights = tf.Variable(tf.truncated_normal([kf3, kf3, nf3, nf3], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[nf3]), name='biases')
            unpool = tf.image.resize_images(pool, size=[7, 7])
            conv = tf.nn.conv2d(unpool, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(conv , name=scope)

        # Unpooling & Convolution Layer 2
        with tf.name_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal([kf2, kf2, nf3, nf2], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[nf2]), name='biases')
            unpool = tf.image.resize_images(relu, size=[14, 14])
            conv = tf.nn.conv2d(unpool, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(conv , name=scope)
    
        # Unpooling & Convolution Layer 3
        with tf.name_scope('conv3') as scope:
            weights = tf.Variable(tf.truncated_normal([kf1, kf1, nf2, nf1], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[nf1]), name='biases')
            unpool = tf.image.resize_images(relu, size=[28, 28])
            conv = tf.nn.conv2d(unpool, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(conv , name=scope)

        # Convolution Layer 4
        with tf.name_scope('conv4') as scope:
            weights = tf.Variable(tf.truncated_normal([1, 1, nf1, 1], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[1]), name='biases')
            conv = tf.nn.conv2d(relu, weights, strides=[1, 1, 1, 1], padding='SAME')
            logits = tf.nn.bias_add(conv, biases)
            decoded_image = tf.nn.sigmoid(logits , name=scope)
    
    with tf.name_scope('loss') as scope:
        # loss needs to be minimised by adjusting W and b
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits, name='loss')
        loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", loss)

    with tf.name_scope('optimizer') as scope:
        # define training step which minimises cross entropy
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    
    merged_summary = tf.summary.merge_all()
    
# Create a session for running operations in the Graph.
with tf.Session(graph=graph) as sess:
    
    # Initialize the variables (the trained variables and the epoch counter).
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter(log_dir)
    writer.add_graph(sess.graph)
    
    for epoch in range(n_epochs): # n_epochs
        for itr in range(n_itrs): #n_itrs
            
            # fetch the batch train images
            train_batch_x, _ = mnist.train.next_batch(batch_size)
            train_batch_x = train_batch_x.reshape((-1, 28, 28, 1))

            sess.run(optimizer, feed_dict={X: train_batch_x})
            
            # evaluate train batch loss
            if ((itr+1) % display_step == 0):
                loss_val, summary = sess.run([loss, merged_summary], feed_dict={X: train_batch_x})
                summary_step = int(epoch*n_itrs + itr)
                writer.add_summary(summary, summary_step)
                print('itr=%d/%d  minibach_loss=%1.5f' % (itr, n_itrs, loss_val))

        
        # Evaluate test images loss
        test_batch_x, _ = mnist.test.next_batch(batch_size)
        test_batch_x = test_batch_x.reshape((-1, 28, 28, 1))
        loss_val = sess.run(loss, feed_dict={X: test_batch_x})
        print('epoch=%d  test_loss=%1.5f' % (epoch, loss_val))
        
        # save 16 image for test in each epoch
        if (epoch == 0):
            test_images = test_batch_x[0:16]
            images = np.reshape(test_images, (-1, 28, 28))
        images_ = sess.run(decoded_image, feed_dict={X: test_images})
        images_ = np.reshape(images_, (-1, 28, 28))
        plt.figure(figsize=(12,8))
        for i in range(8):
            plt.subplot(4,8,i+1)
            plt.imshow(images[i])
            plt.subplot(4,8,i+1+8)
            plt.imshow(images_[i])
            
        for i in range(8,16):
            plt.subplot(4,8,i+1+8)
            plt.imshow(images[i])
            plt.subplot(4,8,i+1+8+8)
            plt.imshow(images_[i])
        
        plt.savefig('epoch_'+str(epoch)+'.png')


