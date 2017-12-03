#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow
# base code: TensorFlow Github
# date: 2017-December-03            
#-----------------------------------------------------------------------------#

"""
    this code is related to training a convolutional neural network to clasify
    images contain two shapes (triangle and square).
    in this code we read the images and labels from shape_train.tfrecord file 
    to train the model and shape_test.tfrecord file to test the model in each 
    epoch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import tensorflow as tf


# Constants used for dealing with the files
train_tfrecord_addr = 'shape_train.tfrecords'
test_tfrecord_addr = 'shape_test.tfrecords'
n_train_samples = 5760
n_test_samples = 2048
batch_size = 32     # number of batches in each iteration
keep_prob = 0.75    # Dropout, probability to keep units
n_epochs = 25

def read_and_decode(filename, batch_size, num_epochs, num_samples):
    
    filename_queue = tf.train.string_input_producer([train_tfrecord_addr],
                                                    num_epochs=num_epochs)
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label_raw': tf.FixedLenFeature([], tf.string),
                    })
    
    # Convert from a scalar string tensor to a uint8 tensor
    image_raw = tf.decode_raw(features['image_raw'], tf.uint8)
    image_resized = tf.reshape(image_raw, [32*32])
   
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image_resized = tf.cast(image_resized, tf.float32) * (1. / 255) - 0.5
    
    # Convert from a scalar string tensor to a uint8 tensor
    label_raw = tf.decode_raw(features['label_raw'], tf.uint8)
    label_resized = tf.reshape(label_raw, [2])
   
    
    images, labels = tf.train.batch([image_resized, label_resized],
                                    batch_size= batch_size,
                                    capacity= num_samples,
                                    num_threads= 2,)
    return images, labels




def convolutional_network_model(x):
    
    # multilayer perceptron parameters
    nF1 = 16
    nF2 = 32
    nFc1 = 120
    nClass = 2
    x_image = tf.reshape(x, shape=[-1, 32, 32, 1])

    # Convolution Layer 1
    w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, nF1], mean=0.0, stddev=0.1), name="wconv1")
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[nF1]), name='bconv1')
    h_conv1 = tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME', name='hconv1')
    h_relu1 = tf.nn.relu(h_conv1 + b_conv1 , name='hrelu1')
    h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='hpool1')
    
    # Convolution Layer 2
    w_conv2 = tf.Variable(tf.truncated_normal([3, 3, nF1, nF2], mean=0.0, stddev=0.1), name='wconv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[nF2]), name='bconv2')
    h_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME', name='hconv2')
    h_relu2 = tf.nn.relu(h_conv2 + b_conv2, name='hrelu2')
    h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='hpool2')
    
    # Fully Connected Layer 1
    w_fc1 = tf.Variable(tf.truncated_normal([(8 * 8 * nF2), nFc1], mean=0.0, stddev=0.1), name='wfc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[nFc1]), name='bfc1')
    h_pool2_flat = tf.reshape(h_pool2, shape=[-1, (8 * 8 * nF2)])
    h_fc1 = tf.matmul(h_pool2_flat, w_fc1, name='hfc1')
    h_fc1 = tf.add(h_fc1, b_fc1)
    h_fc1 = tf.nn.relu(h_fc1)
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    # Fully Connected Layer 2
    w_fc2 = tf.Variable(tf.truncated_normal([nFc1, nClass], mean=0.0, stddev=0.1), name='wfc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[nClass]), name='bfc2')
    h_fc2 = tf.matmul(h_fc1, w_fc2, name='hfc2')
    output = tf.nn.softmax(h_fc2 + b_fc2, name='y')
    
    return output


def run_training():
    """Train ShapeNet for a number of steps."""
    
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        
        # Input train images and labels.
        train_images, train_labels = read_and_decode(filename= train_tfrecord_addr,
                                                     batch_size= batch_size,
                                                     num_epochs= n_epochs,
                                                     num_samples= n_train_samples)
        # Input test images and labels.
        # define batch_size = all test samples
        test_images, test_labels = read_and_decode(filename= test_tfrecord_addr,
                                                   batch_size= n_test_samples,
                                                   num_epochs= n_epochs,
                                                   num_samples= n_test_samples)

        # define placeholder for input images and labels
        X = tf.placeholder(tf.float32, [None, 32*32])
        Y = tf.placeholder(tf.float32, [None, 2])
        
        # Build a Graph that computes predictions from the inference model.
        prediction = convolutional_network_model(X)

        # Backpropagation
        # measure of error of our model
        # this needs to be minimised by adjusting W and b
        cross_entropy = -tf.reduce_sum(Y * tf.log(prediction))
        
        # define training step which minimises cross entropy
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy)
        
        # argmax gives index of highest entry in vector (1st axis of 1D tensor)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        
        # get mean of all entries in correct prediction, the higher the better
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            
        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        st = time.time()
        # Create a session for running operations in the Graph.
        with tf.Session() as sess:
            
            # Initialize the variables (the trained variables and the
            # epoch counter).
            sess.run(init_op)
        
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
            for epoch in range(n_epochs):
                for itr in range(n_train_samples//batch_size):
                    
                    # fetch the batch train images and labels
                    batch_x, batch_y = sess.run([train_images, train_labels])
                    sess.run([train_op], feed_dict={X: batch_x, Y: batch_y})
                
                # fetch whole test images and labels
                batch_x, batch_y = sess.run([test_images, test_labels])
                
                # feed the model with all test images and labels
                acc, _ = sess.run([accuracy, train_op],
                                  feed_dict={X: batch_x, Y: batch_y})
                print('epoch %d/%d: , accuracy = %.3f' 
                      % (epoch, n_epochs, acc))
    
            
            # When done, ask the threads to stop.
            coord.request_stop()
            
            # Wait for threads to finish.
            coord.join(threads)

        et = time.time()
        duration =  et - st
        print(duration)




if __name__ == '__main__':
    
    # run the model to train
    run_training()
    
    
