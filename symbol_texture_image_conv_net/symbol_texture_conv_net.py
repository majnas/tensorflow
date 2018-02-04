#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/tree/master/symbol_texture_image_conv_net
# date: 2018-Feb-04
#------------------------------------------------------------------------------------------#

"""
    this code is related to training a convolutional neural network to clasify
    images contain 7 symbols (spadesuit, clubsuit, Join, Omega, Phi, Psi, 
    heartsuit)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import time
import tensorflow as tf
import numpy as np
import os
from mlxtend.preprocessing import one_hot
import dataset_mean
    

# Constants used for dealing with the files
n_classes = 7
n_train_samples = n_classes * 800
n_test_samples = n_classes  * 200

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./dataset')
parser.add_argument('--train_dir', default='dataset/train')
parser.add_argument('--test_dir', default='dataset/test')
parser.add_argument('--batch_size', default=28, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--n_epochs', default=2000, type=int)
parser.add_argument('--keep_prob', default=0.8, type=int)
parser.add_argument('--log_dir', default='./logs/model/')
parser.add_argument('--checkpoint_name', default='model.ckpt')


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.png
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and test get them in the same order
    labels.sort()
    

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))
    
    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i        

    labels = [label_to_int[l] for l in labels]

    labels = one_hot(labels, num_labels=n_classes, dtype=np.float32)

    return filenames, labels

def main(args):
    # Get the list of filenames and corresponding list of labels for training et test
    train_filenames, train_labels = list_images(args.train_dir)
    test_filenames, test_labels = list_images(args.test_dir)    
    

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        # Preprocessing (for both training and test):
        # (1) Decode the image from png format
        # (2) Resize the image so its smaller side is 48 pixels long
        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)        # (1)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 48.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            scale = tf.cond(tf.greater(height, width),
                            lambda: smallest_side / width,
                            lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)
            
            
            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            
            return resized_image, label

        # Preprocessing (for training)
        # (3) Take a random 36x36 crop to the scaled image
        # (4) Horizontally flip the image with probability 1/2
        # (5) Substract the per color mean `DATASET_MEAN`
        # (6) Divide all pixels by 255 to normalize them between (-1, +1)
        def training_preprocess(image, label):
            crop_image = tf.random_crop(image, [36, 36, 3])                       # (3)
            flip_image = tf.image.random_flip_left_right(crop_image)              # (4)

            means = tf.reshape(tf.constant(DATASET_MEAN), [1, 1, 3])
            centered_image = flip_image - means									  # (5)
            centered_image = tf.divide(centered_image, 255)                       # (6)
            
            return centered_image, label

        # Preprocessing (for test)
        # (7) Take a central 224x224 crop to the scaled image
        # (8) Substract the per color mean `DATASET_MEAN`
        # (9) Divide all pixels by 255 to normalize them between (-1, +1)
        # Note: we don't normalize the data here, as VGG was trained without normalization
        def test_preprocess(image, label):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 36, 36)    # (7)
            means = tf.reshape(tf.constant(DATASET_MEAN), [1, 1, 3])
            centered_image = crop_image - means                                   # (8)
            centered_image = tf.divide(centered_image, 255)						  # (9)

            return centered_image, label

        # ----------------------------------------------------------------------
        # DATASET CREATION using tf.contrib.data.Dataset
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

        # The tf.contrib.data.Dataset framework uses queues in the background to feed in
        # data to the model.
        # We initialize the dataset with a list of filenames and labels, and then apply
        # the preprocessing functions described above.
        # Behind the scenes, queues will load the filenames, preprocess them with multiple
        # threads and apply the preprocessing in parallel, and then batch the data

        # Training dataset
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(_parse_function,
                                          num_threads=args.num_workers, 
                                          output_buffer_size=args.batch_size)
        train_dataset = train_dataset.map(training_preprocess,
                                          num_threads=args.num_workers, 
                                          output_buffer_size=args.batch_size)     
        train_dataset = train_dataset.shuffle(buffer_size=n_train_samples)  # don't forget to shuffle
        train_dataset = train_dataset.repeat(None) # Infinite iterations
        batched_train_dataset = train_dataset.batch(args.batch_size)



        # Test dataset
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_filenames, test_labels))
        test_dataset = test_dataset.map(_parse_function,
                                        num_threads=args.num_workers,
                                        output_buffer_size=args.batch_size)
        test_dataset = test_dataset.map(test_preprocess,
                                        num_threads=args.num_workers, 
                                        output_buffer_size=args.batch_size)

  		# for test_dataset technically it's not needed to shuffle but in each
  		# epoch we test the network on multiple batch (n_batch_for_test) of 
  		# test images, since dataset is sorted its better for testing to shuffle
  		# test_dataset  
        test_dataset = test_dataset.shuffle(buffer_size=n_test_samples)
        test_dataset = test_dataset.repeat(None)
        batched_test_dataset = test_dataset.batch(args.batch_size)

        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(test_init_op)   for 1 epoch on the test set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `test_dataset` here, because they are compatible.
        train_iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                                 batched_train_dataset.output_shapes)
        train_init_op = train_iterator.make_initializer(batched_train_dataset)
        train_images, train_labels = train_iterator.get_next()


        test_iterator = tf.contrib.data.Iterator.from_structure(batched_test_dataset.output_types,
                                                                 batched_test_dataset.output_shapes)
        test_init_op = test_iterator.make_initializer(batched_test_dataset)
        test_images, test_labels = test_iterator.get_next()


        # Define Convolutional Network Graph
        # Convolutional Neural Network parameters
        nF1 = 16
        nF2 = 32
        nFc1 = 120        
        
        # define placeholder for input images and labels
        X = tf.placeholder(tf.float32, [None, 36, 36, 3])
        Y = tf.placeholder(tf.float32, [None, 7])
            
        # Convolution Layer 1
        with tf.name_scope('conv1') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 3, nF1], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[nF1]), name='biases')
            conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(conv , name=scope)
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", conv)            

        # Convolution Layer 2
        with tf.name_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, nF1, nF2], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[nF2]), name='biases')
            conv = tf.nn.conv2d(pool, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(conv, name=scope)
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", conv)
       
        # Fully Connected Layer 1
        with tf.name_scope('fc1') as scope:
            weights = tf.Variable(tf.truncated_normal([(9 * 9 * nF2), nFc1], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[nFc1]), name='biases')
            pool_flat = tf.reshape(pool, shape=[-1, (9 * 9 * nF2)])
            fc = tf.matmul(pool_flat, weights)
            fc = tf.nn.bias_add(fc, biases)
            relu = tf.nn.relu(fc, name=scope)
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", fc)
            fc = tf.nn.dropout(relu, args.keep_prob)
        
        # Fully Connected Layer 2
        with tf.name_scope('fc2') as scope:            
            weights = tf.Variable(tf.truncated_normal([nFc1, n_classes], mean=0.0, stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='biases')
            fc = tf.matmul(fc, weights)
            fc = tf.nn.bias_add(fc, biases)
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", fc)
            prediction = tf.nn.softmax(fc, name=scope)
        
        # Backpropagation
        # measure of error of our model
        # this needs to be minimised by adjusting W and b
        with tf.name_scope("loss"):
            # this needs to be minimised by adjusting weights and biases
            # 1e-8 will help in overcoming this numerical instability.
            loss = -tf.reduce_sum(Y * tf.log(prediction + 1e-8), name='loss')
            tf.summary.scalar("loss", loss)
        
        with tf.name_scope("train"):
            # define training step which minimises cross entropy
            train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
        
        with tf.name_scope("accuracy"):
            # argmax gives index of highest entry in vector (1st axis of 1D tensor)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1), name='correct_prediction')
            
            # get mean of all entries in correct prediction, the higher the better
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
            
            tf.summary.scalar("accuracy", accuracy)
                
        merged_summary = tf.summary.merge_all()

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())


    """Train CNN for a number of steps."""    
    # Tell TensorFlow that the model will be built into the default Graph.
    # Create a session for running operations in the Graph.
    with tf.Session(graph=graph) as sess:
        
        st = time.time()
            
        # Initialize the variables (the trained variables and the epoch counter).
        sess.run(init_op)

        # Here we initialize the iterator with the training set.
        # This means that we can go through an entire epoch until the iterator becomes empty.
        sess.run(train_init_op)
        sess.run(test_init_op)
    
        # save all of ckeckpoints (a checkpoint per epoch)
        saver = tf.train.Saver(max_to_keep=None)
        
        writer = tf.summary.FileWriter(args.log_dir)
        writer.add_graph(sess.graph)
        
        log_cnt = 0
        display_step = 5
        for epoch in range(args.n_epochs):
            for itr in range(n_train_samples//args.batch_size):
                
                # fetch the batch train images and labels
                batch_x, batch_y = sess.run([train_images, train_labels])
                sess.run([train_op], feed_dict={X: batch_x, Y: batch_y})

                if ((itr+1) % display_step == 0):
                    # Evaluate model on training batch
                    acc_val, loss_val, summary = sess.run([accuracy, loss, merged_summary], feed_dict={X: batch_x, Y: batch_y})
                    log_cnt = log_cnt + 1
                    writer.add_summary(summary, log_cnt)
                    print('epoch %d/%d: , minibatch loss = %.3f , minibatch acc = %.3f' % (epoch, args.n_epochs, loss_val, acc_val))
            
            
            # save all variables for each epoch
            saver.save(sess, args.log_dir + args.checkpoint_name, global_step=epoch)
            
            # fetch whole test images and labels
            # feed the model with 10 batch of test images and labels
            n_batch_for_test = 10
            test_loss, test_acc = 0, 0
            for _ in range(n_batch_for_test):
                batch_x, batch_y = sess.run([test_images, test_labels])
                acc_val, loss_val = sess.run([accuracy, loss], feed_dict={X: batch_x, Y: batch_y})
                test_acc += acc_val
                test_loss += loss_val
            test_acc /= n_batch_for_test
            test_loss /= n_batch_for_test
            print('epoch %d/%d: , test loss = %.3f , test accuracy = %.3f' % (epoch, args.n_epochs, test_loss, test_acc))


        et = time.time()
        duration =  et - st
        print(int(duration) + ' s')




if __name__ == '__main__':
    
    args = parser.parse_args()

    # evaluate mean vector value for there layer images. Each mean value for each layer.
    DATASET_MEAN = dataset_mean.mean(args.dataset_dir)

    main(args)

