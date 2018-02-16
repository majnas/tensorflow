#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/blob/master/basic/mnist_mlp.py
# date: 2018-Feb-11            
#-----------------------------------------------------------------------------#

"""
    basic multilayer perceptron network to classify mnist dataset
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./MNIST', one_hot=True)

n_train = mnist.train.num_examples
n_test = mnist.train.num_examples

# network parameters
n_input = 784
n1 = 256
n2 = 64
n_classes = 10

# place holders
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

# First Layer
with tf.name_scope('fc1'):
    weights = tf.Variable(tf.random_normal([n_input, n1]), tf.float32, name='weights')
    biasses = tf.Variable(tf.random_normal([n1]), tf.float32, name='biasess')
    fc = tf.matmul(X, weights) + biasses
    sig = tf.nn.sigmoid(fc)
    
# Second Layer
with tf.name_scope('fc2'):
    weights = tf.Variable(tf.random_normal([n1, n2]), tf.float32, name='weights')
    biasses = tf.Variable(tf.random_normal([n2]), tf.float32, name='biasess')
    fc = tf.matmul(sig, weights) + biasses
    sig = tf.nn.sigmoid(fc)


# Third Layer
with tf.name_scope('fc3'):
    weights = tf.Variable(tf.random_normal([n2, n_classes]), tf.float32, name='weights')
    biasses = tf.Variable(tf.random_normal([n_classes]), tf.float32, name='biasess')
    fc = tf.matmul(sig, weights) + biasses
    logits = tf.nn.softmax(fc)
    
# loss
loss = -tf.reduce_mean(Y * tf.log(logits))

# optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# accuracy
accuracy = tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.cast(accuracy, tf.float32)
accuracy = tf.reduce_mean(accuracy)

# training variables
n_epochs = 10
batch_size = 128
n_itrs = n_train // batch_size

with tf.Session() as sess:
    
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs):
        for itr in range(n_itrs):
            
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            feed_dict = {X: batch_x, Y: batch_y}
            sess.run(opt, feed_dict=feed_dict)
        
        # evaluate in each epoch
        feed_dict = {X: mnist.test.images,
                     Y: mnist.test.labels}
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict=feed_dict)
        
        print('epoch= ', epoch, 'minibatch loss= ', loss_val, 'minibatch acc= ', acc_val)



