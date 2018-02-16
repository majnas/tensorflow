#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/blob/master/basic/regression.py
# date: 2018-Feb-11            
#-----------------------------------------------------------------------------#

"""
    basic regression
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_train = train_X.shape[0]
# Y = w * X + b

# placeholders
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# variables
w = tf.Variable(tf.random_normal([1]), tf.float32)
b = tf.Variable(tf.random_normal([1]), tf.float32)

# pred = w * X + b
pred = tf.multiply(w, X) + b

# loss
loss = tf.reduce_mean(tf.pow((pred - Y), 2))

# optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# training variables
n_epochs = 100
batch_size = 2
n_itrs = n_train // batch_size

with tf.Session() as sess:
    
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs):
        for itr in range(n_itrs):
            
            batch_x = train_X[(itr*batch_size):((itr+1)*batch_size)]
            batch_y = train_Y[(itr*batch_size):((itr+1)*batch_size)]

            feed_dict = {X: batch_x, Y: batch_y}
            sess.run(opt, feed_dict=feed_dict)
        
        # evaluate in each epoch
        feed_dict = {X: train_X, Y: train_Y}
        loss_val = sess.run(loss, feed_dict=feed_dict)
        
        print('epoch= ', epoch, 'minibatch loss= ', loss_val)
    
    
    w_val, b_val = sess.run([w,b])
    x = np.linspace(-10,20,50)
    # pred = w * x + b
    pred_val = w_val * x + b_val
    plt.plot(x, pred_val)
    plt.plot(train_X, train_Y, 'ro')
    




