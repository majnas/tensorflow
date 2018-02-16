#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/blob/master/basic/classification.py
# date: 2018-Feb-11            
#-----------------------------------------------------------------------------#

"""
    basic classification
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# create dataset class 0
C0 = np.ones((100,2))
Cn = np.random.random((100,2))
C0 = C0 + Cn
L0 = np.zeros((100,1))

# create dataset class 1
C1 = 2* np.ones((100,2))
Cn = np.random.random((100,2))
C1 = C1 + Cn
L1 = np.ones((100,1))

# merge dataset
samples = np.concatenate((C0, C1))
labels = np.concatenate((L0, L1))

samples = samples.astype(np.float32)
labels = labels.astype(np.float32)

# shuffle dataset
dataset = list(zip(samples, labels))
np.random.shuffle(dataset)

train_X = [t[0] for t in dataset]
train_X = np.asarray(train_X)
train_Y = [t[1] for t in dataset]
train_Y = np.asarray(train_Y)

n_train = train_X.shape[0]

# placeholders
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# variables
w = tf.Variable(tf.random_normal([2, 1]), tf.float32)
b = tf.Variable(tf.random_normal([1]), tf.float32)

# pred = X * W + b
pred = tf.matmul(X, w) + b

logit = tf.nn.sigmoid(pred)

# loss
loss = tf.reduce_mean(tf.pow((logit - Y), 2))

# optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# accuracy
cond = tf.greater_equal(logit, 0.5)
cond_true = tf.ones(tf.shape(Y))
cond_false = tf.zeros(tf.shape(Y))
logit1 = tf.where(cond, cond_true, cond_false)
pred_true = tf.equal(logit1, Y)
pred_true = tf.cast(pred_true, tf.float32)
accuracy = tf.reduce_mean(pred_true)


# training variables
n_epochs = 100
batch_size = 1
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
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict=feed_dict)
        
        print('epoch= ', epoch, 'minibatch loss= ', loss_val, 'minibatch acc= ', acc_val)
        
    
    w_val, b_val = sess.run([w,b])
    x0 = np.linspace(0,3,50)
    #x0 *w0 + x1 * w1 + b = 0
    #x0 *w_val[0] + x1 * w_val[1] + b_val = 0
    x1 = -(b_val + x0 *w_val[0]) / w_val[1]
    plt.plot(x0, x1)
    
    plt.plot(C0[:, 0], C0[:, 1], 'ro')
    plt.plot(C1[:, 0], C1[:, 1], 'bo')
        
        
        


