
#-----------------------------------------------------------------------------#
#coder: Majid Nasiri
#github: https://github.com/m-nasiri/tensorflow/save_and_restore_models
#date: 2017-December-11
#-----------------------------------------------------------------------------#

"""
    This code is for restoring a trained convolutional network neural and 
    getting placeholder tensors and reuse it.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# one of meta and checkpoint files
meta_file = "./model/model.ckpt-0.meta"
ckpt_file = "./model/model.ckpt-0"

# load mnist dataset
mnist = input_data.read_data_sets("./data/", one_hot=True)


tf.reset_default_graph() 
sess = tf.Session()

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(meta_file)
saver.restore(sess, ckpt_file)

# List of all tensors
print_tensors_in_checkpoint_file(file_name=ckpt_file, tensor_name='', all_tensors='')

# getting placeholders variables
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
accuracy = graph.get_tensor_by_name("accuracy:0")


# fetch whole test images and labels
feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}

# feed the model with all test images and labels
acc_val = sess.run(accuracy, feed_dict=feed_dict)
print('Test Accuracy:', acc_val)



