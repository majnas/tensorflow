
## Training a convolutional neural network using train/test tfrecord files

In this folder there are codes related to training a convolutional neural network to classify images. In the dataset a triangle or sqaure is in each image, then we train a CNN to classify what's the class of image (triangle or square). In order to feed the network first we converted the train and test parts of dataset (using ) to shape_train.tfrecord and shape_test.tfrecord binary files.

### How to use:

Make your own dataset and convert them to tfrecord using this [code](https://github.com/m-nasiri/tensorflow/tree/master/tfrecord), or use my tfrecord files (shape_train.tfrecord and shape_test.tfrecord)

`python shape_tfwrite.py` to train the network.
