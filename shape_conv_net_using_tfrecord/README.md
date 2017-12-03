
## Training a convolutional neural network using train/test tfrecord files

In this folder there are codes related to training a convolutional neural network to classify images. In the dataset a triangle or sqaure is in each image, then we train a CNN to classify what's the class of image (triangle or square). In order to feed the network first we converted the train and test parts of dataset (using ) to shape_train.tfrecord and shape_test.tfrecord.


### How to use:

Suppose we have a dataset contains images related to triangles and squares in two parts (train and test). We must place images in their corrosponding folders to make correct tfrecord files (shape_train.tfrecord and shape_test.tfrecord)

`python shape_tfwrite.py` to make tfrecord file.
