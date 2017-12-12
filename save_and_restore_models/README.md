## Save and restore a model

In this folder there are codes related to training a deep neural network with two convolutional layers and two fully connected layers. In this code the aim is to show how to save a model in different epochs of training then restoring one of saved models and feed input to it and print the accuracy.

### How to use:

* first run `python train_and_save.py` to trian the network and save last 4 model checkpoint in "./model/" folder.
* second run `python restore_and_reuse.py` to restore the network from saved model and feed the network with all test images to calculate accuracy


