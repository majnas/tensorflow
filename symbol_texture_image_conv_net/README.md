### How to use:
 
In this [task](https://github.com/m-nasiri/tensorflow/tree/master/reading_image_dataset) we have learned how to define a queue in the background in order to read images from dataset folder. This folder is contains codes related to making synthatic symbol images dataset, defining queues to load image filenames, preprocess them with multiple threads and apply the preprocessing in parallel, and training a Convolutional Neural Network to classify input images which category they are belong to.

Running `$ python make_symbol_texture_image_dataset.py` will make a dataset. Dataset is contain images of symbols (spadesuit, clubsuit, Join, Omega, Phi, Psi, heartsuit) with different sizes, and each image is comprised of symbols of same type, with different colors and sizes. without doubt you can replace your own dataset as far as not breaking the structure. Dataset structure looks something like this.

     dataset
     '
     '---- train
     '       '
     '       '---- category_1
     '       '       '
     '       '       '---- image_1.png
     '       '       '---- image_2.png
     '       '       '---- ....
     '       '       '---- image_n.png
     '       '
     '       '---- category_2
     '       '       '
     '       '       '---- image_1.png
     '       '       '---- image_2.png
     '       '       '---- ....
     '       '       '---- image_n.png
     '       '
     '       '----  ...
     '       '       
     '       '       
     '       '
     '       '---- category_m
     '               '
     '               '---- image_1.png
     '               '---- image_2.png
     '               '---- ....
     '               '---- image_n.png
     '       
     ' --- test 
             '
             '---- category_1
             '       '
             '       '---- image_1.png
             '       '---- image_2.png
             '       '---- ....
             '       '---- image_n.png
             '
             '---- category_2
             '       '
             '       '---- image_1.png
             '       '---- image_2.png
             '       '---- ....
             '       '---- image_n.png
             '
             '----  ...
             '       
             '       
             '
             '---- category_m
                     '
                     '---- image_1.png
                     '---- image_2.png
                     '---- ....
                     '---- image_n.png
                     
 
Samples of images in datast.

![alt text](https://github.com/m-nasiri/tensorflow/blob/master/symbol_texture_image_conv_net/images/acc_loss.png)
