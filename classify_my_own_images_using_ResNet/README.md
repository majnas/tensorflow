Classify our own images using Resnet_50. 
In fact this code is to show how to use pretrained popular architectures as feature 
extractor and place multiple layers on top of them for own application. In this code we 
used Resnet_50 as backend, obviously it can be replaced with others like (ResNet, ...).
    
* The whole pipline of the project
    >>> Creating train and test splits of our dataset.
    here we create an synthatic image dataset of seven symbols. Symbol names are, "spadesuit",
    "clubsuit", "Join", "Omega", "Phi", "Psi", "heartsuit". this dataset includes images 
    with diffrent sizes. obviously this dataset can be replaced with other datasets if the 
    dataset structure remain inact. Dataset structure is as follows.

      ------- dataset
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

>>> Making a list of images and labels of train and test split.
    If you used your own dataset with same structure as me, using make_images_and_labels_lists.py
    with few changes can make a list of all images and labels.

>>> Loading pretrained architecture weights and placeing some layers on top of it. then train
    the model in two diffrent scenario. Finetuning backend weight or use the weight as is. these
    two scenarios will lead two different results. it's clear that tuning backend will result 
    better performance.

In this code we used tf.data api to read images and labels from folders. And we used Keras an 
high level api form tensorflow 1.x to build our sequentioal model. 

### Examples of synthatic dataset
spadesuit
![alt text](https://github.com/m-nasiri/tensorflow-2.0/blob/master/Classify_My_Own_Images_Using_MobileNet_Backend/images/image_0.png)
<br />
clubsuit
![alt text](https://github.com/m-nasiri/tensorflow-2.0/blob/master/Classify_My_Own_Images_Using_MobileNet_Backend/images/image_1.png)
<br />
Join
![alt text](https://github.com/m-nasiri/tensorflow-2.0/blob/master/Classify_My_Own_Images_Using_MobileNet_Backend/images/image_2.png)
<br />
Omega
![alt text](https://github.com/m-nasiri/tensorflow-2.0/blob/master/Classify_My_Own_Images_Using_MobileNet_Backend/images/image_3.png)
<br />
Phi
![alt text](https://github.com/m-nasiri/tensorflow-2.0/blob/master/Classify_My_Own_Images_Using_MobileNet_Backend/images/image_4.png)
<br />
Psi
![alt text](https://github.com/m-nasiri/tensorflow-2.0/blob/master/Classify_My_Own_Images_Using_MobileNet_Backend/images/image_5.png)
<br />
heartsuit
![alt text](https://github.com/m-nasiri/tensorflow-2.0/blob/master/Classify_My_Own_Images_Using_MobileNet_Backend/images/image_6.png)
<br />


### How to use:
Change model and training parameters in config.py script.

1 - Making image dataset

    python create_symbol_texture_image_dataset.py

2- Make a text file list of all images and labels.
    
    python make_images_and_labels_lists.py

3- Training model with or without backend fine-tuning.
    
    python train.py

