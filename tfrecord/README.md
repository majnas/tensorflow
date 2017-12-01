# tfrecord

In this folder there are codes related to reading dataset images and converting them to tfrecord file at the firt step then reading the tfrecord and restore dataset images.

# How to use:

Suppose we have a dataset contain images related to triangles and squares in two parts (train and test). We must place images in their corrosponding folders to make correct tfrecord files (shape_train.tfrecord and shape_test.tfrecord)

First part training

    dataset
       '
       '---- train
       '       '
       '       '---- squares
       '       '---- triangles
       '
       ' --- test 
               '
               '---- squares
               '---- triangles

run shape_tfwrite.py to make tfrecord file
