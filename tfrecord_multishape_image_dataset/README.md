## tfrecord for datasets with different size of images

In this folder there are codes related to reading dataset images and converting them to tfrecord file at the firt step, then reading the tfrecord and restore dataset images with two different batch sizes.

### How to use:

Suppose we have a dataset contains images related to triangles and squares in two parts (train and test). We must place images in their corrosponding folders to make correct tfrecord files (shape_train.tfrecord and shape_test.tfrecord)

    dataset
       '
       '---- train
       '       '
       '       '---- squares (train squares images .. image_0.png, image_1.png, ....)
       '       '---- triangles (train triangles images .. image_0.png, image_1.png, ....)
       '
       ' --- test 
               '
               '---- squares (test squares images .. image_0.png, image_1.png, ....)
               '---- triangles (test triangles images .. image_0.png, image_1.png, ....)


`python shape_multishape_tfwrite.py` to make tfrecord file.

`python shape_multishape_tfread.py` to read tfrecord and restore dataset.
