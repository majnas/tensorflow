## tfrecord

In this folder there are codes related to reading dataset images and converting them to tfrecord file at the firt step, then reading the tfrecord and restore dataset images with two different batch sizes.

### How to use:

Suppose we have a dataset contains images related to triangles and squares in two parts (train and test). We must place images in their corrosponding folders to make correct tfrecord files (shape_train.tfrecord and shape_test.tfrecord)

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


`python shape_tfwrite.py` to make tfrecord file.

`python shape_tfread.py` to read tfrecord and restore dataset with differnt batches in dataset_restored1 (batch_size = 4) and dataset_restored2 (batch_size = 8).

