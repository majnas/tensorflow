### How to use:
 
Suppose we have a dataset contains images related to triangles and squares in two parts (train and test). We must place images in their corrosponding folders to make correct tfrecord files (shape_train.tfrecord and shape_test.tfrecord)
 
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
                     
 
 
`python shape_tfwrite.py` to make synthatic dataset in dataset folder
 
`python shape_tfread.py` to read tfrecord and restore dataset with differnt batches in dataset_restored1 (batch_size = 4) and dataset_restored2 (batch_size = 8).
