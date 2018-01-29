### How to use:
 
Suppose we have a dataset contains images related to different category in two parts (train and test). In each parts there is a folder named as corresponding category name (category1, category2, ...) and contain images related to folder's names. We must place images in their corrosponding folders to make correct labels for each image sample. The whole data is stored on disk; each m category has its own folder on disk, and n images for each category are stored as png files in the category folder. In other words, the directory structure looks something like this:

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
                     
 
In this example we have prepared a python code to make a synthatic image dataset. Using this code its easy to make dataset for testing the example code for reading dataset.

run `$ python make_dataset.py` to make synthatic dataset.
run `$ python read_dataset.py` to define a queue in the background to read images from dataset folder and restore them in dataset_restored folder.
