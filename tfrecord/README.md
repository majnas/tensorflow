In this folder there are codes related to reading dataset images and converting them to tfrecord file at the firt step then reading
the tfrecord and restore dataset images.

How to use:
Suppose we have a dataset contain images related to triangles and squares in two parts.
We must place images in their corrosponding folders to make correct tfrecord files
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
