#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2020-January-16
#------------------------------------------------------------------------------------------#

import numpy as np
import os
import skimage.io as io


def mean(dataset_dir):    
    section_list = os.listdir(dataset_dir)    
    img_mean = [0, 0, 0]
    im_count = 0
    for sec in section_list:
        sec_dir = os.path.join(dataset_dir, sec)
        category_list = os.listdir(sec_dir)
        for cat in category_list:
            cat_dir = os.path.join(sec_dir, cat)
            file_list = os.listdir(cat_dir) 
            for im in file_list:
                im_count +=1
                filename = os.path.join(cat_dir, im)
                img = io.imread(filename)
                img = img[:,:,0:3]
                img_mean_ = np.mean(img, axis=0)
                img_mean_ = np.mean(img_mean_, axis=0)
                img_mean += img_mean_
                
    img_mean /= im_count 
    return list(img_mean)
           

