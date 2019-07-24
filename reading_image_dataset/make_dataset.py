#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 This code is for making image dataset comprised of train and test sections 
 including different category. In each image there is an polygon with n 
 vertices. Number of vertices of shape in image is the name of category.
 Categories are created according to polygon_list variable, and the number of 
 image for each category is n_samples_per_category.
"""
#import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Turn interactive plotting off
plt.ioff()

polygon_list =  [3,4,5,6]
n_train_samples_per_category = 16
n_test_samples_per_category = 4
n_train = len(polygon_list) * n_train_samples_per_category
n_test = len(polygon_list) * n_test_samples_per_category


# make train dataset
for n_poly in polygon_list:
    dataset_dir = "./dataset/train/"+str(n_poly)+"/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    for sample in range(n_train_samples_per_category):
        polygon_radius = (0.25 - 0.15) * np.random.random(1) + 0.15   # 0.15 ~ 0.25
        polygon_center = np.random.random((2,))
        polygon_orientation = np.random.random()
        polygon_color = (1.0 - 0.5) * np.random.random((4,)) + 0.5   # darker colors
            
        patch = patches.RegularPolygon(polygon_center, 
                                       n_poly,
                                       polygon_radius,
                                       facecolor= polygon_color,
                                       orientation= polygon_orientation
                                       )
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_patch(patch)
        ax.set_axis_off()
        
        fig_name = dataset_dir+'image_'+str(sample)+'.png'
        fig.savefig(fig_name, dpi=90, bbox_inches='tight')
        plt.close("all")
        print('writing '+dataset_dir+'image_'+str(sample)+'.png ...')
     
        
# make test dataset
for n_poly in polygon_list:
    dataset_dir = "./dataset/test/"+str(n_poly)+"/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    for sample in range(n_test_samples_per_category):
        polygon_radius = (0.25 - 0.15) * np.random.random(1) + 0.15   # 0.15 ~ 0.25
        polygon_center = np.random.random((2,))
        polygon_orientation = np.random.random()
        polygon_color = (1.0 - 0.5) * np.random.random((4,)) + 0.5   # darker colors
            
        patch = patches.RegularPolygon(polygon_center, 
                                       n_poly,
                                       polygon_radius,
                                       facecolor= polygon_color,
                                       orientation= polygon_orientation
                                       )
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_patch(patch)
        ax.set_axis_off()
        
        fig_name = dataset_dir+'image_'+str(sample)+'.png'
        fig.savefig(fig_name, dpi=90, bbox_inches='tight')
        plt.close("all")
        print('writing '+dataset_dir+'image_'+str(sample)+'.png ...')

