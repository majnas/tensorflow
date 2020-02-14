#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/
# date: 2020-February-14
#------------------------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
import os
from config import Config as cfg


plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 0})

marker_list = cfg.CLASS_NAMES

dataset = [('train', cfg.N_TRAIN_PER_CLASS),
           ('test',  cfg.N_TEST_PER_CLASS)]

for split, n_samples in dataset:
    for marker in marker_list:
        print("making", split, marker, "dataset ...")
        mkr = "$\\"+marker+"$"
        
        # file_name_prefix = './dataset/' + part[0] + "/" + marker
        file_name_prefix = os.path.join(cfg.DATA_DIR, split, marker)

        if not os.path.exists(file_name_prefix):
            os.makedirs(file_name_prefix)
            
        for i in range(n_samples):
            figsize = 1 + np.random.random(2)
            fig = plt.figure(figsize=figsize)
            plt.axes([0,0,1,1], frameon=False)
            x, y = 10*np.random.rand(2, 50)
            clr = np.random.random((4,))
            s = np.random.randint(10, 300)
            plt.scatter(x,y, marker=mkr, s=s, color=clr)
            plt.savefig(file_name_prefix + '/image_'+str(i)+'.png', dpi=100)
            plt.close()
