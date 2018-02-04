#------------------------------------------------------------------------------------------#
#coder: Majid Nasiri
#github: https://github.com/m-nasiri/tensorflow/tree/master/symbol_texture_image_conv_net
#date: 2018-Feb-04
#------------------------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
import os

plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 0})

marker_list = ["spadesuit",
               "clubsuit",
               "Join",
               "Omega",
               "Phi",
               "Psi",
               "heartsuit",
               ]

dataset = [('train', 800),
           ('test',  200)]

for part in dataset:
    for marker in marker_list:
        print("making", part[0], marker, "dataset ...")
        mkr = "$\\"+marker+"$"
        
        file_name_prefix = './dataset/' + part[0] + "/" + marker
        if not os.path.exists(file_name_prefix):
            os.makedirs(file_name_prefix)
            
        for i in range(part[1]):
            figsize = 1 + np.random.random(2)
            fig = plt.figure(figsize=figsize)
            plt.axes([0,0,1,1], frameon=False)
            x, y = 10*np.random.rand(2, 50)
            clr = np.random.random((4,))
            s = np.random.randint(10, 300)
            plt.scatter(x,y, marker=mkr, s=s, color=clr)
            plt.savefig(file_name_prefix + '/image_'+str(i)+'.png', dpi=100)
