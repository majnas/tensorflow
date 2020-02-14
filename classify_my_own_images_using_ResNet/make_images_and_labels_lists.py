#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/
# date: 2020-February-14
#------------------------------------------------------------------------------------------#

import glob
import os
from config import Config as cfg

print("unicode  labels   names")
for l,c,u in zip(cfg.CLASS_NAMES, cfg.CLASS_LABELS, cfg.CLASS_UNICODES):
    print("  %s ------ %d ----- %s" % (u, c, l))

n_train = 0
fh_image = open(os.path.join(cfg.DATA_DIR, "train_images_list.txt"), "w")
fh_label = open(os.path.join(cfg.DATA_DIR, "train_labels_list.txt"), "w")
for class_name, class_label in zip(cfg.CLASS_NAMES, cfg.CLASS_LABELS):
    path = os.path.join("./dataset/train/", class_name, "*")
    for file in glob.glob(path):
        n_train += 1  
        fh_image.write(file)
        fh_image.write("\n")
        fh_label.write(str(class_label))
        fh_label.write("\n")
fh_image.close()
fh_label.close()
print("n_train:", n_train)

n_test = 0
fh_image = open(os.path.join(cfg.DATA_DIR, "test_images_list.txt"), "w")
fh_label = open(os.path.join(cfg.DATA_DIR, "test_labels_list.txt"), "w")
for class_name, class_label in zip(cfg.CLASS_NAMES, cfg.CLASS_LABELS):
    path = os.path.join("./dataset/test/", class_name, "*")
    for file in glob.glob(path):
        n_test += 1
        fh_image.write(file)
        fh_image.write("\n")
        fh_label.write(str(class_label))
        fh_label.write("\n")
fh_image.close()
fh_label.close()
print("n_test:", n_test)
