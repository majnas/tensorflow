#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/
# date: 2020-February-14
#------------------------------------------------------------------------------------------#

import numpy as np

# Base Configuration Class
class Config(object):

    # Class Names , LABLES, UNICODE (if provided and just for print)
    CLASS_NAMES =  ["spadesuit", "clubsuit", "Join", "Omega", "Phi", "Psi", "heartsuit"]
    CLASS_LABELS = [          0,          1,      2,       3,     4,     5,          6,]
    CLASS_UNICODES = [u"\u2663", u"\u2665", u"\u2A1D", u"\u03A9", u"\u03D5", u"\u03C8", u"\u2660"]

    # Number of classes
    N_CLASSES = len(CLASS_NAMES)

    # Dataset Directorty - Place your own dataset here or make dataset using make_symbol_texture_image_dataset.py
    DATA_DIR = "./dataset"

    # Number of samples per class for train and test splits
    N_TRAIN_PER_CLASS = 800
    N_TEST_PER_CLASS = 200

    # Total number of train and test split
    N_TRAIN = N_CLASSES * N_TRAIN_PER_CLASS
    N_TEST = N_CLASSES * N_TEST_PER_CLASS

    # Image Size
    IMG_HIGHT = 224
    IMG_WIDTH = 224

    # Training Parameters
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 10
    N_EPOCHS = 50

    # whether finetune backend architecture or leave the weights intact.
    FINETUNE = True 

    LEARNING_RATE = 1e-4

    RESNET_CKPT_DIR = "/media/deep/98929AF7929AD8D8/Architectures/ResNet/resnet_v2_50.ckpt"
    BATCH_NORM_DECAY = 0.9997
    WEIGHT_DECAY = 5e-4
    BASE_ARCHITECTURE = 'resnet_v2_50'
    FREEZE_BATCH_NORM = True


