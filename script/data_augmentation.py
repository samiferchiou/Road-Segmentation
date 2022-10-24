#####################################################################################################################################
######################################################### IMPORTS ###################################################################
#####################################################################################################################################
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import random
import os, sys, time
from PIL import Image
import torchvision.transforms as T
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.ndimage as sc
from PIL import ImageFilter

from helper_functions import *
from data_augmentation import *
from training_helpers import *
from run import device

# For Google Drive imports
sys.path.append('/content/drive/MyDrive/ml_epfl/ml_road_segmentation/script/')



#####################################################################################################################################
######################################################### DATASET  ##################################################################
#####################################################################################################################################

#Constants
DIM_IMG = 400
DIM_IMG_CROP=DIM_IMG//2
NB_ROT = 20
ANGLE_ROTATION = 9


# Rotation with mirroring
def rotation(imgs):
    rot_imgs = [torch.from_numpy(sc.rotate(imgs[i], ANGLE_ROTATION*(idx_rot+1), axes =(1,2), reshape=False, mode ='mirror'))
                for idx_rot in range(NB_ROT)
                for i in range(len(imgs))
               ]
    return rot_imgs


#Crop and resize
def crop(imgs):
    cropped_imgs = [T.Resize(size=DIM_IMG)(T.FiveCrop(size=DIM_IMG_CROP)(imgs[i])[tuple_i])
                    for tuple_i in range(5)
                    for i in range(len(imgs))
                   ]
    return cropped_imgs


def compose_all_functions_for_data(imgs):
    #c = imgs + crop(imgs)
    #r = c + rotation(c)
    r = imgs + rotation(imgs)
    return r


def load_train_dataset():
    root_dir = "/content/drive/MyDrive/ml_epfl/ml_road_segmentation/data/training/"
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    files = os.listdir(image_dir)
    n = len(files)

    to_tensor = T.ToTensor()
    imgs = [to_tensor(Image.open(image_dir + files[i])) for i in range(n)]
    gt_imgs = [to_tensor(Image.open(gt_dir + files[i])) for i in range(n)]
    return (imgs, gt_imgs)