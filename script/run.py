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

# For Google Drive imports
sys.path.append('/content/drive/MyDrive/ml_epfl/ml_road_segmentation/script/')

#####################################################################################################################################
######################################################### CONSTANTS #################################################################
#####################################################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_PATH ="/content/drive/MyDrive/ml_epfl/ml_road_segmentation/checkpoint/current_checkpoint.pt"
BEST_MODEL_PATH ="/content/drive/MyDrive/ml_epfl/ml_road_segmentation/checkpoint/best_model.pt"
NBR_EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
GAMMA = 1
K_FOLD = 4
VALIDATION_SET_IDX = 0
BATCH_SIZE_VAL = 5
SEED = 0
DIM_IMG_TRAIN = 400

