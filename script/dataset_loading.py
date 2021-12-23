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


def append_filter(filter_to_apply, images):
    to_tensor = T.ToTensor() 
    toPIL = T.ToPILImage()
    
    concatenated_images = []

    for img in images:
      filtered_img = toPIL(img).filter(filter_to_apply)
      tensored_img = to_tensor(filtered_img)
      stacked_img = torch.cat((img, tensored_img), dim=0)
      concatenated_images.append(stacked_img)
    
    return concatenated_images
    
    
    
class imagesDataset(Dataset): 
    def __init__(self, K_fold, validation_set_idx, batch_size_val, seed, filter_img, angle_rotation, nb_rot):
        X, Y = load_train_dataset()

        #tresholding ground_truth values
        Y = [(y > 0.5).long() for y in Y]
        shape_y = Y[0].shape
        
        #shuffling
        random.seed(seed)
        idx_list = list(range(len(X)))
        random.shuffle(idx_list)
        random.seed()
        X = [X[idx] for idx in idx_list]
        Y = [Y[idx] for idx in idx_list]
        
        #K_fold separation
        fold_size = len(X) // K_fold
        start_validation_idx = validation_set_idx * fold_size
        end_validation_idx = start_validation_idx + fold_size
        self.X_train = X[slice(0, start_validation_idx)] + X[slice(end_validation_idx, None)]
        self.Y_train = Y[slice(0, start_validation_idx)] + Y[slice(end_validation_idx, None)]
        self.X_validation = X[slice(start_validation_idx, end_validation_idx)]
        self.Y_validation = Y[slice(start_validation_idx, end_validation_idx)]

        #data augmentation
        self.X_train = compose_all_functions_for_data(self.X_train, angle_rotation, nb_rot)
        self.Y_train = compose_all_functions_for_data(self.Y_train, angle_rotation, nb_rot)
        self.X_validation = compose_all_functions_for_data(self.X_validation, angle_rotation, nb_rot)
        self.Y_validation = compose_all_functions_for_data(self.Y_validation, angle_rotation, nb_rot)
        self.n_samples = len(self.X_train)

         #apply filter
        self.X_train = append_filter(filter_img, self.X_train)
        self.X_validation = append_filter(filter_img, self.X_validation)

        #2nd shuffling
        random.seed(seed)
        idx_list = list(range(len(self.X_train)))
        random.shuffle(idx_list)
        random.seed()
        self.X_train = [self.X_train[idx] for idx in idx_list]
        self.Y_train = [self.Y_train[idx] for idx in idx_list]
        
        #casting into tensors
        self.X_train = torch.stack(self.X_train)
        self.X_validation = torch.stack(self.X_validation)
        self.Y_train = torch.reshape(torch.stack(self.Y_train) , (-1, shape_y[1], shape_y[2]))
        self.Y_validation = torch.reshape(torch.stack(self.Y_validation) , (-1, shape_y[1], shape_y[2]))

        #creating dataloader for validation data
        class dataset_validation(Dataset):
            def __init__(s,x,y):
                s.x = x
                s.y = y
                s.size = len(s.x)
            def __getitem__(s, index):
                return s.x[index], s.y[index]
            def __len__(s):
                return s.size
               
        self.validation_data_loader = torch.utils.data.DataLoader(
            dataset_validation(self.X_validation, self.Y_validation),
            batch_size = batch_size_val, shuffle = False)
        
        
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        return self.n_samples
    
    def get_validation_dataloader(self):
        return self.validation_data_loader

