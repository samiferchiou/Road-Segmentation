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
######################################################### FUNCTIONS #################################################################
#####################################################################################################################################
def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def save_ckp(model, state, is_best, checkpoint_path, best_model_path):
    """
    state:            checkpoint we want to save
    is_best:          boolean to indicates if it is the best checkpoint
    checkpoint_path:  path to save checkpoint
    best_model_path:  path to save best model
    """
    torch.save(state, checkpoint_path)
    # if it is a best model, save the model's weights
    if is_best:
        torch.save(model.state_dict(), best_model_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model:           model that we want to load checkpoint parameters into       
    optimizer:       optimizer we defined in previous training
    
    Return model, optimizer, scheduler, epoch value, f1 score
    """
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Get epoch number, f1_max, scheduler
    epoch = checkpoint['epoch']
    f1_max = checkpoint['f1_max']
    scheduler = checkpoint['scheduler']
    f1_validation = checkpoint['f1_validation']
    acc_validation = checkpoint['acc_validation']
    f1_training = checkpoint['scheduler']
    acc_training = checkpoint['acc_training']
    return model, optimizer, scheduler, epoch, f1_max, f1_validation, acc_validation, f1_training, acc_training


def compute_scores(model, loader):
    #computing F1 score on validation data
    tp, fp, fn, tn = 0, 0, 0, 0
    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        prediction = torch.argmax(output, dim = 1)
        confusions = prediction / target
        tp += torch.sum(confusions == 1).item()
        fp += torch.sum(confusions == float('inf')).item()
        fn += torch.sum(confusions == 0).item()
        tn += torch.sum(confusions == float("nan")).item()
    f1_score = 2 * tp / (2 * tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp+ fn)
    return f1_score, accuracy


def train(n_epochs, data_loader, model, optimizer, scheduler, criterion, device, checkpoint_path, best_model_path, f1_init):
    train_loader = data_loader
    validation_loader = train_loader.dataset.get_validation_dataloader()
    f1_max = f1_init
    
    save_f1_validation = []
    save_acc_validation = []
    save_f1_training = []
    save_acc_training = []

    for epoch in range(n_epochs):
        start_time_epoch = time.time()
        loss_list = []
        
        # Train the model for one epoch
        model.train()
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_epoch = np.mean(loss_list)

        model.eval()
        #computing scores on validation data
        f1_score_val, accuracy_score_val = compute_scores(validation_loader)
        save_f1_validation.append(f1_score_val)
        save_acc_validation.append(accuracy_score_val)

        #computing scores on training data
        f1_score_train, accuracy_score_train = compute_scores(train_loader)
        save_f1_training.append(f1_score_train)
        save_acc_training.append(accuracy_score_train)

        end_time_epoch = time.time()

        # Prepare saving of the model
        checkpoint = {
            'epoch': epoch,
            'f1_max': f1_max,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
            'f1_validation' : save_f1_validation,
            'acc_validation' : save_acc_validation,
            'f1_training' : save_f1_training,
            'acc_training' : save_acc_training,
        }

        if f1_score_val > f1_max:
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            f1_max = f1_score_val
        
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        print(f"Epoch {epoch} || Loss:{loss_epoch:.6f} || Training F1 {f1_score_train:.6f} || Training Accuracy {accuracy_score_train:.6f} || Validation F1 {f1_score_val:.6f} || Validation Accuracy {accuracy_score_val:.6f} || Max F1 {f1_max:.6f} || LR {scheduler.get_last_lr()[0]:.6f} || Duration {format_time(end_time_epoch-start_time_epoch)}"+"\n")  
        scheduler.step()