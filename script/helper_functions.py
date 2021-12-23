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


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


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


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def compute_conv_output_size(image_size, filter_size, stride = 1, padding = 0):
    return (image_size - filter_size + 2 * padding)/stride + 1

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

def compute_scores(model, loader, device):
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
        f1_score_val, accuracy_score_val = compute_scores(model, validation_loader, device)
        save_f1_validation.append(f1_score_val)
        save_acc_validation.append(accuracy_score_val)

        #computing scores on training data
        f1_score_train, accuracy_score_train = compute_scores(model, train_loader, device)
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
            save_ckp(model, checkpoint, True, checkpoint_path, best_model_path)
            f1_max = f1_score_val
        
        save_ckp(model, checkpoint, False, checkpoint_path, best_model_path)
        scheduler.step()

# FUNCTIONS FOR DATASET AUGMENTATION

#Constants
DIM_IMG = 400
DIM_IMG_CROP=DIM_IMG//2
NB_ROT = 6
ANGLE_ROTATION = 15

#Rotation with mirroring
def rotation(imgs, angle_rotation, nb_rot):
    rot_imgs = [torch.from_numpy(sc.rotate(imgs[i], angle_rotation*(idx_rot+1), axes =(1,2), reshape=False, mode ='mirror'))
                for idx_rot in range(nb_rot)
                for i in range(len(imgs))
               ]
    return rot_imgs

#Composing all transformations    
def compose_all_functions_for_data(imgs, angle_rotation, nb_rot):
    r = imgs + rotation(imgs, angle_rotation, nb_rot)
    return r
    