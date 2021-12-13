import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import torch
import torchvision.transforms as T
from pathlib import Path

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

#def load_train_dataset():
#    # Loaded a set of images
#    root_dir = "../data/training/"
#    image_dir = root_dir + "images/"
#    gt_dir = root_dir + "groundtruth/"
#    files = os.listdir(image_dir)
#    n = len(files)
#    to_tensor = T.ToTensor()
#    imgs = [to_tensor(Image.open(image_dir + files[i])) for i in range(n)]
#    gt_imgs = [to_tensor(Image.open(gt_dir + files[i])).type(torch.LongTensor) for i in range(n)]
#    return (imgs, gt_imgs)

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



# FUNCTIONS FOR DATASET AUGMENTATION

#Constants
DIM_IMG = 400
DIM_IMG_CROP=DIM_IMG//2
NB_ROT = 3
ANGLE_ROTATION = 30

#Horizontal flip
def hor_flip(imgs):
    hflip_imgs = [T.functional.hflip(imgs[i]) for i in range(len(imgs))]
    return hflip_imgs

#Vertical flip
def vert_flip(imgs):
    vflip_imgs = [T.functional.vflip(imgs[i]) for i in range(len(imgs))]
    return vflip_imgs
    
#Rotation
def rotation(imgs):
    rot_imgs = [T.functional.rotate(imgs[i], ANGLE_ROTATION*(i+1), expand=False)
                for _ in range(NB_ROT)
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

# MAYBE TO REMOVE ?
def change_color_imgs(imgs):
    return [T.Grayscale()(imgs[i]) for i in  range(n)]

def blur_imgs(imgs):
    return [T.functional.gaussian_blur(imgs[i], kernel_size=(5, 9)) for i in range(len(imgs))]

def jitter_imgs(imgs):
    jitter = T.ColorJitter(brightness=[2,2], contrast=0, saturation=0, hue=0)
    return [jitter(imgs[i]) for i in  range(len(imgs))]
# END OF REMOVAL

def compose_all_functions_for_data(imgs):
    h = imgs + hor_flip(imgs)
    v = h + vert_flip(h)
    c = v + crop(v)
    r = c + rotation(c)
    return r 