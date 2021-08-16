import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(splits_folderpath):

    with open(os.path.join(splits_folderpath, 'habitat_train_depth.txt'), 'r') as f:
        train_filepaths = f.read().splitlines()

    with open(os.path.join(splits_folderpath, 'habitat_val_depth.txt'), 'r') as f:
        val_filepaths = f.read().splitlines()

    left_train  = [img for img in train_filepaths]
    right_train = [img.replace('left_rgb', 'right_rgb') for img in train_filepaths]
    depth_train_L = [img.replace('left_rgb', 'left_depth').replace('jpg', 'png') for img in train_filepaths]

    left_val  = [img for img in val_filepaths]
    right_val = [img.replace('left_rgb', 'right_rgb') for img in val_filepaths]
    depth_val_L = [img.replace('left_rgb', 'left_depth').replace('jpg', 'png') for img in val_filepaths]

    return left_train, right_train, depth_train_L, left_val, right_val, depth_val_L
