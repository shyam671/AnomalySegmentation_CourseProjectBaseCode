# Code with dataset loader for VOC12 and Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

# allowed image extensions
EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    # read an image
    return Image.open(file)

def is_image(filename):
    # check is the given filename is a valid image file
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    # check if the given filename is a valid label file
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    # construct an image path
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    # construct an image path for cityscapes files
    return os.path.join(root, f'{name}')

def image_basename(filename):
    # get the image filename without extension
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        # define the root directory containing all images
        self.images_root = os.path.join(root, 'images')
        # define the root directory containing all labels
        self.labels_root = os.path.join(root, 'labels')

        # take all image filenames 
        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        # sort the image filenames
        self.filenames.sort()

        # define input transformations
        self.input_transform = input_transform
        # define target transformations
        self.target_transform = target_transform

    def __getitem__(self, index):
        # get a specific filename
        filename = self.filenames[index]

        # read the corresponding RGB image
        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        # read the corresponding label (palettised) image
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        # perform a input transformation if any
        if self.input_transform is not None:
            image = self.input_transform(image)
        # perform a label transformation if any
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class cityscapes(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset='val'):
        
        # define the root directory containing all images
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)
        # define the root directory containing all labels
        self.labels_root = os.path.join(root, 'gtFine/' + subset)
        print(self.images_root, self.labels_root)

        # take all image filenames (dirpath + filename)
        self.filenames = [os.path.join(dp, f) for dp, _, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        # sort the image filenames
        self.filenames.sort()

        # take all label filenames (dirpath + filename)
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        # sort the label filenames
        self.filenamesGt.sort()

        # define input image transformations
        self.input_transform = input_transform
        # define target label transformations
        self.target_transform = target_transform

    def __getitem__(self, index):
        # get a specific image filename
        filename = self.filenames[index]
        # get a specific label filename
        filenameGt = self.filenamesGt[index]

        #print(filename)

        # read the corresponding RGB image
        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        # read the corresponding label (palettised) image
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        # perform a input image transformation if any
        if self.input_transform is not None:
            image = self.input_transform(image)
        # perform a target label transformation if any
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename, filenameGt

    def __len__(self):
        return len(self.filenames)

