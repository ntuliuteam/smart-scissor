"""Preprocessing for ImageNet"""
import numpy
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

import os
import re
import numpy as np
from PIL import Image
import cv2
import math

import data.transforms as transforms
from core.utils import make_divisible
from configs.config import cfg
from core.image import get_interpolation


# Per-channel mean and standard deviation values on ImageNet (in RGB order)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Constants for lighting normalization on ImageNet (in RGB order)
_EIG_VALS = [[0.2175, 0.0188, 0.0045]]
_EIG_VECS = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]
_PCA_STD = 0.1
# Make the resolution divisible by 4 for higher execution efficiency.
ROUND = 4


class ImageNet(Dataset):
    """ImageNet (224x224)"""

    def __init__(self, root, split, cropped=False):
        assert os.path.exists(root), "Data path '{}' not found.".format(root)
        splits = ['train', 'val']
        assert split in splits, "Split '{}' not supported for ImageNet.".format(split)

        self._root = root
        self._split = split
        self._augment = cfg.TRAIN.AUGMENT
        self._res_mult = cfg.SCALING.RES_MULT_A
        self._cropped = cropped
        self._flip_ratio = 0.5
        self._train_crop = 'random'
        self._interpolation = get_interpolation(cfg.SCALING.INTERPOLATE)
        self.train_size = make_divisible(224 * self._res_mult, ROUND)
        self.test_size = make_divisible(256 * self._res_mult, ROUND)
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the image database."""
        # Compile the split data path
        split_path = os.path.join(self._root, self._split)
        # Images are stored per class in subdirs (format: n<number>)
        split_folders = os.listdir(split_path)
        self._class_ids = sorted(f for f in split_folders if re.match(r"^n[0-9]+$", str(f)))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image database
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)

            im_names = os.listdir(im_dir)
            im_names.sort()
            for im_name in im_names:
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({'im_path': im_path, 'class': cont_id})

        # print("|| Number of {} images: {}".format(self._split, len(self._imdb)))
        # print("|| Number of {} classes: {}".format(self._split, len(self._class_ids)))

    def _preprocess_img(self, img):
        """Preprocess the image for network input.

        :param img: A image in PIL format.
        """
        if self._split == 'train':
            # For training use random_sized_crop, horizontal_flip, augment, lighting
            if self._train_crop == 'random':
                img = transforms.random_resized_crop(img, self.train_size, interpolation=self._interpolation)
            else:
                img = transforms.scale_and_center_crop(img, self.train_size, self.train_size)
            img = transforms.horizontal_flip(img, prob=self._flip_ratio)
            img = transforms.augment(img, self._augment)
            img = transforms.lighting(img, _PCA_STD, _EIG_VALS, _EIG_VECS)
        else:
            # For testing use scale and crop
            if self._cropped:
                # Only scale for cropped images
                img = transforms.scale_and_center_crop(img, self.train_size, self.train_size)
            else:
                # For normal images
                img = transforms.scale_and_center_crop(img, self.test_size, self.train_size)

        # transforms.lighting returns a np.array, while scale_and_center_crop returns a PIL Image
        if Image.isImageType(img):
            img = np.asarray(img).astype(np.float32) / 255
        # For training and testing use color normalization
        img = transforms.color_norm(img, _MEAN, _STD)
        # Convert HWC/RGB/float to CHW/RGB/float format
        img = img.transpose([2, 0, 1])

        return img

    def __getitem__(self, index):
        # Load PIL image
        img = Image.open(self._imdb[index]['im_path'])
        img = img.convert('RGB')
        # Preprocess the image for training and testing
        img = self._preprocess_img(img)
        # Retrieve the label
        label = self._imdb[index]['class']

        return img, label

    def __len__(self):
        return len(self._imdb)


class ImageNetForCrop(ImageNet):
    def __init__(self, root, split):
        super(ImageNetForCrop, self).__init__(root, split, cropped=True)
        self._train_crop = 'center'
        self._flip_ratio = 0.0


if __name__ == '__main__':
    dataset = ImageNet(cfg.DATASET.PATH + '/imagenet', 'val')
    print(dataset.__getitem__(0))

