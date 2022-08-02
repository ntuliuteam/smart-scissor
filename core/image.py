import torch

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from data import transforms


def pil_preprocess(imgs, mean, std):
    inputs = []
    for img in imgs:
        img = np.asarray(img).astype(np.float32) / 255
        img = transforms.color_norm(img, mean, std)
        img = np.ascontiguousarray(img.transpose([2, 0, 1]))
        inputs.append(img)
    inputs = torch.tensor(inputs)

    return inputs


def save_imgs(imgs, img_names, savedir):
    assert len(imgs) == len(img_names), "Mismatch of the number of images and image names. " \
                                         "Got {} and {}, respectively.".format(len(imgs), len(img_names))
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    for idx, (img, name) in enumerate(zip(imgs, img_names)):
        savepath = os.path.join(savedir, name)
        img.save(savepath, quality=100, subsampling=0)


def get_interpolation(name):
    interpolations = {
        'bilinear': Image.BILINEAR
    }
    err_message = f"Interpolation {name} not supported."
    assert name in interpolations.keys()

    return interpolations[name]
