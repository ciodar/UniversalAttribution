
import numpy as np
from functools import partial
from torchvision.transforms.v2 import *
from PIL import Image
from typing import Any, Dict, Sequence
import torch
import torchvision.transforms.v2.functional as F
import numbers
import PIL

# perturbations = {
#     'None': A.HorizontalFlip(p=0),
#     'JPEG': A.ImageCompression(quality_lower=10, quality_upper=90, p=0.5, always_apply=True),
#     'Blur': A.Blur(blur_limit=(3,15), p=0.5, always_apply=True),
#     'Noise': A.GaussNoise(var_limit=(10.0, 100.0), p=0.5, always_apply=True),
#     'Brightness': A.RandomBrightnessContrast(brightness_limit=(-0.5,0.5),contrast_limit=(0.,0.),p=0.5, always_apply=True),
#     'Contrast': A.RandomBrightnessContrast(brightness_limit=(0.,0.),contrast_limit=(-0.5,0.5),p=0.5, always_apply=True),
#     'Hue': A.HueSaturationValue(hue_shift_limit=(-0.5,0.5), sat_shift_limit=0, val_shift_limit=0, p=0.5, always_apply=True),
#     'Saturation': A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(0.1, 3.0), val_shift_limit=0, p=0.5, always_apply=True),
#     'Rotate': A.Rotate(limit=30, p=0.5, always_apply=True),
# }

def get_train_perturbations(config):
    perturbations = []
    if config.train_perturbations:
        if config.resize_range:
            perturbations.append(RandomShortestSize(min_size=config.resize_range[0],max_size=config.resize_range[1]))
        if config.blur_range:
            perturbations.append(GaussianBlur(kernel_size=config.blur_range, sigma=3.0))
        if config.jpeg_range:
            perturbations.append(JPEG(config.jpeg_range))
        if config.crop_range:
            perturbations.append(RandomResizedCrop(size=config.resize_size, scale=config.crop_range))
        return RandomChoice(perturbations)

def get_test_perturbations(config):
    perturbations = []
    if config.perturbations is None:
        return perturbations
    if "resize" in config.perturbations:
        start, stop = config.resize_range
        resize_value = np.linspace(start, stop, 5)[config.perturbation_intensity - 1]
        perturbations.append(Resize(size=int(resize_value)))
    if "blur" in config.perturbations:
        start, stop = config.blur_range
        blur_value = [i if i%2!=0 else i-1 for i in np.linspace(3, 15, 5)][config.perturbation_intensity - 1]
        perturbations.append(GaussianBlur(kernel_size=blur_value, sigma=3.0))
    if "jpeg" in config.perturbations:
        start, stop = config.jpeg_range
        jpeg_value = int(np.linspace(start, stop, 5)[config.perturbation_intensity - 1])
        perturbations.append(JPEG((jpeg_value,jpeg_value)))
    if "crop" in config.perturbations:
        start, stop = config.crop_range
        crop_value = np.linspace(start, stop, 5)[config.perturbation_intensity - 1]
        perturbations.append(RandomResizedCrop(size=config.resize_size, scale=(crop_value,crop_value)))
    return RandomChoice(perturbations)