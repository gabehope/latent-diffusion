import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light

from torchvision.datasets import CIFAR10

class CIFARBase(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return dict(image=TF.to_tensor(img).permute(1, 2, 0) * 2. - 1., class_label=target, index=index)

class CIFARTrain(CIFARBase):
    def __init__(self, data_root=None, transform=None, target_transform=None, download=True):
        super().__init__(data_root, train=True, transform=transform, target_transform=target_transform, download=download)

class CIFARValidation(CIFARBase):
    def __init__(self, data_root=None, transform=None, target_transform=None, download=True):
        super().__init__(data_root, train=False, transform=transform, target_transform=target_transform, download=download)
