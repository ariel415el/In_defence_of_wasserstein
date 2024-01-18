import itertools
import os
from random import shuffle

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

import torch.nn.functional as F
from utils.data import get_transforms


def get_data(data_root, im_size, center_crop, gray_scale, limit_data=None):
    """Load entire dataset to memory as a single batch"""
    T = get_transforms(im_size, center_crop, gray_scale)

    img_names = sorted(os.listdir(data_root))
    if limit_data is not None:
        img_names = img_names[:limit_data]

    images = torch.zeros(len(img_names), 1 if gray_scale else 3, im_size, im_size)
    print("Loading data to memory to find NNs")
    for i, fname in enumerate(tqdm(img_names)):
        im = Image.open(os.path.join(data_root, fname))
        im = T(im)
        images[i] = im

    return images


def cut_around_center(img, center, size, margin=0):
    hs = size // 2
    r = img.shape[-2]
    c = img.shape[-1]
    crop = img[..., max(0, center[0] - hs - margin): min(r, center[0] + hs + margin),
                    max(0, center[1] - hs - margin): min(c, center[1] + hs + margin)]
    return crop


def sample_patch_centers(img_dim, p, n_centers, stride=1, offset=0):
    h = p // 2
    centers = np.arange(h, img_dim - h - 1, stride) + offset
    centers = list(itertools.product(centers, repeat=2))
    shuffle(centers)
    centers = centers[:n_centers]
    return centers