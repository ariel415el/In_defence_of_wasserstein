import os

import torch
from tqdm import tqdm
from PIL import Image

from utils.data import get_transforms


def get_data(data_root, im_size, center_crop, limit_data=None):
    """Load entire dataset to memory as a single batch"""
    T = get_transforms(im_size, center_crop)

    images = []
    print("Loading data to memory to find NNs")
    img_names = os.listdir(data_root)
    # shuffle(img_names)
    if limit_data is not None:
        img_names = img_names[:limit_data]
    for fname in tqdm(img_names):
        im = Image.open(os.path.join(data_root, fname))
        im = T(im)
        images += [im]

    data = torch.stack(images)

    if data.shape[1] == 1:
        data = data.repeat(1,3,1,1)

    return data