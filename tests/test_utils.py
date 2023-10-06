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

    images = []
    print("Loading data to memory to find NNs")
    img_names = sorted(os.listdir(data_root))
    # shuffle(img_names)
    if limit_data is not None:
        img_names = img_names[:limit_data]
    for fname in tqdm(img_names):
        im = Image.open(os.path.join(data_root, fname))
        im = T(im)
        images += [im]

    data = torch.stack(images)

    return data


def compute_dists(x, y, dist='edges'):
    if dist == "rgb":
        dists = (x - y)
    elif dist == "gray":
        dists = (x.mean(1) - y.mean(1))  # Compare gray scale images
    else:
        a = torch.Tensor([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]]).view((1, 1, 3, 3))
        dists = (F.conv2d(torch.mean(x, dim=1, keepdim=True), a) -
                 F.conv2d(torch.mean(y, dim=1, keepdim=True), a)
                 )

    return dists.reshape(x.shape[0], -1)


def cut_around_center(img, center, size, margin=0):
    hs = size // 2
    r = img.shape[-2]
    c = img.shape[-1]
    assert size + margin*2 < min(r,c), "Desired crop size + margin is biggere than the image itself"
    ys = center[0] - hs - margin
    ye = center[0] + hs + margin
    if ys < 0:
        ys = 0
        ye = size
    if ye > r:
        ye = r
        ys = r-size

    xs = center[1] - hs - margin
    xe = center[1] + hs + margin
    if center[1] - hs - margin < 0:
        xs = 0
        xe = size
    if center[1] + hs + margin > r:
        xs = r-size
        xe = r
    crop = img[..., xs:xe,ys:ye]
    return crop


def sample_patch_centers(img_dim, p, n_centers, stride=1, offset=0):
    h = p // 2
    centers = np.arange(h, img_dim - h + 1, stride) + offset
    centers = list(itertools.product(centers, repeat=2))
    shuffle(centers)
    centers = centers[:n_centers]
    return centers


def inverse_latent_sampling(G, z, real_images, n_steps=1000, lr=0.01):
    from torch import optim
    z.requires_grad_(True)
    optimizer_image = optim.Adam([z], lr=lr)

    for _ in range(n_steps):
        optimizer_image.zero_grad()
        g_images = G(z)
        rec_loss = F.mse_loss(g_images, real_images)
        rec_loss.backward()
        optimizer_image.step()

    return z.detach()