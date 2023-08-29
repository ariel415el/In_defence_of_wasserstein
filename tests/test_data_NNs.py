import itertools
import os
from random import shuffle

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from torchvision import utils as vutils
import torch.nn.functional as F

from losses.loss_utils import vgg_dist_calculator
from tests.test_utils import cut_around_center, compute_dists, sample_patch_centers


def imshow(img, axs, title="img"):
    cmap = 'gray' if img.shape[0] == 1 else None
    axs.imshow((img.permute(1, 2, 0).numpy() + 1) / 2, cmap=cmap)
    axs.axis('off')
    axs.set_title(title)


def search_for_nn_patches_in_locality(img, data, center, p, search_margin, dist="edge"):
    """
    for a given patch location and size, search the nearest patch from data (only in the same patch location) to
    the patch in img in that location"""
    query_patch = cut_around_center(img, center, p, margin=0)
    # Cut a larger area around the center and  split to patches
    refs = cut_around_center(data, center, p, margin=search_margin)
    refs = F.unfold(refs, kernel_size=p, stride=1)  # shape (b, c*p*p, N_patches)
    n_patches = refs.shape[-1]
    c = img.shape[1]
    refs = refs.permute(0, 2, 1).reshape(-1, c, p, p)

    # Search in RGB values
    print(f"Searching for NN patch in {len(refs)} patches at center {center}:")

    dists = compute_dists(refs, query_patch, dist)
    patch_index = torch.sort(torch.norm(dists, dim=1, p=1))[1][0]

    img_index = patch_index // n_patches
    return img_index


def find_patch_nns(fake_images, data, patch_size, search_margin, outputs_dir, n_centers=10, dist="rgb"):
    """
    Search for nearest patch in data to patches from generated images.
    Search is performed in a constrained locality of the query patch location
    @parm: search_margin: how many big will the search area be (in pixels)
    """
    with torch.no_grad():
        s = 3
        out_dir = f'{outputs_dir}/patch_nns(p-{patch_size}_s-{search_margin})'
        os.makedirs(out_dir, exist_ok=True)

        centers = sample_patch_centers(data.shape[-1], patch_size, n_centers)

        for j in range(len(fake_images)):
            query_image = fake_images[j]
            fig, ax = plt.subplots(nrows=len(centers), ncols=4, figsize=(s * 3, s * len(centers)))
            for i, center in enumerate(tqdm(centers)):
                ref_nn_index = search_for_nn_patches_in_locality(query_image.unsqueeze(0),
                                                                  data, center,
                                                                  p=patch_size,
                                                                  search_margin=search_margin,
                                                                 dist=dist)
                q_patch = cut_around_center(query_image, center, patch_size)
                r_patch = cut_around_center(data[ref_nn_index], center, patch_size)
                imshow(q_patch, ax[i, 0], "Query-Patch")
                imshow(r_patch, ax[i, 1], f"{dist}-NN-Patch: {(q_patch - r_patch).pow(2).sum().sqrt():.3f}")
                imshow(query_image, ax[i, 2], "Query-Image")
                imshow(data[ref_nn_index], ax[i, 3], f"{dist}-NN-Image")


            plt.tight_layout()
            fig.savefig(f'{out_dir}/patches-{j}.png')
            plt.clf()


def find_nns_percept(fake_images, data, outputs_dir, device):
    with torch.no_grad():
        vgg_fe = vgg_dist_calculator(layer_idx=9, device=device)
        # percept = lpips.LPIPS(net='vgg', lpips=False).to(device)

        os.makedirs(f'{outputs_dir}/nns', exist_ok=True)
        results = []
        for i in range(8):
            fake_image = fake_images[i]
            # dists = [percept(fake_image, data[i].unsqueeze(0)).sum().item() for i in range(len(data))]
            dists = [(vgg_fe.extract(fake_image) - vgg_fe.extract(data[i].unsqueeze(0))).pow(2).sum().item() for i in
                     range(len(data))]
            nn_indices = np.argsort(dists)
            nns = data[nn_indices[:4]]

            results.append(torch.cat([fake_image, nns]))

        vutils.save_image(torch.cat(results, dim=0).add(1).mul(0.5), f'{outputs_dir}/nns/im.png', normalize=False,
                          nrow=5)


def find_nns(fake_images, data, outputs_dir, show_first_n=2):
    with torch.no_grad():
        os.makedirs(f'{outputs_dir}/nns', exist_ok=True)
        results = []
        # dists_mat = (fake_image - data).pow(2).sum(dim=(1,2,3)).numpy()#
        for i in range(len(fake_images)):
            fake_image = fake_images[i]
            # dists = dists_mat[i]
            dists = [(fake_image - data[j]).pow(2).sum().item() for j in range(len(data))]
            nn_indices = np.argsort(dists)
            nns = data[nn_indices[:show_first_n]]
            results.append(torch.cat([fake_image.unsqueeze(0), nns]))

        vutils.save_image(torch.cat(results, dim=0).add(1).mul(0.5), f'{outputs_dir}/nns/im.png', normalize=False,
                          nrow=1 + show_first_n)
