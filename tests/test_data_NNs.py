import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from torchvision import utils as vutils
import torch.nn.functional as F

from utils.metrics import VggDistCalculator, DiscriminatorDistCalculator, L2, compute_nearest_neighbors_in_batches, \
    get_batche_slices, get_metric
from tests.test_utils import cut_around_center, sample_patch_centers


def imshow(img, axs, title="img"):
    cmap = 'gray' if img.shape[0] == 1 else None
    axs.imshow((img.permute(1, 2, 0).numpy() + 1) / 2, cmap=cmap)
    axs.axis('off')
    axs.set_title(title)


def search_for_nn_patches_in_locality_in_batches(img, data, center, p, stride, search_margin, metric, device, b):
    """
    for a given patch location and size, search the nearest patch from data (only in the same patch location) to
    the patch in img in that location"""
    n = len(data)
    max_n_patches_in_crop = F.unfold(torch.ones(1,3,p+2*search_margin, p+2*search_margin), kernel_size=p, stride=stride).shape[-1]
    query_patch = cut_around_center(img, center, p, margin=0)
    # Cut a larger area around the center and  split to patches
    dists = torch.ones((n, max_n_patches_in_crop)) * np.inf
    slices = get_batche_slices(n, b)
    for slice in slices:
        refs = cut_around_center(data[slice], center, p, margin=search_margin)
        refs = F.unfold(refs, kernel_size=p, stride=stride)  # shape (b, c*p*p, n_patches_in_crop)
        n_patches_in_crop = refs.shape[-1]
        c = img.shape[1]
        refs = refs.permute(0, 2, 1).reshape(-1, c, p, p)

        dists[slice, :n_patches_in_crop] = metric(refs.to(device), query_patch.to(device))[:, 0].reshape(len(slice), -1).cpu()

    patch_index = torch.argmin(dists.reshape(-1)).item()
    img_index = patch_index // max_n_patches_in_crop
    patch_index = patch_index % max_n_patches_in_crop

    return img_index, patch_index


def crop_specific_patch(image, center, p, stride, search_margin, patch_index):
    crop = cut_around_center(image, center, p, margin=search_margin)
    patches = F.unfold(crop, kernel_size=p, stride=stride)  # shape (b, c*p*p, n_patches_in_crop)
    patch = patches[..., patch_index]
    return patch.reshape(image.shape[0], p, p)


def find_patch_nns(fake_images, data, patch_size, stride, search_margin, outputs_dir, n_centers=10, b=64, metric_name='L2', device=torch.device('cpu')):
    """
    Search for nearest patch in data to patches from generated images.
    Search is performed in a constrained locality of the query patch location
    @parm: search_margin: how many big will the search area be (in pixels)
    @parm: stride: search patches inside the search area with this stride
    """
    with torch.no_grad():
        s = 3
        out_dir = f'{outputs_dir}/{metric_name}-patch_nns(p-{patch_size}_s-{search_margin})'
        os.makedirs(out_dir, exist_ok=True)

        centers = sample_patch_centers(data.shape[-1], patch_size, n_centers)

        metric = get_metric(metric_name)

        for j in range(len(fake_images)):
            query_image = fake_images[j]
            fig, ax = plt.subplots(nrows=len(centers), ncols=4, figsize=(s * 3, s * len(centers)))
            for i, center in enumerate(tqdm(centers)):
                ref_nn_index, patch_index = search_for_nn_patches_in_locality_in_batches(query_image.unsqueeze(0),
                                                                  data, center,
                                                                  p=patch_size,
                                                                  stride=stride,
                                                                  search_margin=search_margin,
                                                                  b=b,
                                                                  metric=metric,
                                                                  device=device)

                q_patch = cut_around_center(query_image, center, patch_size)

                r_patch = crop_specific_patch(data[ref_nn_index], center, patch_size, stride, search_margin, patch_index)
                imshow(q_patch, ax[i, 0], "Query-Patch")
                imshow(r_patch, ax[i, 1], f"{metric_name}-NN-Patch: {(q_patch - r_patch).pow(2).sum().sqrt():.3f}")
                imshow(query_image, ax[i, 2], "Query-Image")
                imshow(data[ref_nn_index], ax[i, 3], f"{metric_name}-NN-Image")

            plt.tight_layout()
            fig.savefig(f'{out_dir}/patches-{j}.png')
            plt.clf()


def find_nns_percept(fake_images, data, outputs_dir, device, netD=None, layer_idx=None):
    with torch.no_grad():
        if netD is None:
            feature_loss = VggDistCalculator(layer_idx, device=device)
        else:
            feature_loss = DiscriminatorDistCalculator(netD, layer_idx, device=device)
        out_path = f'{outputs_dir}/nns/{"vgg" if netD is None else "Discriminator"}_im.png'

        nn_indices = compute_nearest_neighbors_in_batches(fake_images, data, feature_loss, bx=64, by=64)
        debug_img = torch.cat([fake_images, data[nn_indices]], dim=-2)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        vutils.save_image(debug_img, out_path, normalize=True, nrow=len(fake_images), pad_value=1)


def find_nns(fake_images, data, outputs_dir):
    with torch.no_grad():
        nn_indices = compute_nearest_neighbors_in_batches(fake_images, data, L2(), bx=64, by=64)
        debug_img = torch.cat([fake_images, data[nn_indices]], dim=-2)

        os.makedirs(f'{outputs_dir}/nns', exist_ok=True)
        vutils.save_image(debug_img, f'{outputs_dir}/nns/im.png', normalize=True, nrow=len(fake_images), pad_value=1)
