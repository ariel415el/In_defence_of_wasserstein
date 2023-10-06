import itertools
import os
from random import shuffle

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib.patches as patches
from tqdm import tqdm

from torchvision import utils as vutils
import torch.nn.functional as F

from utils.metrics import vgg_dist_calculator
from tests.test_utils import cut_around_center, compute_dists, sample_patch_centers


def imshow(img, axs, title="img"):
    cmap = 'gray' if img.shape[0] == 1 else None
    axs.imshow((img.permute(1, 2, 0).numpy() + 1) / 2, cmap=cmap)
    axs.axis('off')
    axs.set_title(title)


def search_for_nn_patches_in_locality(img, data, center, p, stride, search_margin, dist="edge"):
    """
    for a given patch location and size, search the nearest patch from data (only in the same patch location) to
    the patch in img in that location"""
    query_patch = cut_around_center(img, center, p, margin=0)
    # Cut a larger area around the center and  split to patches
    refs = cut_around_center(data, center, p, margin=search_margin)
    refs = F.unfold(refs, kernel_size=p, stride=stride)  # shape (b, c*p*p, N_patches)
    n_patches = refs.shape[-1]
    c = img.shape[1]
    refs = refs.permute(0, 2, 1).reshape(-1, c, p, p)

    # Search in RGB values
    print(f"Searching for NN patch in {len(refs)} patches at center {center}:")

    dists = compute_dists(refs, query_patch, dist)
    patch_index = torch.sort(torch.norm(dists, dim=1, p=1))[1][0]

    img_index = patch_index // n_patches
    return img_index, refs[patch_index]


def find_patch_nns(fake_images, data, patch_size, stride, search_margin, outputs_dir, n_centers=10, dist="rgb"):
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
                ref_nn_index, r_patch = search_for_nn_patches_in_locality(query_image.unsqueeze(0),
                                                                  data, center,
                                                                  p=patch_size,
                                                                  stride=stride,
                                                                  search_margin=search_margin,
                                                                 dist=dist)
                q_patch = cut_around_center(query_image, center, patch_size)
                # r_patch = cut_around_center(data[ref_nn_index], center, patch_size)
                # r_center = find_patch_location_in_image(r_patch, r_image)
                imshow(q_patch, ax[i, 0], "Query-Patch")
                imshow(r_patch, ax[i, 1], f"{dist}-NN-Patch: {(q_patch - r_patch).pow(2).sum().sqrt():.3f}")
                imshow(query_image, ax[i, 2], "Query-Image")
                imshow(data[ref_nn_index], ax[i, 3], f"{dist}-NN-Image")


            plt.tight_layout()
            fig.savefig(f'{out_dir}/patches-{j}.png')
            plt.clf()


def find_nns(fake_images, data, outputs_dir, device, show_first_n=2, perceptual=True):
    from lpips import lpips
    percept = lpips.LPIPS(net='vgg').to(device)

    with torch.no_grad():
        os.makedirs(f'{outputs_dir}/nns', exist_ok=True)
        results = []
        # dists_mat = (fake_image - data).pow(2).sum(dim=(1,2,3)).numpy()#
        for i in range(len(fake_images)):
            fake_image = fake_images[i]
            # dists = dists_mat[i]
            if perceptual:
                dists = [percept(fake_image, data[j]).item() for j in range(len(data))]
            else:
                dists = [F.mse_loss(fake_image, data[j]).item() for j in range(len(data))]
            nn_indices = np.argsort(dists)
            nns = data[nn_indices[:show_first_n]]
            results.append(torch.cat([fake_image.unsqueeze(0), nns]))

        vutils.save_image(torch.cat(results, dim=0).add(1).mul(0.5), f'{outputs_dir}/nns/im.png', normalize=False,
                          nrow=1 + show_first_n, pad_value=1)



def crop_image(img, xx, yy, d):
    return img[..., yy:yy+d, xx:xx+d]

query_idx = 0
center = (128,128)
dynamic_pathc_size=16
def interactive_nn_debug(fake_images, data, patch_size, stride, search_margin, dist="rgb"):
    fig, axs = plt.subplots(1, 4, figsize=(9,3))
    fake_images = fake_images.detach()
    global dynamic_pathc_size
    dynamic_pathc_size = patch_size
    def on_click(event):
        global center
        global query_idx
        global dynamic_pathc_size
        if event is not None:
            if event.button is MouseButton.LEFT and axs[0].in_axes(event):
                center = int(event.xdata), int(event.ydata)
                print(f"Next center", center)
            if event.button is MouseButton.RIGHT and axs[0].in_axes(event):
                query_idx = (query_idx + 1) % len(fake_images)
                print(f"Next image", query_idx)
            if event.button is MouseButton.LEFT and axs[1].in_axes(event):
                dynamic_pathc_size = min(dynamic_pathc_size + 2, min(fake_images.shape[-2:]))
                print(f"Patch size", dynamic_pathc_size)
            if event.button is MouseButton.RIGHT and axs[1].in_axes(event):
                dynamic_pathc_size = max(dynamic_pathc_size - 2, 4)
                print(f"Patch size", dynamic_pathc_size)

        query_image = fake_images[query_idx]
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        imshow(fake_images[query_idx], axs[0], "Query-image\nRight click: next image\n Left click: choose center")
        rect = patches.Rectangle((center[0] - dynamic_pathc_size//2, center[1] - dynamic_pathc_size//2), dynamic_pathc_size, dynamic_pathc_size, linewidth=1, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)

        ref_nn_index, r_patch = search_for_nn_patches_in_locality(query_image.unsqueeze(0),
                                                         data, center,
                                                         p=dynamic_pathc_size,
                                                         stride=stride,
                                                         search_margin=search_margin,
                                                         dist=dist)

        q_patch = cut_around_center(query_image, center, dynamic_pathc_size)
        # r_patch = cut_around_center(data[ref_nn_index], center, dynamic_pathc_size)

        imshow(q_patch, axs[1], f"Query-Patch ({dynamic_pathc_size}x{dynamic_pathc_size}\nRight click: bigger patch\n Left click: smaller patch")
        imshow(r_patch, axs[2], f"{dist}-NN-Patch: {(q_patch - r_patch).pow(2).sum().sqrt():.3f}")
        imshow(data[ref_nn_index], axs[3], f"{dist}-NN-Image")
        plt.tight_layout()
        plt.show()

    plt.connect('button_press_event', on_click)
    on_click(None)
    plt.show()