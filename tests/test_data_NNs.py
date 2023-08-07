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


def search_for_nn_patches_in_locality(img, data, center, p, s, search_margin):
    """
    for a given patch location and size, search the nearest patch from data (only in the same patch location) to
    the patch in img in that location"""
    h = p // 2
    d = img.shape[-1]
    c = img.shape[1]
    query_patch = img[..., center[0]-h:center[0]+h, center[1]-h:center[1]+h]
    # Cut a larger area around the center and  split to patches
    refs = data[..., max(0 ,center[0]-h-search_margin):min(d, center[0]+h+search_margin), max(0, center[1]-h-search_margin):min(d, center[1]+h+search_margin)]
    refs = F.unfold(refs, kernel_size=p, stride=s) # shape (b, 3*p*p, N_patches)
    refs = refs.permute(0, 2, 1).reshape(-1, c, p, p)

    # Search in RGB values
    print(f"Searching for NN patch in {len(refs)} patches:")
    dists = (refs - query_patch).reshape(refs.shape[0], -1)
    rgb_nn_index = torch.sort(torch.norm(dists, dim=1, p=1))[1][0]

    # Search in gray level values
    dists = (refs.mean(1) - query_patch.mean(1)).reshape(refs.shape[0], -1) # Compare gray scale images
    gs_nn_index = torch.sort(torch.norm(dists, dim=1, p=1))[1][0]

    # Search in edge values
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).view((1,1,3,3))
    dists = (F.conv2d(torch.mean(refs, dim=1, keepdim=True), a) - F.conv2d(torch.mean(query_patch, dim=1, keepdim=True), a)).reshape(refs.shape[0], -1)
    edge_nn_index = torch.sort(torch.norm(dists, dim=1, p=1))[1][0]

    return query_patch, refs[rgb_nn_index].clone(), refs[gs_nn_index].clone(), refs[edge_nn_index].clone()


def find_patch_nns(fake_images, data, patch_size, stride, search_margin, outputs_dir, n_centers=10):
    """
    Search for nearest patch in data to patches from generated images.
    Search is performed in a constrained locality of the query patch location
    @parm: search_margin: how many big will the search area be (in pixels)
    """
    with torch.no_grad():
        out_dir = f'{outputs_dir}/patch_nns(p-{patch_size}_s-{search_margin})'
        os.makedirs(out_dir, exist_ok=True)
        h = patch_size // 2
        img_dim = data.shape[-1]
        centers = np.arange(h, img_dim - h + 1, stride)
        centers = list(itertools.product(centers, repeat=2))[:n_centers]
        shuffle(centers)

        for j in range(len(fake_images)):
            query_image = fake_images[j].unsqueeze(0)

            s = 3
            fig, ax = plt.subplots(nrows=len(centers), ncols=3, figsize=(s * 3, s * len(centers)))
            for i, center in enumerate(tqdm(centers)):
                query_patch, rgb_nn_patch, gs_nn_patch, edge_nn_patch = search_for_nn_patches_in_locality(query_image, data, center, p=patch_size, s=1, search_margin=search_margin)

                cmap = 'gray' if query_patch.shape[1] == 1 else None
                axs = ax[0] if len(centers) == 1 else ax[i, 0]
                axs.imshow((query_patch[0].permute(1, 2,0).numpy() + 1)/2, cmap=cmap)
                axs.axis('off')
                axs.set_title('Query patch')

                axs = ax[1] if len(centers) == 1 else ax[i, 1]
                axs.imshow((rgb_nn_patch.permute(1,2,0).numpy() + 1)/2, cmap=cmap)
                axs.axis('off')
                axs.set_title('RGB NN')

                axs = ax[2] if len(centers) == 1 else ax[i, 2]
                axs.imshow((rgb_nn_patch - query_patch).abs()[0].permute(1,2,0).numpy(), cmap=cmap)
                axs.axis('off')
                axs.set_title('diff NN')

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
            dists = [(vgg_fe.extract(fake_image) - vgg_fe.extract(data[i].unsqueeze(0))).pow(2).sum().item() for i in range(len(data))]
            nn_indices = np.argsort(dists)
            nns = data[nn_indices[:4]]

            results.append(torch.cat([fake_image, nns]))

        vutils.save_image(torch.cat(results, dim=0).add(1).mul(0.5), f'{outputs_dir}/nns/im.png', normalize=False, nrow=5)


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

        vutils.save_image(torch.cat(results, dim=0).add(1).mul(0.5), f'{outputs_dir}/nns/im.png', normalize=False, nrow=1+show_first_n)


