import itertools
import os

import numpy as np
import torch
from tqdm import tqdm

from torchvision import utils as vutils
import torch.nn.functional as F

from losses.loss_utils import vgg_dist_calculator


def _batch_nns(img, data, center, p, s):
    """for a given patch location and size Search the nearest patch from data (only in the same patch location) to
    the patch in img in that location"""
    h = p // 2
    query_patch = img[..., center[0]-h:center[0]+h, center[1]-h:center[1]+h]
    refs = data[..., center[0]-h-s:center[0]+h+s, center[1]-h-s:center[1]+h+s]
    refs = F.unfold(refs, kernel_size=p, stride=1) # shape (b, 3*p*p, N_patches)
    refs = refs.permute(0, 2, 1).reshape(-1, 3, p, p)

    # Search in RGB values
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

    return refs[rgb_nn_index].clone(), refs[gs_nn_index].clone(), refs[edge_nn_index].clone()


def find_patch_nns(G, z_dim, data, patch_size, stride, search_margin, outputs_dir, device):
    """
    Search for nearest patch in data to patches from generated images.
    Search is performed in a constrained locality of the query patch location
    """
    with torch.no_grad():
        out_dir = f'{outputs_dir}/patch_nns(p-{patch_size}_s-{search_margin})'
        os.makedirs(out_dir, exist_ok=True)
        h = patch_size // 2
        img_dim = data.shape[-1]
        centers = np.arange(patch_size, img_dim - patch_size, stride)
        centers = list(itertools.product(centers, repeat=2))

        print(f"Searching for {patch_size}x{patch_size} patch nns in {len(data)} data samples:")
        for j in range(5):
            query_image = G(torch.randn(1, z_dim).to(device))
            vutils.save_image(query_image.add(1).mul(0.5), f'{out_dir}/query_img-{j}.png', normalize=False, nrow=2)

            rgb_NN_patches = []
            gs_NN_patches = []
            edge_NN_patches = []
            q_patches = []
            for i, center in enumerate(tqdm(centers)):
                q_patches.append(query_image[..., center[0]-h:center[0]+h, center[1]-h:center[1]+h])
                rgb_nn_patch, gs_nn_patch, edge_nn_patch = _batch_nns(query_image, data, center, p=patch_size, s=search_margin)
                rgb_NN_patches.append(rgb_nn_patch.unsqueeze(0))
                gs_NN_patches.append(gs_nn_patch.unsqueeze(0))
                edge_NN_patches.append(edge_nn_patch.unsqueeze(0))

            x = torch.cat(q_patches + rgb_NN_patches + gs_NN_patches + edge_NN_patches, dim=0)
            vutils.save_image(x.add(1).mul(0.5), f'{out_dir}/patches-{j}.png', normalize=False, nrow=len(q_patches))


def find_nns(G, z_dim, data, outputs_dir, device):
    with torch.no_grad():
        vgg_fe = vgg_dist_calculator(layer_idx=9, device=device)
        # percept = lpips.LPIPS(net='vgg', lpips=False).to(device)

        os.makedirs(f'{outputs_dir}/nns', exist_ok=True)
        results = []
        for i in range(8):
            fake_image = G(torch.randn((1, z_dim), device=device))
            # dists = [percept(fake_image, data[i].unsqueeze(0)).sum().item() for i in range(len(data))]
            dists = [vgg_fe.extract(fake_image) - vgg_fe.extract(data[i].unsqueeze(0)).pow(2).sum().item() for i in range(len(data))]
            nn_indices = np.argsort(dists)
            nns = data[nn_indices[:4]]

            results.append(torch.cat([fake_image, nns]))

        vutils.save_image(torch.cat(results, dim=0).add(1).mul(0.5), f'{outputs_dir}/nns/im.png', normalize=False, nrow=5)




