import os

import numpy as np
import torch
from torchvision import utils as vutils

from losses.loss_utils import vgg_dist_calculator


def find_mode_collapses(G, D, z_dim, outputs_dir, device):
    with torch.no_grad():

        os.makedirs(f"{outputs_dir}/mode_collapses", exist_ok=True)
        b = 256
        fixed_noise = torch.randn((b, z_dim), device=device)
        g_images = G(fixed_noise)

        calc = vgg_dist_calculator(layer_idx=9)
        dists_mat = calc(g_images, g_images)

        second_NN_indices = torch.argsort(dists_mat, dim=1)[:, 1]

        vutils.save_image(torch.cat([g_images, g_images[second_NN_indices]]),
                          f'{outputs_dir}/mode_collapses/NNs.jpg',
                          nrow=b, normalize=True)

        second_NN_dists = dists_mat[torch.arange(b), second_NN_indices]
        perm = torch.argsort(second_NN_dists)

        vutils.save_image(g_images[perm],
                          f'{outputs_dir}/mode_collapses/sorted_by_NN_distance.jpg',
                          nrow=int(np.sqrt(b)), normalize=True)




        scores = D(g_images)

        perm = torch.argsort(scores)

        vutils.save_image(g_images[perm],
                          f'{outputs_dir}/mode_collapses/sorted_by_D_Score.jpg',
                          nrow=int(np.sqrt(b)), normalize=True)
