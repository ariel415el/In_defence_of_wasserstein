import os

import numpy as np
import torch
from torchvision import utils as vutils

from utils.metrics import vgg_dist_calculator, L2


def find_mode_collapses(G, D, z_dim, outputs_dir, device, n=128):
    """Generate some images and sort them by the Discriminator score and their distance to their nearest neighbor.
    crappy images tend to go together and sorting by distance to NN will reveal this. The sorting by D score shows
    different order meaning it didn't catch the problem.
    """
    with torch.no_grad():
        os.makedirs(f"{outputs_dir}/mode_collapses", exist_ok=True)
        fixed_noise = torch.randn((n, z_dim), device=device)
        g_images = G(fixed_noise)

        calc = vgg_dist_calculator(layer_idx=9)
        dists_mat = calc(g_images, g_images)
        # calc = L2()
        # dists_mat = calc(g_images.reshape(n, -1), g_images.reshape(n, -1))

        NN_indices = torch.argsort(dists_mat, dim=1)[:, 1]

        vutils.save_image(torch.cat([g_images, g_images[NN_indices]]),
                          f'{outputs_dir}/mode_collapses/NNs.jpg',
                          nrow=n, normalize=True)

        NN_dists = dists_mat[torch.arange(n), NN_indices]
        perm = torch.argsort(NN_dists)

        vutils.save_image(g_images[perm],
                          f'{outputs_dir}/mode_collapses/sorted_by_NN_distance.jpg',
                          nrow=int(np.sqrt(n)), normalize=True)

        # scores = D(g_images)
        #
        # perm = torch.argsort(scores)
        #
        # vutils.save_image(g_images[perm],
        #                   f'{outputs_dir}/mode_collapses/sorted_by_D_Score.jpg',
        #                   nrow=int(np.sqrt(n)), normalize=True)
