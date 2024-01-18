import os
import sys
from collections import defaultdict

import torch
from matplotlib import pyplot as plt

from scripts.experiments.OT.ot_means import ot_mean, weisfeld_minimization

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from losses.batch_losses import MiniBatchPatchLoss, MiniBatchLoss, to_patches
from scripts.experiments.experiment_utils import get_data, get_centroids
from utils.common import dump_images


def compute_localized_kmeans(data):
    patches = to_patches(data, p=p, s=s, remove_locations=False)

    locs, n, cpp = patches.shape

    patch_localized_centroids = []
    for i in range(locs):
        # x = get_centroids(patches[i], n_centroids, use_faiss=faiss)
        x = ot_mean(patches[i], n_centroids, n_iters=4, minimization_method=weisfeld_minimization, verbose=False).reshape(-1, *patches[i].shape[1:])
        patch_localized_centroids += [x]
    patch_localized_centroids = torch.stack(patch_localized_centroids)

    patch_centroids_image = torch.nn.functional.fold(patch_localized_centroids.permute(1, 2, 0), (im_size, im_size),
                                                     kernel_size=p, stride=s)
    dump_images(patch_centroids_image, os.path.join(output_dir,  f"local_patch_centroids-{n_centroids}-{p}-{s}.png"))


def compute_patch_kmeans(data):
    patches = to_patches(data, p=p, s=s, remove_locations=True)

    locs = (im_size//s)**2

    init_centroids = ot_mean(data, n_centroids , n_iters=1, minimization_method=weisfeld_minimization, verbose=False).reshape(-1, *data.shape[1:])
    init_centroids = to_patches(init_centroids, p=p, s=s, remove_locations=True)
    # patch_centroids = get_centroids(patches, n_centroids * locs, use_faiss=faiss)
    patch_centroids = ot_mean(patches, n_centroids * locs, init_from=init_centroids,
                              n_iters=1, minimization_method=weisfeld_minimization, verbose=False
                              ).reshape(-1, *patches.shape[1:])

    # patch_centroids = patch_centroids.reshape(locs, n_centroids, -1).permute(1, 2, 0)
    patch_centroids = patch_centroids.reshape(n_centroids, locs, -1).permute(0, 2, 1)

    patch_centroids_image = torch.nn.functional.fold(patch_centroids, (im_size, im_size),
                                                     kernel_size=p, stride=s)
    dump_images(patch_centroids_image, os.path.join(output_dir, f"patch_centroids-{n_centroids}-{p}-{s}.png"))



if __name__ == '__main__':

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "Patch-Kmeans")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    faiss=True
    im_size = 64
    p = s = 16
    n_centroids = 64

    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ'
    c = 3
    gray_scale = False
    center_crop = 80
    limit_data = 10000
    data = get_data(data_path, im_size, c=c, center_crop=center_crop,
                    gray_scale=gray_scale, flatten=False, limit_data=limit_data).to(device)

    # centroids = get_centroids(data, n_centroids, use_faiss=faiss)
    # dump_images(centroids, os.path.join(output_dir,"centroids.png"))

    compute_patch_kmeans(data)
    # compute_localized_kmeans(data)
