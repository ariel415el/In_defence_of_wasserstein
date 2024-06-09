import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.ot_means import ot_means, weisfeld_minimization, sgd_minimization
from losses.batch_losses import MiniBatchPatchLoss, MiniBatchLoss, to_patches
from scripts.experiment_utils import get_data, get_centroids
from utils.common import dump_images


def compute_localized_kmeans(data):
    patches = to_patches(data, p=p, s=s, remove_locations=False)

    locs, n, cpp = patches.shape

    patch_localized_centroids = []
    for i in range(locs):
        print("Loc ", i)
        x = get_centroids(patches[i], n_centroids, use_faiss=False)
        # x = ot_means(patches[i], n_centroids, n_iters=4, minimization_method=weisfeld_minimization, verbose=False).reshape(-1, *patches[i].shape[1:])
        patch_localized_centroids += [x]
    patch_localized_centroids = torch.stack(patch_localized_centroids)

    patch_centroids_image = torch.nn.functional.fold(patch_localized_centroids.permute(1, 2, 0), (im_size, im_size),
                                                     kernel_size=p, stride=s)
    dump_images(patch_centroids_image, os.path.join(output_dir,  f"local_patch_centroids-{n_centroids}-{p}-{s}.png"))


def compute_patch_kmeans(data, ):
    patches = to_patches(data, p=p, s=s, remove_locations=True)
    locs = (im_size//s)**2

    # init_centroids = ot_means(data, n_centroids , n_iters=1, minimization_method=weisfeld_minimization, verbose=False).reshape(-1, *data.shape[1:])
    # init_centroids = to_patches(init_centroids, p=p, s=s, remove_locations=True)
    patch_centroids = get_centroids(patches, n_centroids * locs, use_faiss=True)
    # patch_centroids = ot_means(patches, n_centroids * locs, init_from=None,
    #                           n_iters=4, minimization_method=weisfeld_minimization, verbose=False
    #                           ).reshape(-1, *patches.shape[1:])

    # patch_centroids = patch_centroids.reshape(locs, n_centroids, -1).permute(1, 2, 0)
    patch_centroids = patch_centroids.reshape(n_centroids, locs, -1).permute(0, 2, 1)

    patch_centroids_image = torch.nn.functional.fold(patch_centroids, (im_size, im_size),
                                                     kernel_size=p, stride=s)
    dump_images(patch_centroids_image, os.path.join(output_dir, f"patch_centroids-{n_centroids}-{p}-{s}.png"))


def kmeans(X, k, init_from=None, max_iters=10, tol=1e-4):
    n_samples, n_features = X.shape

    # Randomly initialize centroids
    if init_from is None:
        centroids = X[np.random.choice(n_samples, k, replace=False)]
    else:
        centroids = init_from

    for _ in tqdm(range(max_iters)):
        # Assign each point to the nearest centroid
        from utils.metrics import compute_nearest_neighbors_in_batches
        from utils.metrics import L2
        labels = compute_nearest_neighbors_in_batches(X, centroids, L2(), bx=1024, by=1024)
        # distances = L2(b=1024)(X, centroids)
        # labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = torch.stack([X[labels == i].mean(0) for i in range(k)])

        # Check for convergence
        if torch.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids


def cover_data_patches(data):
    patches = to_patches(data, p=p, s=s, remove_locations=True)

    # init_centroids = to_patches(init_centroids, p=p, s=s, remove_locations=True)
    # init_centroids = ot_means(data, n_centroids, n_iters=4, minimization_method=sgd_minimization, verbose=False).reshape(-1, *data.shape[1:])
    # init_centroids = to_patches(init_centroids, p=p, s=s, remove_locations=True)
    # init_centroids = init_centroids[np.random.choice(len(init_centroids), n_centroids, replace=True)]
    # patch_centroids = get_centroids(patches, n_centroids, use_faiss=faiss)
    patch_centroids = ot_means(patches, n_centroids, init_from=None,
                              n_iters=15, minimization_method=sgd_minimization, verbose=False)
    #                           )
    # patch_centroids = kmeans(patches, n_centroids, init_from=patch_centroids)


    # patch_centroids = patch_centroids.reshape(locs, n_centroids, -1).permute(1, 2, 0)
    patch_centroids = patch_centroids.T.unsqueeze(0)
    new_im_size = int(np.floor(np.sqrt(n_centroids))+1) * p
    canvas = torch.nn.functional.unfold(torch.zeros((1,c,new_im_size, new_im_size)), kernel_size=p, stride=p)
    canvas[..., :n_centroids] = patch_centroids
    patch_centroids_image = torch.nn.functional.fold(canvas, (new_im_size, new_im_size),
                                                     kernel_size=p, stride=p)
    dump_images(patch_centroids_image, os.path.join(output_dir, f"cover_centroids-{n_centroids}-{p}-{s}.png"))


if __name__ == '__main__':

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "Patch-Kmeans")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    im_size = 64
    p = 8
    s = 8
    n_centroids = 64

    data_path = '../../data/FFHQ/FFHQ'
    # data_path = '../../data/MNIST/MNIST/jpgs/training'
    # data_path = '/mnt/storage_ssd/datasets/square_data/black_S-10_O-1_S-1'
    c = 1
    gray_scale = True
    center_crop = None
    limit_data = 1000
    data = get_data(data_path, im_size, c=c, center_crop=center_crop,
                    gray_scale=gray_scale, flatten=False, limit_data=limit_data).to(device)

    # centroids = get_centroids(data, n_centroids, use_faiss=True)
    # dump_images(centroids, os.path.join(output_dir,"centroids.png"))

    # cover_data_patches(data)
    compute_patch_kmeans(data)
    compute_localized_kmeans(data)