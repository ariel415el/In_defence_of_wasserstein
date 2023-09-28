import argparse
import os
import sys

import numpy as np
import ot
import torch

from experiment_utils import get_data

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.common import dump_images


def get_ot_plan(C):
    """Use POT to compute optimal transport between two emprical (uniforms) distriutaion with distance matrix C"""
    uniform_x = np.ones(C.shape[0]) / C.shape[0]
    uniform_y = np.ones(C.shape[1]) / C.shape[1]
    OTplan = ot.emd(uniform_x, uniform_y, C).astype(np.float32)
    return OTplan


def dist_mat(X, Y):
    return ((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T)**0.5


def weisfeld_step(X, dist_mat, W):
    """See Weisfeld algorithm formula: https://ssabach.net.technion.ac.il/files/2015/12/BS2015.pdf equation (4)"""
    dist_mat = torch.clip_(dist_mat, min=0.000001) # Avoid dividing by zero
    nominator = (W / dist_mat) @ X         ## np.allclose(nominator[0],((1/C)[0][:, None] * data).sum(0))
    denominator = (W / dist_mat).sum(1)[:, None]
    denominator = torch.clip_(denominator, min=0.000001) # Avoid dividing by zero
    new_centroids = nominator / denominator
    return new_centroids


def weisfeld_minimization(centroids, data, n_steps=5):
    """at each iteration compute OT and peroform Weisfeld steps to approximate the minimum of the weighted sum of
     distances according OT map weights"""
    for j in range(n_steps):
        C = dist_mat(centroids, data)
        ot_map = get_ot_plan(C.cpu().detach().numpy())
        centroids = weisfeld_step(data, C, ot_map)
    return centroids


def ot_means(data, k, n_iters, init_from=None, debug_dir=None):
    print(f"Running OTmeans with k={k} on data of shape {data.shape}")
    data_shape = data.shape
    data = data.reshape(len(data), -1)

    if init_from is None:
        centroids = torch.randn((k, data.shape[-1])) * 0.5
    else:
        centroids = init_from

    for i in range(n_iters):
        centroids = weisfeld_minimization(centroids, data)

        if debug_dir is not None:
            dump_images(centroids.reshape(-1, *data.shape[1:]), f"{debug_dir}/images/otMeans-{i}.png")

    return centroids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ/FFHQ/FFHQ",
                        help="Path to train images")
    parser.add_argument('--center_crop', default=None, type=int)
    parser.add_argument('--limit_data', default=None, type=int)
    parser.add_argument('--gray_scale', action='store_true', default=False)

    # Model
    parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--n_iters', default=10, type=int)

    # Other
    parser.add_argument('--project_name', default="OTMeans", type=str)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument("--train_name", default=None, type=str)
    args = parser.parse_args()

    if args.train_name is None:
        args.train_name = f"{os.path.basename(args.data_path)}_I-{args.im_size}_K-{args.k}"
    out_dir = f"outputs/{args.project_name}/{args.train_name}"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    data = get_data(args.data_path, args.im_size, c=1 if args.gray_scale else 3,
                    limit_data=args.limit_data, center_crop=args.center_crop, flatten=False)


    centroids = ot_means(data, args.k, args.n_iters, out_dir)
