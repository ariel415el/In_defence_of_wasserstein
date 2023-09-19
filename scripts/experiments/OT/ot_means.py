import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import ot
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiment_utils import get_data
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from losses import get_loss_function
from utils.common import dump_images


def get_ot_plan(C):
    """Use POT to compute optimal transport between two emprical (uniforms) distriutaion with distance matrix C"""
    uniform_x = np.ones(C.shape[0]) / C.shape[0]
    uniform_y = np.ones(C.shape[1]) / C.shape[1]
    OTplan = ot.emd(uniform_x, uniform_y, C).astype(np.float32)
    return OTplan


def dist_mat(X, Y):
    return ((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T)**0.5


def compute_means(ot_map, data, k):
    return (ot_map[:,:, None] * data[None, ] * k).sum(1)

def average_minimization(centroids, data, k):
    C = dist_mat(centroids, data)
    ot_map = get_ot_plan(C)
    centroids = compute_means(ot_map, data, k)
    return centroids


def weisfeld_step(X, dist_mat, W):
    """See Weisfeld algorithm formula: https://ssabach.net.technion.ac.il/files/2015/12/BS2015.pdf equation (4)"""
    nominator = (W / dist_mat) @ X         ## np.allclose(nominator[0],((1/C)[0][:, None] * data).sum(0))
    denominator = (W / dist_mat).sum(1)[:, None]
    new_centroids = nominator / denominator
    return new_centroids


def weisfeld_minimization(centroids, data, n_steps):
    """at each iteration compute OT and peroform Weisfeld steps to approximate the minimum of the weighted sum of
     distances according OT map weights"""
    for j in range(n_steps):
        C = dist_mat(centroids, data)
        ot_map = get_ot_plan(C)
        centroids = weisfeld_step(data, C, ot_map)
    return centroids


def sgd_minimization(centroids, data):
    centroids.requires_grad_()
    opt = torch.optim.Adam([centroids], lr=0.01)
    C = dist_mat(centroids, data)
    ot_map = get_ot_plan(C.cpu().detach().numpy())
    loss = torch.sum(torch.from_numpy(ot_map).cuda() * C)
    opt.zero_grad()
    loss.backward()
    opt.step()
    centroids.dtach()
    return centroids


def ot_mean(data, k, n_iters, minimization_method):

    metrics = [
        'MiniBatchLoss-dist=w1',
        'MiniBatchLoss-dist=swd',
        'MiniBatchPatchLoss-dist=swd-p=8-s=4',
    ]
    metrics = {metric: get_loss_function(metric) for metric in metrics}
    _, c,h,w = data.shape
    data = data.reshape(len(data), -1).numpy()
    centroids = np.random.randn(k, data.shape[-1]).astype(np.float32) * 0.5
    plots = defaultdict(list)
    for i in range(n_iters):
        centroids = minimization_method(centroids, data)

        for metric_name, metric in metrics.items():
            dist = metric(torch.from_numpy(centroids).reshape(-1, c, h, w),
                          torch.from_numpy(data).reshape(-1, c, h, w)).item()
            plots[metric_name].append(dist)
            print(f"{metric_name}: {dist:.4f}\n")

        dump_images(torch.from_numpy(centroids).reshape(args.k, -1, args.im_size, args.im_size),
                    f"{out_dir}/images/otMeans-{i}.png")
    return plots


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


    # Other
    parser.add_argument('--project_name', default="OTMeans", type=str)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument("--tag", default="testt", type=str)
    args = parser.parse_args()

    out_dir = f"outputs/{args.project_name}/{os.path.basename(args.data_path)}_I-{args.im_size}_K-{args.k}_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    data = get_data(args.data_path, args.im_size, c=1 if args.gray_scale else 3,
                    limit_data=args.limit_data, center_crop=args.center_crop, flatten=False)

    minimiztion_func = lambda x, y: weisfeld_minimization(x,y, n_steps=5)
    plots = ot_mean(data, args.k, 10, minimiztion_func)

    COLORS=['r', 'g', 'b' ,'y', 'k']
    for i, (metric_name, plot) in enumerate(plots.items()):
        plt.plot(np.arange(len(plot)), plot, label=metric_name, color=COLORS[i])
        plt.annotate(f"{plot[-1]:.2f}", (len(plot) -1, plot[-1]), textcoords="offset points",
                     xytext=(-2, 2), ha="center")

    plt.legend()
    plt.savefig(f"{out_dir}/Losses.png")
    plt.clf()
