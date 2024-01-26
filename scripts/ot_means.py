import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import ot
import torch
from matplotlib import pyplot as plt

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


def sgd_minimization(centroids, data, n_steps=10):
    centroids.requires_grad_()
    opt = torch.optim.Adam([centroids], lr=0.1)
    for i in range(n_steps):
        C = dist_mat(centroids, data)
        ot_map = get_ot_plan(C.cpu().detach().numpy())
        loss = torch.sum(torch.from_numpy(ot_map) * C)
        opt.zero_grad()
        loss.backward()
        opt.step()
    centroids.detach().cpu().numpy()
    return centroids


def ot_means(data, k, n_iters, minimization_method, init_from=None, verbose=False, out_dir=None):
    print(f"Running OTmeans with k={k} on data of shape {data.shape}")
    if verbose:
        plots = defaultdict(list)
        metrics = [
            'MiniBatchLoss-dist=w1',
            'MiniBatchLoss-dist=swd',
            # 'MiniBatchPatchLoss-dist=swd-p=8-s=4',
        ]
        metrics = {metric: get_loss_function(metric) for metric in metrics}
    data_shape = data.shape
    data = data.reshape(len(data), -1)
    if init_from is None:
        centroids = torch.randn((k, data.shape[-1])) * 0.5
    else:
        centroids = init_from
    for i in range(n_iters):
        centroids = minimization_method(centroids, data)

        if verbose:
            print(f"Iter: {i}")
            for metric_name, metric in metrics.items():
                dist = metric(centroids.reshape(-1, *data_shape[1:]),
                              data.reshape(-1, *data_shape[1:])).item()
                print(f"\b {metric_name}: {dist:.4f}")
                plots[metric_name].append(dist)

            if out_dir is not None:
                dump_images(centroids.reshape(k, -1, data_shape[-2], data_shape[-1]),
                            f"{out_dir}/images/otMeans-{i}.png")
    if verbose:
        return plots
    else:
        return centroids


def log(plots):
    COLORS=['r', 'g', 'b' ,'y', 'k']
    for i, (metric_name, plot) in enumerate(plots.items()):
        plt.plot(np.arange(len(plot)), plot, label=metric_name, color=COLORS[i])
        plt.annotate(f"{plot[-1]:.2f}", (len(plot) -1, plot[-1]), textcoords="offset points",
                     xytext=(-2, 2), ha="center")
        pickle.dump(plot, open(f'{out_dir}/plots/{metric_name}_fixed_noise_gen_to_train.pkl', 'wb'))

    plt.legend()
    plt.savefig(f"{out_dir}/Losses.png")
    plt.clf()


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
    parser.add_argument('--min_method', default="weisfeld", type=str, help="[weisfeld, sgd, mean]")


    # Other
    parser.add_argument('--n_iters', default=10, type=int)
    parser.add_argument('--project_name', default="OTMeans", type=str)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument("--train_name", default=None, type=str)
    args = parser.parse_args()

    if args.train_name is None:
        args.train_name = f"{os.path.basename(args.data_path)}_M-{args.min_method}_I-{args.im_size}_K-{args.k}"
    out_dir = f"outputs/{args.project_name}/{args.train_name}"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    data = get_data(args.data_path, args.im_size, c=1 if args.gray_scale else 3,
                    limit_data=args.limit_data, center_crop=args.center_crop, flatten=False)

    minimiztion_func = globals()[f"{args.min_method}_minimization"]

    plots = ot_means(data, args.k, args.n_iters, minimiztion_func, verbose=True, out_dir=out_dir)

    log(plots)
