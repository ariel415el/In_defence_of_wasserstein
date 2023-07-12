import argparse
import os

import numpy as np
import ot
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from scripts.EMD.utils import get_data, dump_images


def get_ot_plan(C):
    """Use POT to compute optimal transport between two emprical (uniforms) distriutaion with distance matrix C"""
    uniform_x = np.ones(C.shape[0]) / C.shape[0]
    uniform_y = np.ones(C.shape[1]) / C.shape[1]
    OTplan = ot.emd(uniform_x, uniform_y, C)
    return OTplan

def dist_mat(X, Y):
    return ((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T)**0.5

def ot_means(data, k, n_iters):
    data = data.reshape(len(data), -1).numpy()
    centroids = np.random.randn(k, data.shape[-1]) * 0.5
    losses = []
    pbar = tqdm(range(n_iters))
    for i in pbar:
        C =  dist_mat(centroids, data)
        ot_map = get_ot_plan(C)
        centroids = (ot_map[:,:, None] * data[None, ] * k).sum(1)
        loss = np.sum(ot_map * C)
        losses.append(loss)
        pbar.set_description(f"Loss: {loss}")
        dump_images(torch.from_numpy(centroids), 64, 64, 3, f"outputs/OTMeans-{i}.png")
    return losses



def pixel_ot(data, k, n_iters):
    data = data.reshape(len(data), -1).cuda()
    centroids = torch.randn(k, data.shape[-1]).cuda() * 0.5
    centroids.requires_grad_()
    opt = torch.optim.Adam([centroids], lr=0.01)

    losses = []
    pbar = tqdm(range(n_iters))
    for i in pbar:
        C =  dist_mat(centroids, data)
        ot_map = get_ot_plan(C.cpu().detach().numpy())
        loss = torch.sum(torch.from_numpy(ot_map).cuda() * C)
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        pbar.set_description(f"Loss: {loss}")
        if i % 100:
            dump_images(centroids, 64, 64, 3, f"outputs/pixelOT-{i}.png")
    return losses


if __name__ == "__main__":
    os.makedirs("outputs",exist_ok=True)
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ/FFHQ_1000/FFHQ_1000",
                        help="Path to train images")

    # Model
    parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--im_size', default=64, type=int)


    # Other
    parser.add_argument('--n_workers', default=4, type=int)

    args = parser.parse_args()

    data = get_data(args.data_path, args.im_size, 3)

    losses_pixel_ot = pixel_ot(data, args.k, 250)
    losses_ot_means = ot_means(data, args.k, 10)

    plt.plot(np.arange(len(losses_pixel_ot)), losses_pixel_ot, label="PixelOT", color="r")
    plt.plot(np.arange(len(losses_ot_means)), losses_ot_means, label="OTMeans", color="b")
    plt.legend()
    plt.savefig("Losses.png")
    plt.clf()

