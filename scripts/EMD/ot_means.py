import argparse
import os
import sys

import numpy as np
import ot
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from utils import get_data, dump_images


def get_ot_plan(C):
    """Use POT to compute optimal transport between two emprical (uniforms) distriutaion with distance matrix C"""
    uniform_x = np.ones(C.shape[0]) / C.shape[0]
    uniform_y = np.ones(C.shape[1]) / C.shape[1]
    OTplan = ot.emd(uniform_x, uniform_y, C)
    return OTplan

def dist_mat(X, Y):
    return ((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T)**0.5

def compute_means(ot_map, data, k):
    return (ot_map[:,:, None] * data[None, ] * k).sum(1)

def ot_means(data, k, n_iters):
    data = data.reshape(len(data), -1).numpy()
    centroids = np.random.randn(k, data.shape[-1]) * 0.5
    losses = []
    pbar = tqdm(range(n_iters))
    for i in pbar:
        C =  dist_mat(centroids, data)
        ot_map = get_ot_plan(C)
        centroids = compute_means(ot_map, data, k)

        C =  dist_mat(centroids, data)
        loss = np.sum(ot_map * C)
        losses.append(loss)
        pbar.set_description(f"Loss: {loss}")
        dump_images(torch.from_numpy(centroids), 64, 64, 3, f"{out_dir}/OTMeans-{i}.png")
    return losses

def weisfeld_step(X, dist, W):
    nominator = (W / dist) @ X         ## np.allclose(nominator[0],((1/C)[0][:, None] * data).sum(0))
    denominator = (W / dist).sum(1)[:, None]
    new_centroids = nominator / denominator
    return new_centroids

def ot_means_weisfeld(data, k, n_iters, n_weisfeld):
    data = data.reshape(len(data), -1).numpy()
    centroids = np.random.randn(k, data.shape[-1]).astype(np.float64) * 0.5
    losses = []
    pbar = tqdm(range(n_iters))
    for i in pbar:
        for j in range(n_weisfeld):
            C = dist_mat(centroids, data)
            ot_map = get_ot_plan(C)
            centroids = weisfeld_step(data, C, ot_map)
        C = dist_mat(centroids, data)
        ot_map = get_ot_plan(C)
        loss = np.sum(ot_map * C)
        losses.append(loss)
        pbar.set_description(f"Loss: {loss}")
        dump_images(torch.from_numpy(centroids), k, 64, 3, f"{out_dir}/OTMeans_weisfeld-{i}.png")
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
            dump_images(centroids, k, 64, 3, f"{out_dir}/pixelOT-{i}.png")
    return losses

def block(in_feat, out_feat, normalize='in'):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize == "bn":
        layers.append(nn.BatchNorm1d(out_feat))
    elif normalize == "in":
        layers.append(nn.InstanceNorm1d(out_feat))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim=64, nf=128, depth=4, normalize='none', **kwargs):
        super(Generator, self).__init__()
        self.output_dim = output_dim
        nf = int(nf)
        depth = int(depth)

        layers = block(z_dim, nf, normalize=normalize)

        for i in range(depth - 1):
            layers += block(nf, nf, normalize=normalize)

        layers += [nn.Linear(nf, 3*output_dim**2), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        img = self.model(z)
        return img

def generator_ot_means(data, k, n_iters, n_steps):
    z_dim = 64
    data = data.reshape(len(data), -1).cuda()
    latents = torch.randn(k, z_dim).cuda()
    netG = Generator(z_dim, output_dim=64, nf=32, depth=1).cuda()
    opt = torch.optim.Adam(netG.parameters(), lr=0.01)

    losses = []
    pbar = tqdm(range(n_iters))
    for i in pbar:
        centroids = netG(latents)
        C =  dist_mat(centroids, data)
        ot_map = get_ot_plan(C.cpu().detach().numpy())
        for j  in range(n_steps):
            C = dist_mat(centroids, data)
            loss = torch.sum(torch.from_numpy(ot_map).cuda() * C)
            opt.zero_grad()
            loss.backward()
            opt.step()
            centroids = netG(latents)
            losses.append(loss.item())

            pbar.set_description(f"Loss: {loss}")
        if i % 10 == 0:
            dump_images(centroids, k, 64, 3, f"{out_dir}/GeneratorOTMeans-{i}.png")
    return losses

if __name__ == "__main__":
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

    out_dir = f"outputs/OTMeans/{os.path.basename(args.data_path)}_K-{args.k}"
    os.makedirs(out_dir, exist_ok=True)

    data = get_data(args.data_path, args.im_size, 3)

    # losses_pixel_ot = pixel_ot(data, args.k, 250)
    losses_generator_ot_means = generator_ot_means(data, args.k, 100, 100)
    losses_ot_weisfeld_means = ot_means_weisfeld(data, args.k, 10, 10)
    # losses_ot_means = ot_means(data, args.k, 10)

    # plt.plot(np.arange(len(losses_pixel_ot)), losses_pixel_ot, label="PixelOT", color="r")
    plt.plot(np.arange(len(losses_generator_ot_means)), losses_generator_ot_means, label="GenOTMeans", color="y")
    # plt.plot(np.arange(len(losses_ot_means)), losses_ot_means, label="OTMeans", color="b")
    plt.plot(np.arange(len(losses_ot_weisfeld_means)), losses_ot_weisfeld_means, label="OTMeansWeisfeld", color="g")
    plt.legend()
    plt.savefig(f"{out_dir}/Losses.png")
    plt.clf()

