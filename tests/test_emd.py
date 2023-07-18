import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import utils as vutils

def efficient_l2(X, Y):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    return (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))

def compute_distances_batch(X, Y, b):
    """
    Computes distance matrix in batches of rows to reduce memory consumption from (n1 * n2 * d) to (d * n2 * d)
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    :param b: rows batch size
    Returns a (n2, n1) matrix of L2 distances
    """
    """"""
    b = min(b, len(X))
    dist_mat = torch.zeros((X.shape[0], Y.shape[0]), dtype=torch.float16, device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        dist_mat[i * b:(i + 1) * b] = efficient_l2(X[i * b:(i + 1) * b], Y)
    if len(X) % b != 0:
        dist_mat[n_batches * b:] = efficient_l2(X[n_batches * b:], Y)

    return dist_mat
def test_emd(G, z_dim, data, outputs_dir, device):
    """Generate len(data) images and compute EMD (by finding NNs and computing distances)"""
    with torch.no_grad():
        n = len(data)
        fake_imgs = G(torch.randn((n, z_dim)).to(device))

        dists = compute_distances_batch(fake_imgs.reshape(n, -1), data.reshape(n, -1), b=128)
        nn_indices = torch.argsort(dists, dim=1)[:, 0]

        nns = data[nn_indices]
        print(f"Distance: {torch.sqrt((fake_imgs - nns).pow(2).sum(dim=(1,2,3))).mean().item():.3f}")

        ncols = int(np.sqrt(n))
        vutils.save_image(fake_imgs, f'{outputs_dir}/samples.png', normalize=True, nrow=ncols)
        vutils.save_image(nns, f'{outputs_dir}/data.png', normalize=True, nrow=ncols)

        n_samples = 5
        s = 3
        fig, axs = plt.subplots(n_samples, 4, figsize=(4*s, n_samples*s))

        perm = torch.from_numpy(np.load('/mnt/storage_ssd/datasets/FFHQ/FFHQ64_1000_shuffled/perm.npy'))
        inverser_perm = np.argsort(perm)
        fake_imgs_inv = fake_imgs.reshape(n, data.shape[1], -1)[:, :, inverser_perm].reshape(data.shape)
        nns_inv = nns.reshape(n, data.shape[1], -1)[:, :, inverser_perm].reshape(data.shape)

        for i in range(n_samples):
            axs[i, 0].imshow(((fake_imgs[i] +1) / 2).permute(1,2,0).numpy())
            axs[i, 0].axis('off')
            axs[i, 0].set_title("Fake")
            axs[i, 1].imshow(((nns[i] +1) / 2).permute(1,2,0).numpy())
            axs[i, 1].axis('off')
            axs[i, 1].set_title(f"Data NN \n(L2: {torch.sqrt((fake_imgs[i] - nns[i]).pow(2).sum()).item():.3f})")

            if perm is not None:
                axs[i, 2].imshow(((fake_imgs_inv[i] + 1) / 2).permute(1, 2, 0).numpy())
                axs[i, 2].axis('off')
                axs[i, 2].set_title("Fake before erm")
                axs[i, 3].imshow(((nns_inv[i] + 1) / 2).permute(1, 2, 0).numpy())
                axs[i, 3].axis('off')
                axs[i, 3].set_title(f"NN before perm")


        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, "nns.png"))