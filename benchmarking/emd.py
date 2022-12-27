import numpy as np
import ot
import torch
import torch.nn.functional as F


def emd(x, y, p=1):
    uniform_x = np.ones(len(x)) / len(x)
    uniform_y = np.ones(len(y)) / len(y)
    M = ot.dist(x, y, p=p) / x.shape[1]
    # from utils import compute_distances_batch
    # M = compute_distances_batch(x, y, b=1024)
    return ot.emd2(uniform_x, uniform_y, M)


def to_patches(x, p=8, s=4):
    patches = F.unfold(x, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)
    patches = patches.reshape(-1, patches.shape[-1])
    return patches


class EMD:
    def __init__(self, norm=1):
        self.norm = norm
    def __str__(self):
        return f"EMD-L{self.norm}"

    def __call__(self, x, y):
        return emd(x.reshape(x.shape[0], -1).detach().cpu().numpy(),
                   y.reshape(y.shape[0], -1).detach().cpu().numpy(),
                   p=self.norm)

class patchEMD:
    def __init__(self, p, n, norm=1):
        self.norm = norm
        self.p = p
        self.n = n

    def __str__(self):
        return f"PatchEMD_L{self.norm}_P-{self.p}_#-{self.n}"

    def __call__(self, x, y):
        x = to_patches(x, p=self.p, s=1).detach().cpu().numpy()
        y = to_patches(y, p=self.p, s=1).detach().cpu().numpy()
        assert (len(x) == len(y))
        patch_indices = np.random.choice(len(x), size=self.n, replace=False)
        x = x[patch_indices]
        y = y[patch_indices]

        return emd(x, y, p=self.norm)