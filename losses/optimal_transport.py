import numpy as np
import ot
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dists import get_dist_metric, batch_NN


def get_ot_plan(C, epsilon=0):
    """Use POT to compute optimal transport between two emprical (uniforms) distriutaion with distance matrix C"""
    uniform_x = np.ones(C.shape[0]) / C.shape[0]
    uniform_y = np.ones(C.shape[1]) / C.shape[1]
    if epsilon > 0:
        OTplan = ot.sinkhorn(uniform_x, uniform_y, C, reg=epsilon)
    else:
        OTplan = ot.emd(uniform_x, uniform_y, C)
    return OTplan


def to_patches(x, p=8, s=4, sample_patches=None):
    """extract flattened patches from a pytorch image"""
    patches = F.unfold(x, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)
    patches = patches.reshape(-1, x.shape[1], p, p)
    if sample_patches is not None:
        patches = patches[torch.randperm(len(patches))[:int(sample_patches)]]
    return patches


def w1(x, y, epsilon=0, **kwargs):
    base_metric = get_dist_metric("L2")
    C = base_metric(x.reshape(len(x), -1), y.reshape(len(y), -1))
    OTPlan = get_ot_plan(C.detach().cpu().numpy(), int(epsilon))
    OTPlan = torch.from_numpy(OTPlan).to(C.device)
    W1 = torch.sum(OTPlan * C)
    return W1, {"W1-L2": W1}


def nn(x, y, alpha=None, **kwargs):
    base_metric = get_dist_metric("L2")
    C = base_metric(x.reshape(len(x), -1), y.reshape(len(y), -1))
    if alpha is not None:
        C = C / (C.min(dim=0)[0] + float(alpha))  # compute_normalized_scores
    nn_loss = C.min(dim=1)[0].mean()
    return nn_loss, {"nn_loss": nn_loss}

def remd(x, y, **kwargs):
    base_metric = get_dist_metric("L2")
    C = base_metric(x.reshape(len(x), -1), y.reshape(len(y), -1))
    nn_loss = max(C.min(dim=0)[0].mean(), C.min(dim=1)[0].mean())
    return nn_loss, {"remd_loss": nn_loss}



def duplicate_to_match_lengths(arr1, arr2):
    """
    Duplicates randomly selected entries from the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if arr1.shape[1] == arr2.shape[1]:
        return arr1, arr2
    elif arr1.shape[1] < arr2.shape[1]:
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[1] // arr2.shape[1]
    arr2 = torch.cat([arr2] * b, dim=1)
    if arr1.shape[1] > arr2.shape[1]:
        indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
        arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

    return arr1, arr2

def swd(x, y, num_proj=1024, **kwargs):
    num_proj = int(num_proj)
    b, c, h, w = x.shape

    # Sample random normalized projections
    rand = torch.randn(num_proj, c * h * w).to(x.device)  # (slice_size**2*ch)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions

    # Project images
    projx = torch.mm(x.reshape(-1, c * h * w), rand.T).T
    projy = torch.mm(y.reshape(-1, c * h * w), rand.T).T

    projx, projy = duplicate_to_match_lengths(projx, projy)

    # Sort and compute L1 loss
    projx, _ = torch.sort(projx, dim=1)
    projy, _ = torch.sort(projy, dim=1)

    n = projx.shape[1]
    # SWD = (projx - projy).pow(2).sum(1).sqrt().mean() / n
    SWD = (projx - projy).abs().mean()

    return SWD, {"SWD": SWD}


def sinkhorn(x, y, epsilon=1, **kwargs):
    from geomloss import SamplesLoss
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=1, blur=int(epsilon))
    SH = sinkhorn_loss(x.reshape(len(x), -1), y.reshape(len(y), -1))
    return SH, {"Sinkhorm-eps=1": SH}


def discrete_dual(x, y, n_steps=500, batch_size=None, lr=0.001, verbose=False, nnb=256, dist="L2"):
    pbar = range(n_steps)
    if verbose:
        print(f"Optimizing duals: {x.shape}, {y.shape}")
        pbar = tqdm(pbar)

    if batch_size is None:
        batch_size = len(x)

    loss_func = get_dist_metric(dist)
    psi = torch.zeros(len(x), requires_grad=True, device=x.device)
    opt_psi = torch.optim.Adam([psi], lr=lr)
    # scheduler = ReduceLROnPlateau(opt_psi, 'min', threshold=0.0001, patience=200)
    for _ in pbar:
        opt_psi.zero_grad()

        mini_batch = y[torch.randperm(len(y))[:batch_size]]

        phi, outputs_idx = batch_NN(mini_batch, x, psi, nnb, loss_func)

        dual_estimate = torch.mean(phi) + torch.mean(psi)

        loss = -1 * dual_estimate  # maximize over psi
        loss.backward()
        opt_psi.step()
        # scheduler.step(dual_estimate)
        if verbose:
            pbar.set_description(f"dual estimate: {dual_estimate.item()}, LR: {opt_psi.param_groups[0]['lr']}")

    return dual_estimate.item()

class MiniBatchLoss:
    def __init__(self, dist='w1', **kwargs):
        self.metric =globals()[dist]
        self.kwargs = kwargs

    def compute(self, x, y):
        return self.metric(x, y, **self.kwargs)

    def __call__(self, images_X, images_Y):
        with torch.no_grad():
            return self.compute(images_X, images_Y)[0]

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("MiniBatchLosses should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        return self.compute(real_data, fake_data)


class MiniBatchPatchLoss(MiniBatchLoss):
    def __init__(self, dist='w1', p=5, s=1, n_samples=None, **kwargs):
        super(MiniBatchPatchLoss, self).__init__(dist,  **kwargs)
        self.p = int(p)
        self.s = int(s)
        self.n_samples = n_samples

    def compute(self, x, y):
        return self.metric(to_patches(x, self.p, self.s, self.n_samples),
                           to_patches(y, self.p, self.s, self.n_samples), **self.kwargs)


