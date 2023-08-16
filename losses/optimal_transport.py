import numpy as np
import ot
import torch
import torch.nn.functional as F

from losses.loss_utils import get_dist_metric


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
    return W1, {"W1-L1": W1}

def swd(x, y, num_proj=512, **kwargs):
    num_proj = int(num_proj)
    b, c, h, w = x.shape

    # Sample random normalized projections
    rand = torch.randn(num_proj, c * h * w).to(x.device)  # (slice_size**2*ch)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions

    # Project images
    projx = torch.mm(x.reshape(b, c * h * w), rand.T)
    projy = torch.mm(y.reshape(b, c * h * w), rand.T)

    # Sort and compute L1 loss
    projx, _ = torch.sort(projx, dim=1)
    projy, _ = torch.sort(projy, dim=1)

    SWD = torch.abs(projx - projy).mean()
    return SWD, {"SWD": SWD}

def sinkhorn(x, y, epsilon=1, **kwargs):
    from geomloss import SamplesLoss
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=1, blur=int(epsilon))
    SH = sinkhorn_loss(x.reshape(len(x), -1), y.reshape(len(y), -1))
    return SH, {"Sinkhorm-eps=1": SH}

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


