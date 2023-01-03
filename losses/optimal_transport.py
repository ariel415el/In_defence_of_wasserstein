import numpy as np
import ot
import torch
import torch.nn.functional as F
from losses.loss_utils import get_dist_metric


def get_ot_plan(C):
    """Use POT to compute optimal transport between two emprical (uniforms) distriutaion with distance matrix C"""
    uniform_x = np.ones(C.shape[0]) / C.shape[0]
    uniform_y = np.ones(C.shape[1]) / C.shape[1]
    OTplan = ot.emd(uniform_x, uniform_y, C)
    return OTplan

def to_patches(x, p=8, s=4):
    """extract flattened patches from a pytorch image"""
    patches = F.unfold(x, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)
    patches = patches.reshape(-1, patches.shape[-1])
    return patches


class BatchEMD:
    def __init__(self, dist='L1'):
        self.metric = get_dist_metric(dist)

    def compute(self, images_X, images_Y):
        C = self.metric(images_X, images_Y)

        OTPlan = get_ot_plan(C.detach().cpu().numpy())
        OTPlan = torch.from_numpy(OTPlan).to(C.device)

        OT = torch.sum(OTPlan * C)

        return OT

    def __call__(self, images_X, images_Y):
        with torch.no_grad():
            return self.compute(images_X, images_Y)

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("BatchEMD should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        OT = self.__call__(real_data, fake_data)
        return OT, {"OT": OT.item()}


class BatchPatchEMD:
    """Split images to patches and sample n_samples from each to compute on."""
    def __init__(self, dist='L1', n_samples=128, p=8, s=1):
        self.metric = get_dist_metric(dist)
        self.n_samples = int(n_samples if n_samples != "all" else -1)
        self.p = int(p)
        self.s = int(s)

    def compute(self, images_X, images_Y):
        X_patches = to_patches(images_X, p=self.p, s=self.s)
        Y_patches = to_patches(images_Y, p=self.p, s=self.s)

        X_patches = X_patches[torch.randperm(len(X_patches))[:self.n_samples]]
        Y_patches = Y_patches[torch.randperm(len(Y_patches))[:self.n_samples]]

        C = self.metric(X_patches, Y_patches)

        OTPlan = get_ot_plan(C.detach().cpu().numpy())
        OTPlan = torch.from_numpy(OTPlan).to(C.device)

        OT = torch.sum(OTPlan * C)

        return OT

    def __call__(self, images_X, images_Y):
        with torch.no_grad():
            return self.compute(images_X, images_Y)

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("BatchPatchEMD should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        OT = self.__call__(real_data, fake_data)
        return OT, {"OT": OT.item()}


class BatchSWD:
    """Project images on random directions and solve 1D OT"""
    def __init__(self, num_proj=128):
        self.num_proj = int(num_proj)

    def compute(self, images_X, images_Y):
        b, c, h, w = images_X.shape

        # Sample random normalized projections
        rand = torch.randn(self.num_proj, c*h*w).to(images_X.device) # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True) # noramlize to unit directions

        # Project images
        projx = torch.mm(images_X.reshape(b, c*h*w), rand.T)
        projy = torch.mm(images_Y.reshape(b, c*h*w), rand.T)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        return torch.abs(projx - projy).mean()

    def __call__(self, images_X, images_Y):
        with torch.no_grad():
            return self.compute(images_X, images_Y)

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("BatchSWD should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        SWD = self.__call__(real_data, fake_data)
        return SWD, {"SWD": SWD.item()}



class BatchPatchSWD:
    """Project patches into 1D with random convolutions and compute 1D OT"""
    def __init__(self, n_proj=128, p=8, s=1):
        self.num_proj = int(n_proj)
        self.p = int(p)
        self.s = int(s)

    def compute(self, images_X, images_Y):
        b, c, h, w = images_X.shape

        # Sample random normalized projections
        rand = torch.randn(self.num_proj, c*self.p**2).to(images_X.device) # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
        rand = rand.reshape(self.num_proj, c, self.p, self.p)

        # Project patches
        projx = F.conv2d(images_X, rand, stride=self.s).transpose(1,0).reshape(self.num_proj, -1)
        projy = F.conv2d(images_Y, rand, stride=self.s).transpose(1,0).reshape(self.num_proj, -1)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        return torch.abs(projx - projy).mean()

    def __call__(self, images_X, images_Y):
        with torch.no_grad():
            return self.compute(images_X, images_Y)

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("BatchPatchSWD should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        PatchSWD = self.__call__(real_data, fake_data)
        return PatchSWD, {"PatchSWD": PatchSWD.item()}
