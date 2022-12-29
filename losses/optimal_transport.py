import numpy as np
import ot
import torch
import torch.nn.functional as F
from losses.loss_utils import vgg_dist_calculator, L1_metric, L2_metric


def get_ot_plan(C):
    uniform_x = np.ones(C.shape[0]) / C.shape[0]
    uniform_y = np.ones(C.shape[1]) / C.shape[1]
    OTplan = ot.emd(uniform_x, uniform_y, C)
    return OTplan

def to_patches(x, p=8, s=4):
    patches = F.unfold(x, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)
    patches = patches.reshape(-1, patches.shape[-1])
    return patches

def get_dist_metric(name):
    if name == 'L1':
        metric = L1_metric()
    elif name == 'L2':
        metric = L2_metric()
    elif name == 'vgg':
         metric = vgg_dist_calculator()
    else:
        raise ValueError(f"No such metric name {name}")
    return metric

class BatchW2D:
    def __init__(self, dist='L1'):
        self.metric = get_dist_metric(dist)

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("BatchW2D should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        C = self.metric(real_data, fake_data)

        OTPlan = get_ot_plan(C.detach().cpu().numpy())
        OTPlan = torch.from_numpy(OTPlan).to(C.device)

        OT = torch.sum(OTPlan * C)

        # perm = OTPlan.argmax(0)

        return OT, {"OT": OT.item()}


class BatchPatchW2D:
    def __init__(self, dist='L1', n_batches=1, n_samples=128, p=8, s=1):
        self.metric = get_dist_metric(dist)
        self.n_batches = int(n_batches)
        self.n_samples = int(n_samples)
        self.p = int(p)
        self.s = int(s)

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("BatchW2D should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        real_patches = to_patches(real_data, p=self.p, s=self.s)
        fake_patches = to_patches(fake_data, p=self.p, s=self.s)

        OT = 0
        for i in range(self.n_batches):
            reals = real_patches[torch.randperm(len(real_patches))[:self.n_samples]]
            fakes = fake_patches[torch.randperm(len(fake_patches))[:self.n_samples]]

            C = self.metric(reals, fakes)

            OTPlan = get_ot_plan(C.detach().cpu().numpy())
            OTPlan = torch.from_numpy(OTPlan).to(C.device)

            OT += torch.sum(OTPlan * C)

        OT /= self.n_batches

        return OT, {"OT": OT.item()}


class BatchSWD:
    def __init__(self, dist='L1'):
        self.num_proj = 128

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("BatchW2D should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        b, c, h, w = real_data.shape

        # Sample random normalized projections
        rand = torch.randn(self.num_proj, c*h*w).to(real_data.device) # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True) # noramlize to unit directions

        # Project patches

        projx = torch.mm(real_data.reshape(b, c*h*w), rand.T)
        projy = torch.mm(fake_data.reshape(b, c*h*w), rand.T)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        Gloss = torch.abs(projx - projy).mean()

        return Gloss, {"OT": Gloss.item()}


class BatchPatchSWD:
    def __init__(self, n_proj=64, patch_size=8):
        self.num_proj = int(n_proj)
        self.patch_size = int(patch_size)

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("BatchW2D should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        b, c, h, w = real_data.shape

        # Sample random normalized projections
        rand = torch.randn(self.num_proj, c*self.patch_size**2).to(real_data.device) # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
        rand = rand.reshape(self.num_proj, c, self.patch_size, self.patch_size)

        # Project patches
        projx = F.conv2d(real_data, rand).transpose(1,0).reshape(self.num_proj, -1)
        projy = F.conv2d(fake_data, rand).transpose(1,0).reshape(self.num_proj, -1)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        Gloss = torch.abs(projx - projy).mean()

        return Gloss, {"OT": Gloss.item()}
