import json

import torch
import torch.nn.functional as F

from utils import distribution_metrics
from utils.metrics import get_metric


class MiniBatchLoss:
    def __init__(self, dist='w1', **kwargs):
        self.metric = getattr(distribution_metrics, dist)
        self.kwargs = kwargs

    def compute(self, x, y):
        return self.metric(x.reshape(len(x), -1),
                           y.reshape(len(y), -1),
                           **self.kwargs)

    def __call__(self, images_X, images_Y):
        with torch.no_grad():
            return self.compute(images_X, images_Y)[0]

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("MiniBatchLosses should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        return self.compute(real_data, fake_data)


def to_patches(x, p=8, s=4, sample_patches=None, remove_locations=True):
    """extract flattened patches from a pytorch image"""
    b, c, _, _ = x.shape
    patches = F.unfold(x, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    if remove_locations:
        patches = patches.permute(0, 2, 1).reshape(-1, c*p**2)
    else:
        patches = patches.permute(2, 0, 1).reshape(-1, b, c*p**2)
    if sample_patches is not None:
        patches = patches[torch.randperm(len(patches))[:int(sample_patches)]]
    return patches


class MiniBatchPatchLoss(MiniBatchLoss):
    def __init__(self, dist='w1', p=5, s=1, **kwargs):
        super(MiniBatchPatchLoss, self).__init__(dist,  **kwargs)
        self.p = int(p)
        self.s = int(s)

    def compute(self, x, y):
        x_patches = to_patches(x, self.p, self.s)
        y_patches = to_patches(y, self.p, self.s)
        return self.metric(x_patches, y_patches, **self.kwargs)


class MiniBatchLocalPatchLoss(MiniBatchLoss):
    def __init__(self, dist='w1', p=5, s=1, **kwargs):
        super(MiniBatchLocalPatchLoss, self).__init__(dist,  **kwargs)
        self.dist_name = dist
        self.p = int(p)
        self.s = int(s)

    def compute(self, x, y):
        x_patches = to_patches(x, self.p, self.s, remove_locations=False)
        y_patches = to_patches(y, self.p, self.s, remove_locations=False)
        n_locs, _, _ = x_patches.shape
        loss = torch.stack([self.metric(x_patches[l], y_patches[l], **self.kwargs)[0] for l in range(n_locs)]).mean()
        return loss, {f"Local-{self.dist_name}": loss.item()}


class MiniBatchNeuralLoss(MiniBatchLoss):
    def __init__(self, dist='w1', b=256, layer_idx=18, device=torch.device('cpu'), **kwargs):
        super(MiniBatchNeuralLoss, self).__init__(dist, **kwargs)
        self.vgg = get_metric("vgg", layer_idx=int(layer_idx), device=device)
        self.b = int(b)

    def compute(self, x, y):
        x = self.vgg.batch_extract(x, self.b).mean(-1).mean(-1)
        y = self.vgg.batch_extract(y, self.b).mean(-1).mean(-1)
        return self.metric(x, y, **self.kwargs)


class MiniBatchNeuralPatchLoss(MiniBatchLoss):
    def __init__(self, dist='w1', b=256, layer_idx=18, device=torch.device('cpu'), **kwargs):
        super(MiniBatchNeuralPatchLoss, self).__init__(dist, **kwargs)
        self.vgg = get_metric("vgg", layer_idx=int(layer_idx), device=device)
        self.b = int(b)

    def compute(self, x, y):
        x = self.vgg.batch_extract(x, self.b)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).permute(0, 2, 1).reshape(-1, c)

        y = self.vgg.batch_extract(y, self.b)
        b, c, h, w = y.shape
        y = y.reshape(b, c, h*w).permute(0, 2, 1).reshape(-1, c)

        return self.metric(x, y, **self.kwargs)
