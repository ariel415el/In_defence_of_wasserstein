import torch
import torch.nn.functional as F

from utils import distribution_metrics

def to_patches(x, p=8, s=4, sample_patches=None):
    """extract flattened patches from a pytorch image"""
    patches = F.unfold(x, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)
    patches = patches.reshape(-1, x.shape[1], p, p)
    if sample_patches is not None:
        patches = patches[torch.randperm(len(patches))[:int(sample_patches)]]
    return patches


class MiniBatchLoss:
    def __init__(self, dist='w1', **kwargs):
        self.metric = getattr(distribution_metrics, dist)
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


