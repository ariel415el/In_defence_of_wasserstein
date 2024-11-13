import numpy as np
import torch
from torchvision import transforms, models


def get_metric(name, **kwargs):
    """Choose how to calulate pairwise distances for EMD"""
    if name == 'L1':
        metric = L1()
    elif name == 'L2':
        metric = L2()
    elif name == 'mean':
        metric = MeanL2()
    elif name == 'edge':
        metric = EdgeL2()
    elif name == 'vgg':
         metric = VggDistCalculator(**kwargs)
    else:
        raise ValueError(f"No such metric name {name}")
    return metric


class L1:
    def __call__(self, X, Y):
        X = X.reshape(len(X), -1)
        Y = Y.reshape(len(Y), -1)
        return torch.abs(X[:, None] - Y[None, :]).sum(-1)


class L2:
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e sqrt((X[:, None] - Y[None, :])**2)
    :param normalize: divide distance by the dimension
    """
    def __call__(self, X, Y, normalize=False):
        X = X.reshape(len(X), -1).double()
        Y = Y.reshape(len(Y), -1).double()
        dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
        # dist = (X[:, None] - Y[None, :]).pow(2).sum(-1).sqrt()
        dist = torch.sqrt(torch.clamp(dist, min=1e-10)) # When loss is 0 the gradient of sqrt is nan
        if normalize:
            dist = dist / X.shape[-1]
        return dist.float()


class MeanL2(L2):
    def __call__(self, X, Y):
        return super().__call__(X.mean(1), Y.mean(1))


class EdgeL2(L2):
    def __init__(self):
        super().__init__()
        self.filter = torch.Tensor([[1, 0, -1],
                                      [2, 0, -2],
                                      [1, 0, -1]]).view((1, 1, 3, 3))

    def get_edge_map(self , X):
        self.filter = self.filter.to(X.device)
        return torch.nn.functional.conv2d(torch.mean(X, dim=1, keepdim=True), self.filter)

    def __call__(self, X, Y):
        return super().__call__(self.get_edge_map(X), self.get_edge_map(Y))


class VggDistCalculator:
    def __init__(self,  layer_idx=18, device=None):
        self.layer_idx = layer_idx  # [4, 9, 18]
        self.vgg_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.device = device
        self.vgg_features.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.vgg_features = self.vgg_features.to(self.device)

    def extract(self, X, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.layer_idx
        if self.device is None:
            self.device = X.device
            self.vgg_features = self.vgg_features.to(self.device)
        X = X.to(self.device)
        for i, layer in enumerate(self.vgg_features):
            X = layer(X)
            if i == layer_idx:
                return X

    def batch_extract(self, x, b=16):
        return torch.cat([self.extract(x[slice]).cpu() for slice in get_batche_slices(len(x), b)], dim=0)

    def __call__(self, X, Y):
        features_x = self.extract(X).reshape(len(X), -1)
        features_y = self.extract(Y).reshape(len(Y), -1)
        return L2()(features_x, features_y)


class DiscriminatorDistCalculator:
    def __init__(self,  netD, layer_idx=None, device=None):
        assert hasattr(netD, 'features'), "netD has no attribute 'features'"
        self.netD = netD
        self.netD.eval()
        self.device = device
        self.layer_idx = layer_idx
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def extract(self, X):
        if self.device is None:
            self.device = X.device
            self.netD = self.netD.to(self.device)
        X = X.to(self.device)
        if self.layer_idx is None:
            return self.netD.features(X)
        else:
            for i, layer in enumerate(self.netD.features.children()):
                X = layer(X)
                # if i in layer_indices:
                if i == self.layer_idx:
                    return X
            return X

    def __call__(self, X, Y):
        features_x = self.extract(X).reshape(len(X), -1)
        features_y = self.extract(Y).reshape(len(Y), -1)
        return L2()(features_x, features_y)


def get_batche_slices(n, b):
    n_batches = n // b
    slices = []
    for i in range(n_batches):
        slices.append(np.arange(i * b, (i + 1) * b))

    if n % b != 0:
        slices.append(np.arange(n_batches * b, n))
    return slices


def compute_nearest_neighbors_in_batches(X, Y, nn_function, bx=64, by=64):
    """
    Compute nearest-neighbor index from Y for each X but restrict maximum inference batch to 'b'
    :param nn_function: a function that computes the nearest neighbor
    """
    X = X.cpu()
    Y = Y.cpu()
    x_slices = get_batche_slices(len(X), bx)
    y_slices = get_batche_slices(len(Y), by)
    NNs = torch.zeros(len(X), dtype=torch.long, device=X.device)
    for x_slice in x_slices:
        dists = torch.zeros(len(x_slice), len(Y), device=X.device, dtype=X.dtype)
        for y_slice in y_slices:
            dists[:, y_slice] = nn_function(X[x_slice], Y[y_slice]).cpu()
        NNs[x_slice] = dists.argmin(1).long()

    return NNs


def compute_pairwise_distances_in_batches(X, Y, dist_function, bx=64, by=64):
    """
    Compute pairwise distances between X and Y in batches
    :param dist_function: a function that computes parwise distances
    """
    X = X.cpu()
    Y = Y.cpu()
    x_slices = get_batche_slices(len(X), bx)
    y_slices = get_batche_slices(len(Y), by)
    dists = torch.zeros(len(X), len(Y), device=X.device)
    for x_slice in x_slices:
        for y_slice in y_slices:
            dists[x_slice[0]:x_slice[-1]+1, y_slice[0]:y_slice[-1]+1] = dist_function(X[x_slice], Y[y_slice])
    return dists


if __name__ == '__main__':
    x = torch.ones((1,3,8,7))
    print(VggDistCalculator(layer_idx=18).extract(x).shape)