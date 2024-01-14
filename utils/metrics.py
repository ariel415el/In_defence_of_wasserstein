import torch
from torchvision import transforms, models


def get_metric(name):
    """Choose how to calulate pairwise distances for EMD"""
    if name == 'L1':
        metric = L1()
    elif name == 'L2':
        metric = L2()
    elif name == 'vgg':
         metric = VggDistCalculator()
    elif name == 'inception':
        metric = InceptionDistCalculator()
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
    """
    def __call__(self, X, Y):
        X = X.reshape(len(X), -1)
        Y = Y.reshape(len(Y), -1)
        dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
        dist = torch.sqrt(torch.clamp(dist, min=1e-10)) # When loss is 0 the gradient of sqrt is nan
        # dist = (X[:, None] - Y[None, :]).pow(2).sum(-1).sqrt()
        return dist


class InceptionDistCalculator:
    def __init__(self, device=None):
        from benchmarking.inception import myInceptionV3
        self.device = device
        self.inception = myInceptionV3()
        self.inception.eval()

    def extract(self, X):
        X = X.to(self.device)
        if self.device is None:
            self.device = torch.device("cpu")
            self.inception.to(self.device)
        return self.inception(X)

    def __call__(self, X, Y):
        features_x = self.extract(X).reshape(len(X), -1)
        features_y = self.extract(Y).reshape(len(Y), -1)
        return L2()(features_x, features_y)


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
        slices.append(range(i * b, (i + 1) * b))

    if n % b != 0:
        slices.append(range(n_batches * b, n))
    return slices


def compute_features_nearest_neighbors_batches(X, Y, loss_function, bx=64, by=64):
    """Compute distance matrix in features of a function f(X) but restrict maximum inference batch to 'b'"""
    X = X.cpu()
    Y = Y.cpu()
    x_slices = get_batche_slices(len(X), bx)
    y_slices = get_batche_slices(len(Y), by)
    NNs = torch.zeros(len(X), dtype=torch.long, device=X.device)
    for x_slice in x_slices:
        dists = torch.zeros(len(x_slice), len(Y), device=X.device, dtype=X.dtype)
        for y_slice in y_slices:
            dists[:, y_slice] = loss_function(X[x_slice], Y[y_slice])
        NNs[x_slice] = dists.argmin(1).long()

    return NNs


if __name__ == '__main__':
    vgg_dist = InceptionDistCalculator()
    x = torch.ones(16,3,128,128)
    y = torch.zeros(16,3,128,128)
    d = vgg_dist.__call__(x,y)
    print(d)
