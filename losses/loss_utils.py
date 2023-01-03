import torch


def calc_gradient_penalty(netD, real_data, fake_data):
    """Ensure the netD is smooth by forcing the gradient between real and fake data to ahve norm of 1"""
    device = real_data.device
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def get_dist_metric(name):
    """Choose how to calulate pairwise distances for EMD"""
    if name == 'L1':
        metric = L1_metric()
    elif name == 'L2':
        metric = L2_metric()
    elif name == 'vgg':
         metric = vgg_dist_calculator()
    elif name == 'inception':
        metric = inception_dist_calculator()
    else:
        raise ValueError(f"No such metric name {name}")
    return metric


class L1_metric:
    def __call__(self, X, Y):
        X = X.view(len(X), -1)
        Y = Y.view(len(Y), -1)
        return torch.mean(torch.abs(X[:, None] - Y[None, :]), dim=-1)

class L2_metric:
    def __call__(self, X, Y):
        X = X.view(len(X), -1)
        Y = Y.view(len(Y), -1)
        return torch.mean((X[:, None] - Y[None, :])**2, dim=-1)

def get_batche_slices(n, b):
    n_batches = n // b
    slices = []
    for i in range(n_batches):
        slices.append(slice(i * b, (i + 1) * b))

    if n % b != 0:
        slices.append(slice(n_batches * b, n))
    return slices


def compute_features_dist_mat_in_batches(X, Y, f, b=64):
    """Compute distance matrix in features of a function f(X) but restrict maximum inference batch to 'b'"""
    dists = -1 * torch.ones(len(X), len(Y))
    x_slices = get_batche_slices(len(X), b)
    y_slices = get_batche_slices(len(Y), b)
    for slice_x in x_slices:
        features_x = f(X[slice_x])
        for slice_y in y_slices:
            features_y = f(Y[slice_y])
            D = features_x[:, None] - features_y[None,]
            D = D.reshape(D.shape[0], D.shape[1], -1) # In case featueres are still spatial
            dists[slice_x, slice_y] = torch.norm(D, dim=-1)
    return dists
class inception_dist_calculator:
    def __init__(self):
        from benchmarking.inception import myInceptionV3
        self.device = None
        self.inception = myInceptionV3()
        self.inception.eval()

    def extract(self, X):
        if self.device is None:
            self.device = X.device
            self.inception.to(self.device)
        return self.inception(X)
    def __call__(self, X, Y, b=64):
        return compute_features_dist_mat_in_batches(X, Y, self.extract, b=b)


class vgg_dist_calculator:
    def __init__(self,  layer=18):
        self.layer = layer  # [4, 9, 18]
        from torchvision import models, transforms
        self.vgg_features = models.vgg19(pretrained=True).features
        self.device = None
        self.vgg_features.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def extract(self, X):
        if self.device is None:
            self.device = X.device
            self.vgg_features.to(self.device)
        for i, layer in enumerate(self.vgg_features):
            X = layer(X)
            # if i in layer_indices:
            if i == self.layer:
                return X

    def __call__(self, X, Y, b=64):
        return compute_features_dist_mat_in_batches(X, Y, self.extract, b=b)


if __name__ == '__main__':
    vgg_dist = inception_dist_calculator()

    x = torch.ones(16,3,128,128)
    y = torch.zeros(16,3,128,128)

    d = vgg_dist.__call__(x,y)
    print(d)
