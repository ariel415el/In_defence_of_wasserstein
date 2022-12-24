import torch


def calc_gradient_penalty(netD, real_data, fake_data):
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


def get_batche_slices(n, b):
    n_batches = n // b
    slices = []
    for i in range(n_batches):
        slices.append(slice(i * b, (i + 1) * b))

    if n % b != 0:
        slices.append(slice(n_batches * b, n))
    return slices


class vgg_dist_calculator:
    def __init__(self,  layer=18):
        self.layer = layer  # [4, 9, 18]
        from torchvision import models, transforms
        self.vgg_features = models.vgg19(pretrained=True).features
        self.device = None
        self.vgg_features.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def extract_intermediate_feature_maps(self, x):
        for i, layer in enumerate(self.vgg_features):
            x = layer(x)
            # if i in layer_indices:
            if i == self.layer:
                return x

    def get_dist_mat(self, X, Y, b=64):
        if self.device is None:
            self.device = X.device
            self.vgg_features.to(self.device)
        x_slices = get_batche_slices(len(X), b)
        y_slices = get_batche_slices(len(Y), b)
        dists = -1 * torch.ones(len(X), len(Y))
        for slice_x in x_slices:
            features_x = self.extract_intermediate_feature_maps(X[slice_x])
            for slice_y in y_slices:
                features_y = self.extract_intermediate_feature_maps(Y[slice_y])
                D = features_x[:, None] - features_y[None, ]
                D = D.reshape(slice_x.stop - slice_x.start, slice_y.stop - slice_y.start, -1)
                dists[slice_x, slice_y] = torch.norm(D, dim=-1)

        return dists

if __name__ == '__main__':
    vgg_dist = vgg_dist_calculator(torch.device("cpu"))

    x = torch.ones(16,3,128,128)
    y = torch.zeros(16,3,128,128)

    d = vgg_dist.get_dist_mat(x,y)
    print(d.min())
