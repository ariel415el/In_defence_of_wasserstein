import torch
from torch import nn


class PixelGenerator(nn.Module):
    def __init__(self, z_dim, n=64, b=64):
        super(PixelGenerator, self).__init__()
        self.n = int(n)
        self.b = int(b)
        self.outputs = nn.Parameter(torch.randn(self.n, 2), requires_grad=True)

    def forward(self, input):
        if self.b < self.n:
            return self.outputs[torch.randperm(self.n)[:self.b]]
        else:
            return self.outputs

def block(in_feat, out_feat, normalize='in'):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize == "bn":
        layers.append(nn.BatchNorm1d(out_feat))
    elif normalize == "in":
        layers.append(nn.InstanceNorm1d(out_feat))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

class FCGenerator(nn.Module):
    def __init__(self, z_dim=64,  out_dim=2, nf=128, depth=2, normalize='none', **kwargs):
        super(FCGenerator, self).__init__()
        nf = int(nf)
        depth = int(depth)

        layers = block(z_dim, nf, normalize='none') # bn is not good for RGB values

        for i in range(depth - 1):
            layers += block(nf, nf, normalize=normalize)

        layers += [nn.Linear(nf, out_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        validity = self.model(img)

        return validity


class Discriminator(nn.Module):
    def __init__(self, input_dim=2,  nf=128, depth=2, normalize='none', **kwargs):
        super(Discriminator, self).__init__()
        nf = int(nf)
        depth = int(depth)

        layers = block(input_dim, nf, normalize='none') # bn is not good for RGB values

        for i in range(depth - 1):
            layers += block(nf, nf, normalize=normalize)

        layers += [nn.Linear(nf, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat).view(img.size(0))

        return validity