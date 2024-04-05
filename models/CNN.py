from math import ceil

import torch
from torch import nn as nn
from models.DCGAN import conv_block


def compute_receptive_field(depth, k):
    rf = 1
    for i in range(depth):
        rf = rf * 2 + max(0,k-2)

    return rf


def compute_final_feature_size(im_dim, depth, k):
    x = im_dim
    for i in range(depth):
        x  = ceil((x - k + 1) / 2)
    return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, channels=3, depth=3, nf=64, normalize='none', k=3, pad=0, GAP="True", **kwargs):
        super(Discriminator, self).__init__()
        self.GAP = GAP == "True"
        self.input_dim = input_dim
        depth = int(depth)
        nf = int(nf)
        k = int(k)
        normalize = str(normalize)
        print(f"Discriminator receptive field is {compute_receptive_field(depth, k)}")

        layers = [conv_block(channels, nf, k, 2, pad, normalize='none', transpose=False)]  # bn is not good for RGB values
        for i in range(depth - 1):
            layers.append(
                conv_block(nf * 2**i, nf * 2**(i+1), k, 2, pad, normalize=normalize, transpose=False)
            )
        self.convs = nn.Sequential(*layers)
        if self.GAP:
            print("GAP mode")
            self.classifier = nn.Linear(nf * 2**(depth-1), 1, bias=False)
        else:
            print("FC mode")
            d = compute_final_feature_size(self.input_dim, depth, k)
            self.classifier = nn.Linear((d**2) * (nf * 2 ** (depth - 1)), 1, bias=False)

    def features(self, img):
        return self.convs(img)

    def forward(self, img):
        b = img.size(0)
        features = self.convs(img)
        if self.GAP:
            features = torch.mean(features, dim=(2, 3)) # GAP
        else:
            features = features.reshape(features.shape[0], -1)
        output = self.classifier(features).view(b)
        return output


if __name__ == '__main__':
    # print(compute_receptive_field(3, 1))
    # x = torch.ones(5, 3, 64, 64)
    # D = Discriminator(64, depth=3, k=3, normalize='none')
    # print(D(x).shape)
    # D = Discriminator(64, depth=3, k=3, normalize='none', GAP=True)
    # print(D(x).shape)
    im_dim=128
    k=5
    x = torch.ones(5, 3, im_dim, im_dim)
    for d in [1,2,3,4]:
        D = Discriminator(None, depth=d, k=k, normalize='none')
        print(compute_final_feature_size(im_dim, d, k))
        print(D.features(x).shape[-1])