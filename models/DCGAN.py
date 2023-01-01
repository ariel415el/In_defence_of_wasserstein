import numpy as np
import torch
import torch.nn.init as init
from torch import nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))

class Generator(nn.Module):
    def __init__(self, z_dim):
        channels = 3
        super(Generator, self).__init__()
        layer_depths = [z_dim, 512, 512, 256, 128]
        kernel_dim = [4, 4, 4, 4, 4]
        strides = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]
        layers = []
        for i in range(len(layer_depths) - 1):
            layers += [
                nn.ConvTranspose2d(layer_depths[i], layer_depths[i + 1], kernel_dim[i], strides[i], padding[i],
                                   bias=False),
                nn.BatchNorm2d(layer_depths[i + 1]),
                nn.ReLU(True),
            ]
        layers += [
            nn.ConvTranspose2d(layer_depths[-1], channels, kernel_dim[-1], strides[-1], padding[-1], bias=False),
            nn.Tanh()
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.network(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        channels=3

        layer_depth = [channels, 128, 256, 512, 512]
        layers = []
        for i in range(len(layer_depth) - 1):
            layers += [
                nn.Conv2d(layer_depth[i], layer_depth[i + 1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(layer_depth[i + 1]),
                nn.ReLU(True)
            ]
        self.convs = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(layer_depth[-1], 1, 4, 1, 0, bias=False)

    def features(self, img):
        return self.convs(img)

    def forward(self, img):
        features = self.convs(img)
        output = self.classifier(features).view(img.size(0))
        return output
