import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_dim=64,
                 ksize=7,
                 hdim=128,
                 n_local_layers=5,
                 strides=1):
        super().__init__()
        layers = [nn.Conv2d(3, hdim, ksize, stride=strides, padding=ksize // 2)]
        for i in range(n_local_layers):
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Conv2d(hdim, hdim, 1))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=hdim, out_features=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x).view(len(x))
        return x

