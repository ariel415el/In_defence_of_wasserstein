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
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def conv_block(c_in, c_out, k_size, stride, pad, use_bn=True, transpose=False):
    module = []

    conv_type = nn.ConvTranspose2d if transpose else nn.Conv2d
    module.append(conv_type(c_in, c_out, k_size, stride, pad, bias=not use_bn))

    if use_bn:
        module.append(nn.BatchNorm2d(c_out))

    module.append(nn.ReLU(True))
    return nn.Sequential(*module)


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim=64, bn=True, **kwargs):
        channels = 3
        super(Generator, self).__init__()
        layer_depths = [z_dim, 512, 512, 256, 128]
        kernel_dim = [4, 4, 4, 4, 4]
        strides = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]

        if output_dim == 128:
            layer_depths += [64]
            kernel_dim += [4]
            strides += [2]
            padding += [1]

        layers = []
        for i in range(len(layer_depths) - 1):
            layers.append(
                conv_block(layer_depths[i], layer_depths[i + 1], kernel_dim[i], strides[i], padding[i], use_bn=bn, transpose=True)
            )
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
    def __init__(self, input_dim=64, bn=True, GAP=False, **kwargs):
        super(Discriminator, self).__init__()
        self.GAP = GAP
        channels=3

        # layer_depth = [channels, 32, 64, 128, 256]
        layer_depth = [channels, 64, 128, 256, 512]
        if input_dim == 128:
            layer_depth += [512]

        layers = []
        for i in range(len(layer_depth) - 1):
            layers.append(
                conv_block(layer_depth[i], layer_depth[i + 1], 4, 2, 1, use_bn=bn, transpose=False)
            )
        self.convs = nn.Sequential(*layers)
        if GAP:
            self.classifier = nn.Linear(layer_depth[-1], 1, bias=False)
        else:
            self.classifier = nn.Linear(layer_depth[-1]*4**2, 1, bias=False)

    def features(self, img):
        return self.convs(img)

    def forward(self, img):
        b = img.size(0)
        features = self.convs(img)
        if self.GAP:
            features = torch.mean(features, dim=(2, 3)) # GAP
        else:
            features = features.reshape(b, -1)
        output = self.classifier(features).view(b)
        return output


if __name__ == '__main__':
    z = torch.ones(5,128)
    x = torch.ones(5,3,64,64)
    G = Generator(128, 128)
    D = Discriminator(128)
    from model_utils.discriminator_ensamble import Ensemble
    D = Ensemble(D, 4)

    print(G(z).shape)
    print(D(G(z)).shape)