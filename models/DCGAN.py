import torch
from torch import nn as nn


def conv_block(c_in, c_out, k_size, stride, pad, normalize='in', transpose=False):
    module = []

    conv_type = nn.ConvTranspose2d if transpose else nn.Conv2d
    module.append(conv_type(c_in, c_out, k_size, stride, pad, bias=normalize == 'none'))

    if normalize == "bn":
        module.append(nn.BatchNorm2d(c_out))
    elif normalize == "in":
        module.append(nn.InstanceNorm2d(c_out))

    module.append(nn.ReLU(True))
    return nn.Sequential(*module)


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim=64, nf=64,  normalize='none', channels=3, **kwargs):
        super(Generator, self).__init__()
        nf = int(nf)
        normalize = str(normalize)
        layer_depths = [z_dim, nf*8, nf*4, nf*2, nf]
        kernel_dim = [4, 4, 4, 4, 4]
        strides = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]

        if output_dim == 128:
            layer_depths += [nf//2]
            kernel_dim += [4]
            strides += [2]
            padding += [1]

        if output_dim == 256:
            layer_depths += [nf//2, nf//4]
            kernel_dim += [4, 4]
            strides += [2, 2]
            padding += [1, 1]

        layers = []
        for i in range(len(layer_depths) - 1):
            layers.append(
                conv_block(layer_depths[i], layer_depths[i + 1], kernel_dim[i], strides[i], padding[i], normalize=normalize, transpose=True)
            )
        layers += [
            nn.ConvTranspose2d(layer_depths[-1], channels, kernel_dim[-1], strides[-1], padding[-1]),
            nn.Tanh()
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.network(input)
        return output


class Discriminator(nn.Module):
    """ DC-discriminator receptive field by layer (4, 10, 22, 46, 94)"""
    def __init__(self, input_dim=64, nf='64', normalize='in', num_outputs=1, channels=3, **kwargs):
        super(Discriminator, self).__init__()
        nf = int(nf)
        normalize = str(normalize)
        input_dim = int(input_dim)
        num_outputs = int(num_outputs)
        layer_depth = [channels, nf, nf*2, nf*4, nf*8]
        if input_dim == 128:
            layer_depth += [nf*16]
        if input_dim == 256:
            layer_depth += [nf * 16, nf*32]

        layers = []
        for i in range(len(layer_depth) - 1):
            normalize_layer = normalize if i > 0 else 'none'  # bn is not good for RGB values
            layers.append(
                conv_block(layer_depth[i], layer_depth[i + 1], 4, 2, 1, normalize=normalize_layer, transpose=False)
            )
        self.convs = nn.Sequential(*layers)
        self.classifier = nn.Linear(layer_depth[-1]*4**2, num_outputs)
        self.num_outputs = num_outputs

    def features(self, img):
        return self.convs(img)

    def forward(self, img):
        b = img.size(0)
        features = self.convs(img)

        features = features.reshape(b, -1)

        output = self.classifier(features)
        if self.num_outputs == 1:
            output = output.view(len(img))
        else:
            output = output.view(len(img), self.num_outputs)

        return output


if __name__ == '__main__':
    netD = Discriminator(bn=False)
    netD(torch.ones(5,3,32,32))