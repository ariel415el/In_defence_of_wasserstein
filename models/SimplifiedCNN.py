import torch
from torch import nn


class SCNNNetwork(nn.Module):
    def __init__(self,
                 ksize=3,
                 n_classes=1,
                 hdim=1024,
                 n_local_layers=6,
                 normalize_spatial_filter=False,
                 strides=1,
                 pool_size=32,
                 pool_strides=16):
        super().__init__()
        spatial_conv = nn.Conv2d(3, hdim, ksize, stride=strides, padding=ksize // 2)
        layers = [spatial_conv]
        for i in range(n_local_layers):
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Conv2d(hdim, hdim, 1))
        layers.append(nn.LeakyReLU(0.2))

        if pool_size is not None:
            layers.append(nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides))
        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=hdim, out_features=n_classes))
        for i, l in enumerate(layers):
            setattr(self, "layer{}".format(i), l)
        self.layers = layers
        self.ksize = ksize
        self.normalize_spatial_filter = normalize_spatial_filter

    def forward(self, x):
        for i, l in self.children():
            x = l(x)
        return x

    def reinitialize(self, gain=None):
        for i, layer in enumerate(self.children()):
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                if i < (len(self.layers) - 1):
                    nn.init.kaiming_normal_(layer.weight, a=0.2)
                    if i == 0 and self.normalize_spatial_filter:
                        layer.weight.data -= torch.mean(layer.weight.data, dim=(2, 3), keepdim=True)
                else:
                    nn.init.xavier_normal_(layer.weight, gain=gain if gain is not None else 1.)
                if hasattr(layer, "bias"):
                    nn.init.zeros_(layer.bias.data)

