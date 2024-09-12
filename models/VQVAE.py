
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class Encoder(nn.Module):
    def __init__(self, dims=[3,32,64,128,256]):
        super(Encoder, self).__init__()
        layers = []
        for i in range(len(dims)-1):
            cout = dims[i+1]
            layers += [
            nn.Conv2d(dims[i], cout, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            ]
        self.encoder = nn.Sequential(*layers)
        self.f = len(dims)-1


    def forward(self, x):
        x = self.encoder(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dims=[256,128,64,32,3]):
        super(Decoder, self).__init__()
        layers = []
        for i in range(len(dims)-2):
            cout = dims[i+1]
            layers += [
                ResBlock(dims[i], dims[i]),
                nn.ConvTranspose2d(dims[i], cout, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
            ]
        layers += [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dims[-2], dims[-1], kernel_size=4, stride=2, padding=1),
            ]

        self.decoder = nn.Sequential(*layers)
        self.f = len(dims)-1

    def forward(self, z):
        # z = z.view(-1, self.d, self.f, self.f)
        z = self.decoder(z)
        return torch.tanh(z)



if __name__ == '__main__':
    ENCODER = Encoder()
    DECODER = Decoder()
    x = torch.randn(1, 3, 256, 256)
    print(ENCODER(x).shape)
    print(DECODER(ENCODER(x)).shape)
