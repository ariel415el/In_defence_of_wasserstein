import torch
from torch import nn as nn
from models.DCGAN import conv_block


def compute_receptive_field(depth, k):
    rf = 1
    for i in range(depth):
        rf = rf * 2 + k-2

    return rf


class Discriminator(nn.Module):
    def __init__(self, input_dim, depth=3, nf=64, normalize='in', k=4, pad=1, **kwargs):
        super(Discriminator, self).__init__()
        channels=3
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
        self.classifier = nn.Linear(nf * 2**(depth-1), 1, bias=False)

    def features(self, img):
        return self.convs(img)

    def forward(self, img):
        b = img.size(0)
        features = self.convs(img)
        features = torch.mean(features, dim=(2, 3)) # GAP
        output = self.classifier(features).view(b)
        return output



if __name__ == '__main__':
    x = torch.ones(5,3,64,64)
    D = Discriminator(4)

    print(D(x).shape)