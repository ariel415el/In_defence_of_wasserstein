import torch
from torch import nn as nn
from models.DCGAN import conv_block

def compute_receptive_field(depth, k):
    rf = 1
    for i in range(depth):
        rf = rf * 2 + k-2

    return rf

class Discriminator(nn.Module):
    def __init__(self, input_dim, depth=4, nf=64, bn=True, k=4, pad=1, **kwargs):
        print(f"Discriminator receptive field is {compute_receptive_field(depth, k)}")
        super(Discriminator, self).__init__()
        channels=3
        depth = int(depth)
        nf = int(nf)

        layers = [conv_block(channels, nf, k, 2, pad, use_bn=bn, transpose=False)]
        for i in range(depth - 1):
            layers.append(
                conv_block(nf * 2**i, nf * 2**(i+1), k, 2, pad, use_bn=bn, transpose=False)
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