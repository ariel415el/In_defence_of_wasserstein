import torch
from torch import nn

from models.DCGAN import conv_block


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, **kwargs):
        super(Discriminator, self).__init__()
        self.f1 = conv_block(3, 64, 4, 2, 1)
        self.f2 = conv_block(64, 128, 4, 2, 1)
        self.f3 = conv_block(128, 256, 4, 2, 1)
        self.f4 = conv_block(256, 512, 4, 2, 1)

        self.classifier = nn.Linear(512, 1, bias=False)

    def get_feature_maps(self, img):
        f1 = self.f1(img)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        f4 = self.f4(f3)
        return [f1, f2, f3, f4]

    def forward(self, img):
        b = img.size(0)
        f4 = self.get_feature_maps(img)[-1]
        features = torch.mean(f4, dim=(2, 3)) # GAP
        output = self.classifier(features).view(b)
        return output



if __name__ == '__main__':
    x = torch.ones(5,3,64,64)
    D = Discriminator(64)

    print(D(x).shape)