import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def conv2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4),
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes, down_sample=True):
        super(DownBlockComp, self).__init__()
        stride = 2 if down_sample else 1
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 3, stride, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2)
        )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2) if down_sample else nn.Identity(),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, input_dim=128, num_outputs=1, channels=3, nf='48', depth=3, **kwargs):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.num_outputs = num_outputs

        nfc_multi = {4: 32, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5}
        nfc = {k: int(v * int(nf)) for k, v in nfc_multi.items()}

        # Build layers
        layers =[
            conv2d(channels, nfc[input_dim], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)]
        cur_dim = input_dim
        for i in range(depth):
            layers.append(DownBlockComp(nfc[cur_dim], nfc[cur_dim // 2], down_sample=i < depth-1))
            cur_dim = cur_dim // 2

        self.features = nn.Sequential(*layers)

        self.spatial_logits = nn.Sequential(
            conv2d(nfc[cur_dim], nfc[cur_dim // 2], 1, 1, 0, bias=False),
            batchNorm2d(nfc[cur_dim // 2]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[cur_dim // 2], 1, 4, 1, 0, bias=False))

        self.apply(weights_init)

    def forward(self, img):
        features = self.features(img)
        print(features.shape)
        output = self.spatial_logits(features)
        if self.num_outputs == 1:
            output = output.view(-1)
        else:
            output = output.view(-1, self.num_outputs)

        return output


if __name__ == '__main__':
    dim = 64
    D = Discriminator(input_dim=dim, depth=3)
    x = torch.ones((6, 3, dim, dim))
    print(D(x).shape)
