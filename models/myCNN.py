import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def convTranspose2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


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


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2),
            GLU())

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        batchNorm2d(out_planes * 2),
        GLU())
    return block


class Generator(nn.Module):
    def __init__(self, output_dim=128, z_dim=100, c=3, nf='64', **kwargs):
        super(Generator, self).__init__()
        self.output_dim = output_dim
        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5}
        nfc = {k: int(v * int(nf)) for k, v in nfc_multi.items()}

        self.init = InitLayer(z_dim, channel=nfc[4])

        self.feat_8 = UpBlock(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlock(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        if output_dim > 64:
            self.feat_128 = UpBlock(nfc[64], nfc[128])
        if output_dim > 128:
            self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.to_full = conv2d(nfc[output_dim], c, 3, 1, 1, bias=False)
        self.apply(weights_init)

    def forward(self, input):
        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.feat_64(feat_32)
        if self.output_dim == 64:
            return self.to_full(feat_64)

        feat_128 = self.feat_128(feat_64)
        if self.output_dim == 128:
            return self.to_full(feat_128)

        feat_256 = self.feat_256(feat_128)
        return torch.tanh(self.to_full(feat_256))


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
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2)
        )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, input_dim=128, num_outputs=1, nf='32', **kwargs):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.num_outputs = num_outputs

        nc = 3
        nfc_multi = {4: 32, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5}
        nfc = {k: int(v * int(nf)) for k, v in nfc_multi.items()}

        # Build layers
        conv1 = nn.Sequential(
            conv2d(nc, nfc[input_dim], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        down_2 = DownBlockComp(nfc[input_dim], nfc[input_dim // 2])
        down_4 = DownBlockComp(nfc[input_dim // 2], nfc[input_dim // 4])
        down_8 = DownBlockComp(nfc[input_dim // 4], nfc[input_dim // 8])
        layers = [conv1, down_2, down_4, down_8]
        final_dim = input_dim // 8
        if input_dim > 64:
            down_16 = DownBlockComp(nfc[input_dim // 8], nfc[input_dim // 16])
            layers.append(down_16)
            final_dim = input_dim // 16

        if input_dim > 128:
            down_32 = DownBlockComp(nfc[input_dim // 16], nfc[input_dim // 32])
            layers.append(down_32)
            final_dim = input_dim // 32

        self.features = nn.Sequential(*layers)

        self.spatial_logits = nn.Sequential(
            conv2d(nfc[final_dim], nfc[final_dim // 2], 1, 1, 0, bias=False),
            batchNorm2d(nfc[final_dim // 2]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[final_dim // 2], 1, 4, 1, 0, bias=False))

        self.apply(weights_init)

    def forward(self, img):
        features = self.features(img)
        output = self.spatial_logits(features)
        if self.num_outputs == 1:
            output = output.view(-1)
        else:
            output = output.view(-1, self.num_outputs)

        return output


if __name__ == '__main__':
    D = Discriminator()
    x = torch.ones((6, 3, 128, 128))
    print(D(x).shape)
