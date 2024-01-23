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


class Generator(nn.Module):
    def __init__(self, output_dim=128, z_dim=100, channels=3, nf='64', skip_connections=True, **kwargs):
        super(Generator, self).__init__()
        self.skip_connections = bool(skip_connections)
        self.output_dim = output_dim
        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5}
        nfc = {k: int(v * int(nf)) for k, v in nfc_multi.items()}

        self.init = InitLayer(z_dim, channel=nfc[4])

        self.feat_8 = UpBlock(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlock(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.se_64 = SEBlock(nfc[4], nfc[64])
        if output_dim > 64:
            self.feat_128 = UpBlock(nfc[64], nfc[128])
            self.se_128 = SEBlock(nfc[8], nfc[128])
        if output_dim > 128:
            self.feat_256 = UpBlock(nfc[128], nfc[256])
            self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_full = conv2d(nfc[output_dim], channels, 3, 1, 1, bias=False)
        self.apply(weights_init)

    def forward(self, input):
        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.feat_64(feat_32)
        if self.skip_connections:
            feat_64 = self.se_64(feat_4, feat_64)
        if self.output_dim == 64:
            return self.to_full(feat_64)

        feat_128 = self.feat_128(feat_64)
        if self.skip_connections:
            feat_128 = self.se_128(feat_8, feat_128)
        if self.output_dim == 128:
            return self.to_full(feat_128)

        feat_256 = self.feat_256(feat_128)
        if self.skip_connections:
            feat_256 = self.se_256(feat_16, feat_256)
        return self.to_full(feat_256)


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
    def __init__(self, input_dim=128, num_outputs=1, channels=3, nf='64', skip_connections=True, **kwargs):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.skip_connections = bool(skip_connections)
        self.num_outputs = num_outputs

        nfc_multi = {4: 32, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5}
        nfc = {k: int(v * int(nf)) for k, v in nfc_multi.items()}

        # Build layers
        self.conv1 = nn.Sequential(
            conv2d(channels, nfc[input_dim], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.down_2 = DownBlockComp(nfc[input_dim], nfc[input_dim // 2])
        self.down_4 = DownBlockComp(nfc[input_dim // 2], nfc[input_dim // 4])
        self.down_8 = DownBlockComp(nfc[input_dim // 4], nfc[input_dim // 8])

        if self.skip_connections:
            self.skip_0_to_8 = SEBlock(nfc[input_dim], nfc[input_dim // 8])

        final_dim = input_dim // 8

        if input_dim > 64:
            self.down_16 = DownBlockComp(nfc[input_dim // 8], nfc[input_dim // 16])
            final_dim = input_dim // 16
            if self.skip_connections:
                self.skip_2_to_16 = SEBlock(nfc[input_dim // 2], nfc[input_dim // 16])

        if input_dim > 128:
            self.down_32 = DownBlockComp(nfc[input_dim // 16], nfc[input_dim // 32])
            final_dim = input_dim // 32
            if self.skip_connections:
                self.skip_4_to_32 = SEBlock(nfc[input_dim // 4], nfc[input_dim // 32])

        self.spatial_logits = nn.Sequential(
            conv2d(nfc[final_dim], nfc[final_dim // 2], 1, 1, 0, bias=False),
            batchNorm2d(nfc[final_dim // 2]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[final_dim // 2], 1, 4, 1, 0, bias=False))

        self.apply(weights_init)

    def features(self, x):
        feat_0 = self.conv1(x)
        feat_2 = self.down_2(feat_0)
        feat_4 = self.down_4(feat_2)

        feat_8 = self.down_8(feat_4)
        if self.skip_connections:
            feat_8 = self.skip_0_to_8(feat_0, feat_8)
        if self.input_dim == 64:
            return feat_8

        feat_16 = self.down_16(feat_8)
        if self.skip_connections:
            feat_16 = self.skip_2_to_16(feat_2, feat_16)
        if self.input_dim == 128:
            return feat_16

        feat_32 = self.down_32(feat_16)
        if self.skip_connections:
            feat_32 = self.skip_4_to_32(feat_4, feat_32)
        return feat_32  # self.input_dim == 256

    def forward(self, img):
        features = self.features(img)
        output = self.spatial_logits(features)
        if self.num_outputs == 1:
            output = output.view(-1)
        else:
            output = output.view(-1, self.num_outputs)

        return output


if __name__ == '__main__':
    dim = 64
    G = Generator(output_dim=dim, z_dim=dim)
    z = torch.ones((6, dim))
    print(G(z).shape)
    D = Discriminator(input_dim=dim)
    x = torch.ones((6, 3, dim, dim))
    print(D(x).shape)
