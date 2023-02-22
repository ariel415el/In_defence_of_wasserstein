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
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        batchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
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
    def __init__(self, z_dim=100, skip_connections=True, c=3, **kwargs):
        super(Generator, self).__init__()
        self.skip_connections = skip_connections
        ngf = 64
        nfc_multi = {4 :16, 8 :8, 16 :4, 32 :2, 64 :2, 128 :1}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int( v *ngf)

        self.init = InitLayer(z_dim, channel=nfc[4])

        self.feat_8   = UpBlockComp(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])

        if self.skip_connections:
            self.se_64  = SEBlock(nfc[4], nfc[64])
            self.se_128 = SEBlock(nfc[8], nfc[128])

        self.to_full = conv2d(nfc[128], c, 1, 1, 0, bias=False)

    def forward(self, input):
        feat_4   = self.init(input)
        feat_8   = self.feat_8(feat_4)
        feat_16  = self.feat_16(feat_8)
        feat_32  = self.feat_32(feat_16)
        feat_64 = self.feat_64(feat_32)
        if self.skip_connections:
            feat_64  = self.se_64(feat_4, feat_64)
        feat_128 = self.feat_128(feat_64)
        if self.skip_connections:
            feat_128 = self.se_128(feat_8, feat_128)

        im_full = torch.tanh(self.to_full(feat_128))

        return im_full


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.ndf = 48
        nc = 3
        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * self.ndf)

        self.down_from_full = nn.Sequential(
            conv2d(nc, nfc[128], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.down_64 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64], nfc[32])
        self.down_16 = DownBlockComp(nfc[32], nfc[16])
        self.down_8 = DownBlockComp(nfc[16], nfc[8])
        # self.down_4 = DownBlockComp(nfc[8], nfc[4])

        self.spatial_logits = nn.Sequential(
            conv2d(nfc[8], nfc[4], 4, 2, 0, bias=False),
            batchNorm2d(nfc[4]),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[4], 1, 3, 1, 0, bias=False))

    def features(self, img):
        feat_128 = self.down_from_full(img)
        feat_64 = self.down_64(feat_128)
        feat_32 = self.down_32(feat_64)
        feat_16 = self.down_16(feat_32)
        feat_8 = self.down_8(feat_16)
        return feat_8

    def forward(self, img):
        feat_8 = self.features(img)
        output = self.spatial_logits(feat_8).view(len(img))

        return output

if __name__ == '__main__':
    D = Discriminator()
    x = torch.ones((6,3,128,128))
    print(D(x).shape)