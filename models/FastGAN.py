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
    def __init__(self, input_dim=128, z_dim=100, skip_connections='True', c=3, **kwargs):
        super(Generator, self).__init__()
        self.skip_connections = skip_connections == 'True'
        self.input_dim = input_dim
        ngf = 64
        nfc_multi = {4 :16, 8 :8, 16 :4, 32 :2, 64 :2, 128 :1, 256:0.5}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int( v *ngf)

        self.init = InitLayer(z_dim, channel=nfc[4])

        self.feat_8   = UpBlockComp(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        if input_dim > 64:
            self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        if input_dim > 128:
            self.feat_256 = UpBlock(nfc[128], nfc[256])

        if self.skip_connections:
            self.se_64 = SEBlock(nfc[4], nfc[64])
            if input_dim > 64:
                self.se_128 = SEBlock(nfc[8], nfc[128])
            if input_dim > 128:
                self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_full = conv2d(nfc[input_dim], c, 3, 1, 1, bias=False)
        self.apply(weights_init)

    def forward(self, input):
        feat_4   = self.init(input)
        feat_8   = self.feat_8(feat_4)
        feat_16  = self.feat_16(feat_8)
        feat_32  = self.feat_32(feat_16)

        feat_64 = self.feat_64(feat_32)
        if self.skip_connections:
            feat_64  = self.se_64(feat_4, feat_64)
        if self.input_dim == 64:
            return self.to_full(feat_64)

        feat_128 = self.feat_128(feat_64)
        if self.skip_connections:
            feat_128 = self.se_128(feat_8, feat_128)
        if self.input_dim == 128:
            return self.to_full(feat_128)

        feat_256 = self.feat_256(feat_128)
        if self.skip_connections:
            feat_256 = self.se_256(feat_16, feat_256)
        return torch.tanh(self.to_full(feat_256))
        # return self.to_full(feat_256)



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
    def __init__(self, input_dim=128, num_outputs=1, skip_connections='True', **kwargs):
        super(Discriminator, self).__init__()
        self.ndf = 32
        self.input_dim = input_dim
        self.skip_connections = skip_connections == 'True'
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
        self.num_outputs = num_outputs

        nc = 3
        nfc_multi = {4: 32, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256:0.5}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * self.ndf)

        self.conv1 = nn.Sequential(
            conv2d(nc, nfc[input_dim], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.down_2 = DownBlockComp(nfc[input_dim], nfc[input_dim//2])
        self.down_4 = DownBlockComp(nfc[input_dim//2], nfc[input_dim//4])
        self.down_8 = DownBlockComp(nfc[input_dim//4], nfc[input_dim//8])
        if self.skip_connections:
            self.se_1_8 = SEBlock(nfc[input_dim], nfc[input_dim//8])

        final_dim = input_dim//8
        if input_dim > 64:
            self.down_16 = DownBlockComp(nfc[input_dim//8], nfc[input_dim//16])
            if self.skip_connections:
                self.se_2_16 = SEBlock(nfc[input_dim//2], nfc[input_dim//16])
            final_dim = input_dim // 16

        if input_dim > 128:
            self.down_32 = DownBlockComp(nfc[input_dim//16], nfc[input_dim//32])
            if self.skip_connections:
                self.se_4_32 = SEBlock(nfc[input_dim//4], nfc[input_dim//32])
            final_dim = input_dim // 32

        self.spatial_logits = nn.Sequential(
                            conv2d(nfc[final_dim] , nfc[final_dim//2], 1, 1, 0, bias=False),
                            batchNorm2d(nfc[final_dim//2]), nn.LeakyReLU(0.2, inplace=True),
                            conv2d(nfc[final_dim//2], 1, 4, 1, 0, bias=False))

        self.decoder_big = SimpleDecoder(nfc[final_dim], nc)
        self.apply(weights_init)

    def features(self, img):
        feat_x1 = self.conv1(img)
        feat_x2 = self.down_2(feat_x1)
        feat_x4 = self.down_4(feat_x2)

        feat_x8 = self.down_8(feat_x4)
        if self.skip_connections:
            feat_x8 = self.se_1_8(feat_x1, feat_x8)
        if self.input_dim == 64:
            return feat_x8

        feat_x16 = self.down_16(feat_x8)
        if self.skip_connections:
            feat_x16 = self.se_2_16(feat_x2, feat_x16)
        if self.input_dim == 128:
            return feat_x16

        feat_x32 = self.down_32(feat_x16)
        if self.skip_connections:
            feat_x32 = self.se_4_32(feat_x4, feat_x32)
        return feat_x32

    def forward(self, img, reconstruct=False):
        feat_8 = self.features(img)
        output = self.spatial_logits(feat_8)
        if self.num_outputs == 1:
            output = output.view(-1)
        else:
            output = output.view(-1, self.num_outputs)

        if reconstruct:
            return output, self.decoder_big(feat_8)
        return output


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 8 x 8
        return self.main(input)

if __name__ == '__main__':
    D = Discriminator()
    x = torch.ones((6,3,128,128))
    print(D(x).shape)