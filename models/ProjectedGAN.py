"""Code taken and slightly modified from https://github.com/RangiLyu/EfficientNet-Lite"""

import math
import os

from torch import nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.nn.utils import spectral_norm
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def kaiming_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')


def round_filters(filters, multiplier, divisor=8, min_width=None):
    """Calculate and round number of filters based on width multiplier."""
    if not multiplier:
        return filters
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x


class MBConvBlock(nn.Module):
    def __init__(self, inp, final_oup, k, s, expand_ratio, se_ratio, has_se=False):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, padding=(k - 1) // 2, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._momentum, eps=self._epsilon)
        self._relu = nn.ReLU6(inplace=True)

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._relu(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1  and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x


class EfficientNetLite(nn.Module):
    def __init__(self, widthi_multiplier, depth_multiplier, num_classes, drop_connect_rate, dropout_rate):
        super(EfficientNetLite, self).__init__()

        # Batch norm parameters
        momentum = 0.01
        epsilon = 1e-3
        self.drop_connect_rate = drop_connect_rate

        mb_block_settings = [
            #repeat|kernal_size|stride|expand|input|output|se_ratio
            [1, 3, 1, 1, 32,  16,  0.25],
            [2, 3, 2, 6, 16,  24,  0.25],
            [2, 5, 2, 6, 24,  40,  0.25],
            [3, 3, 2, 6, 40,  80,  0.25],
            [3, 5, 1, 6, 80,  112, 0.25],
            [4, 5, 2, 6, 112, 192, 0.25],
            [1, 3, 1, 6, 192, 320, 0.25]
        ]

        # Stem
        out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        # Build blocks
        self.blocks = nn.ModuleList([])
        for i, stage_setting in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            num_repeat, kernal_size, stride, expand_ratio, input_filters, output_filters, se_ratio = stage_setting
            # Update block input and output filters based on width multiplier.
            input_filters = input_filters if i == 0 else round_filters(input_filters, widthi_multiplier)
            output_filters = round_filters(output_filters, widthi_multiplier)
            num_repeat= num_repeat if i == 0 or i == len(mb_block_settings) - 1  else round_repeats(num_repeat, depth_multiplier)

            # The first block needs to take care of stride and filter size increase.
            stage.append(MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio, has_se=False))
            if num_repeat > 1:
                input_filters = output_filters
                stride = 1
            for _ in range(num_repeat - 1):
                stage.append(MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio, has_se=False))

            self.blocks.append(stage)

        # Head
        in_channels = round_filters(mb_block_settings[-1][5], widthi_multiplier)
        out_channels = 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.fc = torch.nn.Linear(out_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        features = []
        # print(x.shape)
        x = self.stem(x)
        idx = 0
        for i, stage in enumerate(self.blocks):
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.blocks)
                x = block(x, drop_connect_rate)
                idx +=1
            if i in [1, 2, 3, 6]:
                features.append(x)
            # print(f"After block{i}", x.shape)
        # x = self.head(x)
        # # print(x.shape)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # if self.dropout is not None:
        #     x = self.dropout(x)
        # x = self.fc(x)
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

    def load_pretrain(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)


def load_checkpoint(net, checkpoint):
    from collections import OrderedDict

    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    # for k in checkpoint:
    #     k2 = 'module.'+k if not k.startswith('module.') else k
    #     temp[k2] = checkpoint[k]

    net.load_state_dict(checkpoint, strict=True)


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, 4, 2, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.leaky_relu(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, channels, l):
        super(MultiScaleDiscriminator, self).__init__()
        self.head_conv = spectral_norm(nn.Conv2d(512, 1, 3, 1, 1))

        # layers = []
        # if l == 1:
        #     layers.append(DownBlock(c_in, 64))
        #     layers.append(DownBlock(64, 128))
        #     layers.append(DownBlock(128, 256))
        #     layers.append(DownBlock(256, 512))
        # elif l == 2:
        #     layers.append(DownBlock(c_in, 128))
        #     layers.append(DownBlock(128, 256))
        #     layers.append(DownBlock(256, 512))
        # elif l == 3:
        #     layers.append(DownBlock(c_in, 256))
        #     layers.append(DownBlock(256, 512))
        # else:
        #     layers.append(DownBlock(c_in, 512))

        layers = [DownBlock(channels, 64 * [1, 2, 4, 8][l - 1])] + [DownBlock(64 * i, 64 * i * 2) for i in [1, 2, 4][l - 1:]]
        self.model = nn.Sequential(*layers)
        self.optim = Adam(self.model.parameters(), lr=0.0002, betas=(0, 0.99))

    def forward(self, x):
        x = self.model(x)
        return self.head_conv(x)


class CSM(nn.Module):
    """
    Implementation for the proposed Cross-Scale Mixing.
    """

    def __init__(self, channels, conv3_out_channels):
        super(CSM, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, conv3_out_channels, 3, 1, 1)

        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.conv3.parameters():
            param.requires_grad = False

        self.apply(kaiming_init)

    def forward(self, high_res, low_res=None):
        batch, channels, width, height = high_res.size()
        if low_res is None:
            # high_res_flatten = rearrange(high_res, "b c h w -> b c (h w)")
            high_res_flatten = high_res.view(batch, channels, width * height)
            high_res = self.conv1(high_res_flatten)
            high_res = high_res.view(batch, channels, width, height)
            high_res = self.conv3(high_res)
            high_res = F.interpolate(high_res, scale_factor=2., mode="bilinear")
            return high_res
        else:
            high_res_flatten = high_res.view(batch, channels, width * height)
            high_res = self.conv1(high_res_flatten)
            high_res = high_res.view(batch, channels, width, height)
            high_res = torch.add(high_res, low_res)
            high_res = self.conv3(high_res)
            high_res = F.interpolate(high_res, scale_factor=2., mode="bilinear")
            return high_res


class Discriminator:
    def __init__(self, img_size=128):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.efficient_net = EfficientNetLite(widthi_multiplier=1.0, depth_multiplier=1.1, num_classes=1000, drop_connect_rate=0.2, dropout_rate=0.2)
        self.efficient_net.eval()
        load_checkpoint(self.efficient_net, torch.load(os.path.join(os.path.dirname(__file__), 'efficientnet_lite1.pth')))

        feature_sizes = self.get_feature_channels()
        self.csms = nn.ModuleList([
            CSM(feature_sizes[3], feature_sizes[2]),
            CSM(feature_sizes[2], feature_sizes[1]),
            CSM(feature_sizes[1], feature_sizes[0]),
            CSM(feature_sizes[0], feature_sizes[0]),
        ])

        self.discs = nn.ModuleList([
           MultiScaleDiscriminator(feature_sizes[0], 1),
           MultiScaleDiscriminator(feature_sizes[1], 2),
           MultiScaleDiscriminator(feature_sizes[2], 3),
           MultiScaleDiscriminator(feature_sizes[3], 4),
        ][::-1])

    def load_state_dict(self, ckpt):
        load_checkpoint(self.efficient_net, ckpt)

    def parameters(self):
        return list(self.csms.parameters()) + list(self.discs.parameters())

    def to(self, device):
        self.efficient_net.to(device)
        self.csms.to(device)
        self.discs.to(device)
        return self
    def apply(self, weight_init):
        self.efficient_net.apply(weight_init)
        self.csms.apply(weight_init)
        self.discs.apply(weight_init)

    def zero_grad(self):
        self.csms.zero_grad()
        self.discs.zero_grad()

    def state_dict(self):
        return {"csms": self.csms.state_dict(), "discs": self.discs.state_dict()}
    def csm_forward(self, features):
        features = features[::-1]
        csm_features = []
        for i, csm in enumerate(self.csms):
            if i == 0:
                d = csm(features[i])
                csm_features.append(d)
            else:
                d = csm(features[i], d)
                csm_features.append(d)
        return features

    def __call__(self, x):
        features = self.efficient_net(x)
        features = self.csm_forward(features)
        dics_maps = []
        for feature, disc in zip(features, self.discs):
            disc_map = disc(feature).sum(1) # bx1x4x4
            dics_maps.append(disc_map)  # Cx4x4

        return torch.stack(dics_maps, dim=1)

    def get_feature_channels(self):
        sample = torch.randn(1, 3, self.img_size, self.img_size)
        features = self.efficient_net(sample)
        return [f.shape[1] for f in features]

if __name__ == '__main__':
    efficient_net = EfficientNetLite(widthi_multiplier=1.0, depth_multiplier=1.1, num_classes=1000, drop_connect_rate=0.2, dropout_rate=0.2)
    x = torch.ones((1,3,128,128))
    # print([ x.shape for x in efficient_net(x)])
    print([x.shape for x in efficient_net(x)])
