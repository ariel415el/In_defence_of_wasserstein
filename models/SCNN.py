from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_dim=64,
                 ksize=7,
                 hdim=128,
                 n_local_layers=4,
                 strides=1,
                 bn=False):
        super().__init__()
        ksize = int(ksize)
        # layers = [nn.Conv2d(3, hdim, ksize, stride=strides, padding=ksize // 2)]
        layers = [nn.Conv2d(3, hdim, ksize, stride=strides, padding=0), nn.LeakyReLU(0.2)]
        for i in range(n_local_layers):
            layers.append(nn.Conv2d(hdim, hdim, 1))
            if bn:
                layers.append(nn.BatchNorm2d(hdim))
            else:
                layers.append(nn.InstanceNorm2d(hdim))
            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=hdim, out_features=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x).view(len(x))
        return x

