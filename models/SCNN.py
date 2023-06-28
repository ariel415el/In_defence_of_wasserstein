from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_dim=64,
                 p: int=7,
                 hdim: int =128,
                 depth: int =4,
                 s: int =1,
                 normalize='none'):
        super().__init__()
        p = int(p)
        s = int(s)
        depth = int(depth)
        hdim = int(hdim)
        layers = [nn.Conv2d(3, hdim, p, stride=s, padding=0), nn.LeakyReLU(0.2)]
        for i in range(depth):
            layers.append(nn.Conv2d(hdim, hdim, 1))

            if normalize == "bn":
                layers.append(nn.BatchNorm2d(hdim))
            elif normalize == "in":
                layers.append(nn.InstanceNorm2d(hdim))

            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=hdim, out_features=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x).view(len(x))
        return x

