from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_dim=64,
                 p: int=3,
                 hdim: int =128,
                 depth: int =4,
                 s: int =2,
                 channels=3,
                 normalize='none',
                 win=3):
        super().__init__()
        p = int(p)
        s = int(s)
        win = int(win)
        depth = int(depth)
        hdim = int(hdim)
        layers = [nn.Conv2d(channels, hdim, p, stride=s, padding=0), nn.LeakyReLU(0.2)]
        for i in range(depth):
            layers.append(nn.Conv2d(hdim, hdim, 1))

            if normalize == "bn":
                layers.append(nn.BatchNorm2d(hdim))
            elif normalize == "in":
                layers.append(nn.InstanceNorm2d(hdim))

            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.AvgPool2d(kernel_size=win, stride=s))
        self.FC = nn.Linear(in_features=hdim*15**2, out_features=1)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        features = self.layers(x)
        x = self.FC(features.reshape(features.shape[0], -1)).view(len(x))
        return x


if __name__ == '__main__':
    import torch
    netD = Discriminator(p=3, s=2, win=3)
    print(netD(torch.ones(5,3,64,64)).shape)