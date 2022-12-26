from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim=64):
        super(Generator, self).__init__()
        self.output_dim = output_dim
        nf = 128 if output_dim == 64 else 64
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(z_dim, nf, normalize=False),
            *block(nf, 2*nf),
            *block(2*nf, 4*nf),
            *block(4*nf, 8*nf),
            nn.Linear(8*nf, 3*output_dim**2),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, self.output_dim, self.output_dim)
        return img


class Discriminator(nn.Module):
    def __init__(self, in_dim=64):
        super(Discriminator, self).__init__()
        nf = 256 if in_dim == 64 else 512
        self.model = nn.Sequential(
            nn.Linear(3*in_dim**2, 2*nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2*nf, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nf, 1),
            # nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat).view(img.size(0))

        return validity