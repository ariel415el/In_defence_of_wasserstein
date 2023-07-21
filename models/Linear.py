from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim=64, channels=3):
        super(Generator, self).__init__()
        self.model = nn.Linear(z_dim, channels*output_dim**2)
        self.channels = channels
        self.output_dim = output_dim

    def forward(self, z):
        return self.model(z).reshape(-1, self.channels, self.output_dim, self.output_dim)


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, **kwargs):
        super(Discriminator, self).__init__()
        self.model = nn.Linear(3*input_dim**2, 1)

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat).view(img.size(0))

        return validity
