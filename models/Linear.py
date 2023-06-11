from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, **kwargs):
        super(Discriminator, self).__init__()
        self.model = nn.Linear(3*input_dim**2, 1)


    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat).view(img.size(0))

        return validity
