import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim, n=64, init_mode='noise'):
        super(Generator, self).__init__()
        n = int(n)
        if init_mode == "noise":
            images = torch.randn(n, 3 ,output_dim, output_dim) * 0.5
        elif init_mode == "ones":
            images = torch.ones(n, 3 ,output_dim, output_dim)
        else:
            raise ValueError("Bad init mode")
        self.images = nn.Parameter(images, requires_grad=True)
        # self.clip()

    def forward(self, input):
        return torch.tanh(self.images)