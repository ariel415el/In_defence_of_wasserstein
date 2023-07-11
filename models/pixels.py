import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim, n=64, init_mode='noise'):
        super(Generator, self).__init__()
        self.n = int(n)
        if init_mode == "noise":
            images = torch.randn(self.n, 3 ,output_dim, output_dim) * 0.5
        elif init_mode == "ones":
            images = torch.ones(self.n, 3 ,output_dim, output_dim)
        elif init_mode == "zeros":
            images = torch.ones(self.n, 3, output_dim, output_dim)
        else:
            raise ValueError("Bad init mode")
        self.images = nn.Parameter(images, requires_grad=True)
        # self.clip()

    def forward(self, input):
        b = input.shape[0]
        if b < self.n:

            outputs = self.images[torch.randperm(self.n)[:b]]
        else:
            outputs = self.images
        return torch.tanh(outputs)