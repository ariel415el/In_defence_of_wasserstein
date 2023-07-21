import random

import torch
from torch import nn


# class Generator(nn.Module):
#     def __init__(self, z_dim, output_dim=64, channels=3, n_linears=16):
#         super(Generator, self).__init__()
#         self.linears = nn.ModuleList(
#             nn.Linear(z_dim, channels * output_dim ** 2) for i in range(n_linears)
#         )
#         self.channels = channels
#         self.output_dim = output_dim
#         self.n_linears = n_linears
#
#     def forward(self, z):
#         c = random.randint(0, self.n_linears-1)
#         return self.linears[c](z).reshape(-1, self.channels, self.output_dim, self.output_dim)

class Generator(nn.Module):
    def __init__(self, z_dim, output_dim=64, channels=3, n_linears=16):
        super(Generator, self).__init__()
        self.model = nn.Linear(z_dim, channels * output_dim ** 2 * n_linears)
        self.channels = channels
        self.output_dim = output_dim
        self.n_linears = n_linears

    def forward(self, z):
        c = torch.randint(0, self.n_linears-1, (z.shape[0], ), dtype=torch.long)
        output = self.model(z).reshape(-1, self.n_linears, self.channels, self.output_dim, self.output_dim)
        return output[torch.arange(len(c)), c]


if __name__ == '__main__':
    netD = Generator(64, 64)
    print(netD(torch.ones(12,64)).shape)