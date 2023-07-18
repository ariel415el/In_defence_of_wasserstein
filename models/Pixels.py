import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim, n=64, init_mode='noise', channels=3):
        super(Generator, self).__init__()
        self.n = int(n)
        if init_mode == "noise":
            images = torch.randn(self.n, channels ,output_dim, output_dim) * 0.5
        elif init_mode == "ones":
            images = torch.ones(self.n, channels ,output_dim, output_dim)
        elif init_mode == "zeros":
            images = torch.ones(self.n, channels, output_dim, output_dim)
        else:
            raise ValueError("Bad init mode")
        self.images = nn.Parameter(images, requires_grad=True)
        # self.clip()

    def forward(self, input):
        b = input.shape[0]
        if b != self.n:
            outputs = self.images[hash_vectors(input.detach(), n=self.n)]
            # outputs = self.images[torch.randperm(self.n)[:b]]
        else:
            outputs = self.images
        return torch.tanh(outputs)


def find_nth_decimal(x, first, size=2):
    "extract the nth to n+lth digits from a float"
    return (x * 10**(first-1) % 1 * 10**size).to(int)

def hash_vectors(x, n=1000):
    """maps a (b,d) float tensor into (d,) integer tensor in range (0,n-1) in a deterministic manner (using 3 of its decimals)"""
    l = 1
    while 10**l < n:
        l+=1
    decimals = find_nth_decimal(x.mean(1), first=3, size=l)
    return (decimals / 10 ** l * n).to(torch.long)

if __name__ == '__main__':
    netG = Generator(100, 64, n=1000)
    z = torch.randn((16,100))
    print(netG(z).shape)