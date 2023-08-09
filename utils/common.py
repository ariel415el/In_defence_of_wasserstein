from copy import deepcopy
from math import sqrt

import torch
from torchvision.utils import save_image


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def parse_classnames_and_kwargs(string, kwargs=None):
    """Import a class and and create an instance with kwargs from strings in format
    '<class_name>_<kwarg_1_name>=<kwarg_1_value>_<kwarg_2_name>=<kwarg_2_value>"""
    name_and_args = string.split('-')
    class_name = name_and_args[0]
    if kwargs is None:
        kwargs = dict()
    for arg in name_and_args[1:]:
        name, value = arg.split("=")
        kwargs[name] = value
    return class_name, kwargs


def dump_images(batch, fname):
    nrow = int(sqrt(len(batch)))
    # save_image((batch + 1)/2, fname, nrow=nrow, normalize=False, pad_value=1, scale_each=True)
    save_image(batch, fname, nrow=nrow, normalize=True, pad_value=1, scale_each=True)


class Prior:
    def __init__(self, prior_type, z_dim):
        self.prior_type = prior_type
        self.z_dim = z_dim
        if "const" in self.prior_type:
            self.z = None
            self.b = int(self.prior_type.split("=")[1])

    def sample(self, b):
        if "const" in self.prior_type:
            if self.z is None:
                self.z = torch.randn((self.b, self.z_dim))
            if b != self.b:
                z = self.z[torch.randint(self.b, (b,))]
            else:
                z = self.z
        elif self.prior_type == "binary":
            z = torch.sign(torch.randn((b, self.z_dim)))
        elif self.prior_type == "uniform":
            z = torch.rand((b, self.z_dim))
        else:
            z = torch.randn((b, self.z_dim))
        return z



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
