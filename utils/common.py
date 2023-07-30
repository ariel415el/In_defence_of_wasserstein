import sys
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
    save_image((batch + 1)/2, fname, nrow=nrow, normalize=False, pad_value=1, scale_each=True)


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
