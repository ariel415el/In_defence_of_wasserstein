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
