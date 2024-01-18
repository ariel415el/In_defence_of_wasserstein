import os
from math import sqrt

import torch
from torchvision.utils import save_image

from utils.metrics import get_batche_slices


def compose_experiment_name(args):
    return f"{os.path.basename(args.data_path)}_I-{args.im_size}x{args.im_size}_G-{args.gen_arch}_D-{args.disc_arch}" \
                f"{'_GS' if args.gray_scale else ''}{f'_CC-{args.center_crop}' if args.center_crop else ''}" \
                f"_L-{args.loss_function}_Z-{args.z_dim}x{args.z_prior}_B-{args.r_bs}-{args.f_bs}"


def parse_classnames_and_kwargs(string, kwargs=None):
    """Import a class and create an instance with kwargs from strings in format
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


def batch_generation(netG, prior, n, b, device, org_device):
    netG.to(device)
    fake_data = []
    if "const" in prior.prior_type: # generate images for all 'm' zs
        slices = get_batche_slices(len(prior.z), b)
        for slice in slices:
            zs = prior.z[slice].to(device)
            fake_data.append(netG(zs))
    else:  # Generate 'n' images for random zs in batches of size b
        slices = get_batche_slices(n, b)
        for slice in slices:
            z = prior.sample(len(slice)).to(device)
            fake_data.append(netG(z))
    fake_data = torch.cat(fake_data)
    netG.to(org_device)
    return fake_data
