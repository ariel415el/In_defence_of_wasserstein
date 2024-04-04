import os
from math import sqrt

import torch
from torchvision.utils import save_image
from tqdm import tqdm

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


def dump_images(batch, fname, separate_images=False):
    if separate_images:
        dir_name = os.path.splitext(fname)[0]
        os.makedirs(dir_name, exist_ok=True)
        batch = batch.detach()
        batch = batch - batch.min()
        batch = batch / batch.max()
        for i, img in enumerate(range(len(batch))):
            save_image(batch[i], os.path.join(dir_name, f'{i}.png'), normalize=False, pad_value=1, scale_each=True)
    else:
        nrow = int(sqrt(len(batch)))
        save_image(batch, fname, nrow=nrow, normalize=True, pad_value=1, scale_each=True)


def batch_generation(netG, prior, n, b, inference_device, verbose=False):
    """
    Generate images in batches of size b on 'inference_device'
    # for a discrete prior generate images for all 'm' zs
    # for continous priors Generate 'n' images
    """
    org_device = next(netG.parameters()).device
    netG.to(inference_device)
    fake_data = []
    is_const_prior = "const" in prior.prior_type
    slices = get_batche_slices(len(prior.z) if is_const_prior else n, b)
    if verbose:
        slices = tqdm(slices)
    for slice in slices:
        if is_const_prior:
            zs = prior.z[slice].to(inference_device)
        else:
            zs = prior.sample(len(slice)).to(inference_device)
        fake_data.append(netG(zs).cpu())
    fake_data = torch.cat(fake_data)
    netG.to(org_device)
    return fake_data
