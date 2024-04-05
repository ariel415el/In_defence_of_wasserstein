import os
from math import sqrt

import torch
from torchvision.utils import save_image


def compose_experiment_name(args):
    return f"{os.path.basename(args.data_path)}_I-{args.im_size}x{args.im_size}" \
                f"{'_GS' if args.gray_scale else ''}{f'_CC-{args.center_crop}' if args.center_crop else ''}" \
                f"_Z-{args.z_dim}_FZ-{args.force_norm_every}_ZS-{args.noise_sigma}_G-{args.gen_arch}_B-{args.r_bs}-{args.f_bs}"


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


def batch_generation(latent_codes, netG, n, b, device):
    n_batches = n // b
    fake_data = []
    from tqdm import tqdm
    for i in tqdm(range(n_batches)):
        zs = latent_codes(torch.arange(i*b, (i+1)*b).to(device))
        fake_data.append(netG(zs))
    if n_batches * b < n:
        zs = latent_codes(torch.arange(n-n_batches * b, n).to(device))
        fake_data.append(netG(zs))
    fake_data = torch.cat(fake_data)
    return fake_data