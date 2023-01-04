import os

import torch
from tqdm import tqdm
from PIL import Image

from models import get_discriminator
from utils.data import get_transforms


def get_data(data_root, im_size, center_crop, limit_data=None):
    """Load entire dataset to memory as a single batch"""
    T = get_transforms(im_size, center_crop)

    images = []
    print("Loading data to memory to find NNs")
    img_names = os.listdir(data_root)
    if limit_data is not None:
        img_names = img_names[:limit_data]
    for fname in tqdm(img_names):
        im = Image.open(os.path.join(data_root, fname))
        im = T(im)
        images += [im]

    return torch.stack(images)


def load_pretrained_generator(args, ckpt_path, device):
    from models import get_generator
    G = get_generator(args['gen_arch'], args['im_size'], args['z_dim'])
    weights = torch.load(ckpt_path, map_location=device)['netG']
    # weights = {k.replace('module.', ''): v for k, v in weights.items()}
    # weights = {k.replace('network', 'init.init'): v for k, v in weights.items()}
    G.load_state_dict(weights)
    G.to(device)
    G.eval()
    return G


def load_pretrained_discriminator(args, ckpt_path, device):
    D = get_discriminator(args['disc_arch'], args['im_size'])
    weights = torch.load(ckpt_path, map_location=device)['netD']
    D.load_state_dict(weights)
    D.to(device)
    D.eval()
    return D