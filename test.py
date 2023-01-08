import argparse
import json
import os

import torch

from tests.interpolate import interpolate
from tests.test_data_NNs import find_nns, find_patch_nns, inverse_image
from tests.test_mode_collapse import find_mode_collapses
from tests.test_utils import load_pretrained_generator, load_pretrained_discriminator, get_data
from tests.visualize_discriminator import visualize_discriminator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help="Path to trained model dir")
    parser.add_argument('--ckpt_name', default='1000', type=str)
    parser.add_argument('--device', default="cpu")
    args = parser.parse_args()
    model_dir = args.model_dir
    device = torch.device(args.device)

    ckpt_path = f'{model_dir}/models/{args.ckpt_name}.pth'  # path to the checkpoint
    outputs_dir = f'{model_dir}/test_outputs'
    os.makedirs(outputs_dir, exist_ok=True)

    args = json.load(open(os.path.join(model_dir, "args.txt")))
    z_dim = args['z_dim']
    data_root = args['data_path']

    print("Loading models", end='...')
    netG = load_pretrained_generator(args, ckpt_path, device)
    netD = load_pretrained_discriminator(args, ckpt_path, device)
    print("Done")

    # No data tests
    # find_mode_collapses(netG, netD, z_dim, outputs_dir, device)
    # interpolate(netG, z_dim, n_zs=15, steps=25, outputs_dir=outputs_dir, device=device)

    # partial data tests
    data = get_data(args['data_path'], args['im_size'], args['center_crop'], limit_data=9)
    visualize_discriminator(netG, netD, z_dim, data, outputs_dir, device)
    inverse_image(netG, z_dim, data, outputs_dir=outputs_dir, device=device)

    # Full data tests
    data = get_data(args['data_path'], args['im_size'], args['center_crop'], limit_data=None)
    data = data.to(device)

    find_nns(netG, z_dim, data, outputs_dir=outputs_dir, device=device)
    find_patch_nns(netG, z_dim, data, patch_size=16, stride=4, search_margin=6, outputs_dir=outputs_dir, device=device)
    find_patch_nns(netG, z_dim, data, patch_size=24, stride=4, search_margin=6, outputs_dir=outputs_dir, device=device)

