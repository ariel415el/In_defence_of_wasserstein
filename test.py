import argparse
import json
import os

import torch

from models import get_models
from tests.generate_images import generate_images
from tests.interpolate import interpolate
from tests.test_data_NNs import find_nns, find_patch_nns
from tests.test_emd import test_emd
# from tests.latent_inversion import inverse_image
from tests.test_mode_collapse import find_mode_collapses
from tests.test_utils import get_data
from tests.test_discriminator import saliency_maps, test_range


def load_pretrained_models(args, ckpt_path, device):
    netG, netD = get_models(args, device)

    weights = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(weights['netG'])
    netG.to(device)
    netG.eval()

    netD.load_state_dict(weights['netD'])
    netD.to(device)
    netD.eval()

    return netG, netD


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help="Path to trained model dir")
    parser.add_argument('--ckpt_name', default='last', type=str)
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
    netG, netD = load_pretrained_models(argparse.Namespace(**args), ckpt_path, device)
    print("Done")

    # No data tests
    generate_images(netG, z_dim, outputs_dir, device)
    # find_mode_collapses(netG, netD, z_dim, outputs_dir, device)
    # interpolate(netG, z_dim, n_zs=15, steps=25, outputs_dir=outputs_dir, device=device)

    # Partial data tests
    # data = get_data(args['data_path'], args['im_size'], args['center_crop'], limit_data=9).to(device)
    # saliency_maps(netG, netD, z_dim, data, outputs_dir, device)
    # test_range(netG, netD, z_dim, data, outputs_dir, device)
    # inverse_image(netG, z_dim, data, outputs_dir=outputs_dir, device=device)

    # Full data tests
    data = get_data(args['data_path'], args['im_size'], args['center_crop'], args['gray_scale'], limit_data=args['limit_data']).to(device)
    # test_emd(netG, z_dim, data, outputs_dir=outputs_dir, device=device)
    find_nns(netG, z_dim, data, outputs_dir=outputs_dir, device=device)
    # find_patch_nns(netG, z_dim, data, patch_size=64, stride=1, search_margin=8, outputs_dir=outputs_dir, device=device)
    # find_patch_nns(netG, z_dim, data, patch_size=48, stride=24, search_margin=8, outputs_dir=outputs_dir, device=device)
    find_patch_nns(netG, z_dim, data, patch_size=24, stride=12, search_margin=1, outputs_dir=outputs_dir, device=device)


# import os
# for dname in os.listdir("outputs/GANs"):
#     os.system(f"python3 test.py outputs/GANs/{dname}")