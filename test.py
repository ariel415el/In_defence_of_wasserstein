import argparse
import json
import os

import torch

from models import get_models
from scripts.experiment_utils import find_last_file
from tests.compute_metrics import compute_metrics
from tests.generate_images import generate_images
from tests.interpolate import interpolate
from tests.latent_inversion import inverse_image
from tests.test_data_NNs import find_nns, find_patch_nns, find_nns_percept
# from tests.latent_inversion import inverse_image
from tests.test_utils import get_data
from utils.train_utils import Prior


def load_pretrained_models(args, ckpt_path, device):
    netG, netD = get_models(args, device)

    weights = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(weights['netG'])
    netG.to(device)
    netG.eval()

    netD.load_state_dict(weights['netD'])
    netD.to(device)
    netD.eval()
    prior = Prior(args.z_prior, args.z_dim)
    prior.z = weights['prior']

    return netG, netD, prior


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help="Path to trained model dir")
    parser.add_argument('--ckpt_name', default=None, type=str)
    parser.add_argument('--device', default="cpu")
    args = parser.parse_args()
    model_dir = args.model_dir
    device = torch.device(args.device)

    if args.ckpt_name is None:
        ckpt_path = find_last_file(f'{model_dir}/models', ext='.pth')
    else:
        ckpt_path = f'{model_dir}/models/{args.ckpt_name}.pth'  # path to the checkpoint
    outputs_dir = f'{model_dir}/test_outputs'
    os.makedirs(outputs_dir, exist_ok=True)

    args = json.load(open(os.path.join(model_dir, "args.txt")))
    z_dim = args['z_dim']
    data_root = args['data_path']
    print("Loading models", end='...')
    netG, netD, prior = load_pretrained_models(argparse.Namespace(**args), ckpt_path, device)
    print("Done")

    # No data tests
    generate_images(netG, prior, outputs_dir, device)
    # find_mode_collapses(netG, netD, z_dim, outputs_dir, device)
    # interpolate(netG, z_dim, n_zs=15, seconds=60, fps=30, outputs_dir=outputs_dir, device=device)

    # Partial data tests
    # data = get_data(args['data_path'], args['im_size'], args['center_crop'], args['gray_scale'], limit_data=9).to(device)
    # saliency_maps(netG, netD, z_dim, data, outputs_dir, device)
    # test_range(netG, netD, z_dim, data, outputs_dir, device)
    # inverse_image(netG, z_dim, data, outputs_dir=outputs_dir)

    # Full data tests
    data = get_data(args['data_path'], args['im_size'], args['center_crop'], args['gray_scale'], limit_data=args['limit_data'])
    netG = netG.cpu()

    # compute_metrics(netG, prior, data, device)
    # Nearest neighbor visualizations
    fake_images = netG(prior.sample(16).cpu())

    find_nns(fake_images, data, outputs_dir=outputs_dir)
    # find_nns_percept(fake_images, data, outputs_dir, device, layer_idx=18)
    # find_nns_percept(fake_images, data, outputs_dir, device, netD, layer_idx=3)
    # find_patch_nns(fake_images, data, patch_size=32, stride=3, search_margin=21, outputs_dir=outputs_dir, n_centers=5, b=256, metric_name='L2')
    find_patch_nns(fake_images, data, patch_size=16, stride=1, search_margin=12, outputs_dir=outputs_dir, n_centers=12, b=1024, metric_name='edge', device=device)
    # find_patch_nns(fake_images, data, patch_size=16, stride=1, search_margin=12, outputs_dir=outputs_dir, n_centers=12, b=1024, metric_name='mean', device=device)
    # find_patch_nns(fake_images, data, patch_size=32, stride=7, search_margin=21, outputs_dir=outputs_dir, n_centers=5, b=16, metric_name='vgg', device=torch.device("cuda:0"))
    # find_patch_nns(fake_images, data, patch_size=22, search_margin=4, outputs_dir=outputs_dir, n_centers=4)
    # find_patch_nns(fake_images, data, patch_size=12, search_margin=2, outputs_dir=outputs_dir, n_centers=4)
