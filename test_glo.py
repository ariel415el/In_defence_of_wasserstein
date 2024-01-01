import argparse
import json
import os

import torch

from models import get_generator
from train_glo import LatentCodesDict
from utils.common import dump_images


def load_pretrained_models(args, ckpt_path, device):
    netG = get_generator(args, device)

    weights = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(weights['netG'])
    netG.to(device)
    netG.eval()
    latent_codes_weights = weights['latent_codes']
    latent_codes = LatentCodesDict(args.z_dim, len(latent_codes_weights['emb.weight'])).to(device)
    latent_codes.load_state_dict(latent_codes_weights)
    latent_codes.to(device)
    latent_codes.eval()

    return netG, latent_codes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help="Path to trained model dir")
    parser.add_argument('--ckpt_name', default='last', type=str)
    parser.add_argument('--noise_sigma', default=0.1, type=float)
    parser.add_argument('--device', default="cpu")
    args = parser.parse_args()
    model_dir = args.model_dir
    device = torch.device(args.device)

    ckpt_path = f'{model_dir}/models/{args.ckpt_name}.pth'  # path to the checkpoint
    outputs_dir = f'{model_dir}/test_outputs'
    os.makedirs(outputs_dir, exist_ok=True)

    model_args = json.load(open(os.path.join(model_dir, "args.txt")))

    z_dim = model_args['z_dim']
    data_root = model_args['data_path']
    print("Loading models", end='...')
    netG, latent_codes = load_pretrained_models(argparse.Namespace(**model_args), ckpt_path, device)
    print("Done")

    zs = latent_codes(torch.arange(64))
    dump_images(netG(zs), f'{outputs_dir}/reconstructions.png')
    dump_images(netG(zs + args.noise_sigma*torch.rand_like(zs)), f'{outputs_dir}/noisy_recs.png')

    os.makedirs(f'{outputs_dir}/interpolations', exist_ok=True)
    i = 0
    n_steps = 10
    frame = 0
    f = 1 / (n_steps - 1)
    for i in range(len(zs)-1):
        for a in range(n_steps):
            noise = zs[i+1] * a * f + zs[i] * (1 - a * f)
            fake_imgs = netG(noise.unsqueeze(0))
            dump_images(fake_imgs, f'{outputs_dir}/interpolations/fakes-{frame}.png')
            frame += 1
