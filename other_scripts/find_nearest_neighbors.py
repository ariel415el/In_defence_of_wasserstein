import argparse
import json
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from experiment_utils import get_data, find_last_file

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models import get_models
from utils.train_utils import Prior


def to_np(img):
    if img.shape[0] == 1:
        img = img.repeat(3,1,1)
    img = img.add_(1).div(2).mul(255).clamp_(0, 255)
    if len(img.shape) == 3:
        img = img.permute(1, 2, 0)
    return img.to("cpu", torch.uint8).cpu().numpy()


def find_nns(fake_images, data, outputs_dir,s=4):
    os.makedirs(f'{outputs_dir}', exist_ok=True)
    n = len(fake_images)
    with torch.no_grad():
        fig, axes = plt.subplots(2, n, figsize=(s * n, s * 2))

        for i in range(len(fake_images)):
            fake_image = fake_images[i]
            # dists = dists_mat[i]
            dists = [(fake_image - data[j]).pow(2).sum().item() for j in range(len(data))]
            nn_index = np.argsort(dists)[0]
            nn = data[nn_index]
            axes[0, i].imshow(to_np(fake_image))
            axes[0, i].axis('off')
            axes[0, i].set_ylabel('Generate image')
            axes[1, i].imshow(to_np(nn))
            axes[1, i].axis('off')
            axes[1, i].set_ylabel('Nearest neighbor')
            axes[1, i].set_title(f"NN L2: {dists[nn_index]:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f"NNs.png"))


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
    parser.add_argument('--n_samples', default=6, type=int)
    script_args = parser.parse_args()
    model_dir = script_args.model_dir
    device = torch.device("cpu")

    if script_args.ckpt_name is None:
        ckpt_path = find_last_file(f'{model_dir}/models', ext='.pth')
    else:
        ckpt_path = f'{model_dir}/models/{script_args.ckpt_name}.pth'  # path to the checkpoint
    outputs_dir = f'{model_dir}/test_outputs'
    os.makedirs(outputs_dir, exist_ok=True)

    args = json.load(open(os.path.join(model_dir, "args.txt")))

    z_dim = args['z_dim']
    data_root = args['data_path']
    print("Loading models", end='...')
    netG, netD, prior = load_pretrained_models(argparse.Namespace(**args), ckpt_path, device)
    print("Done")

    # Full data tests
    data = get_data(args['data_path'], args['im_size'], args['center_crop'], args['gray_scale'], limit_data=args['limit_data'])

    fake_images = netG(prior.sample(script_args.n_samples).to(device))
    find_nns(fake_images, data, outputs_dir=outputs_dir)
