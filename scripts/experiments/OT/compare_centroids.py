import argparse
import json
import os
import sys

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from losses import get_loss_function
from models import get_models
from utils.common import batch_generation, dump_images
from utils.train_utils import Prior
from scripts.experiments.experiment_utils import batch_to_image

from utils.data import get_transforms
from tqdm import tqdm
from PIL import Image


def get_data(data_root, im_size, center_crop, gray_scale, limit_data=None, ending=None):
    """Load entire dataset to memory as a single batch"""
    T = get_transforms(im_size, center_crop, gray_scale)

    images = []
    img_names = sorted(os.listdir(data_root))
    # shuffle(img_names)
    if ending is not None:
        img_names = [n for n in img_names if n.endswith(ending)]
    if limit_data is not None:
        img_names = img_names[:limit_data]
    for fname in tqdm(img_names):
        im = Image.open(os.path.join(data_root, fname))
        im = T(im)
        images += [im]

    data = torch.stack(images)

    return data


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


def plot(metric_names, named_batches, named_refs, plot_path):
    fig, axes = plt.subplots(1 + len(named_refs), len(named_batches)+1, figsize=(4*(len(named_batches)+1), 4*2))

    axes[0, 0].axis('off')
    for i, (name, ref_batch) in enumerate(named_refs.items()):
        axes[1+i, 0].imshow(batch_to_image(ref_batch, im_size, c, n=9))
        axes[1+i, 0].set_title(name, size=10)
        axes[1+i, 0].axis('off')
        for j, (name, batch) in enumerate(named_batches.items()):
            title = '\n\n'
            for metric_name in metric_names:
                print(name, metric_name)#, ref_batch.shape, batch.shape)
                metric = get_loss_function(metric_name)
                title += f"{metric_name}: {metric(ref_batch, batch):.5f}\n"
            axes[0, j+1].imshow(batch_to_image(batch, im_size, c, n=9))
            axes[0, j+1].axis('off')
            axes[0, j+1].set_title(name)
            axes[1+i, j+1].set_title(title[:-2], size=10)
            axes[1+i, j+1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.clf()


def generate_from_ckpt(model_dir, n=128):
    ckpt = f'{model_dir}/models/last.pth'
    args = json.load(open(os.path.join(model_dir, "args.txt")))
    print(ckpt)
    netG, netD, prior = load_pretrained_models(argparse.Namespace(**args), ckpt, device)
    gen_centroids = batch_generation(netG, prior, n, n, device)
    return gen_centroids


def main(output_name):
    named_batches = {
        'centroids': get_data(centroids_dir, im_size,  False, gray_scale, None).to(device),
        "samples": get_data(data_dir, im_size, False, gray_scale, None, ending='-0.png').to(device),
    }

    # named_batches["Pixels_DC_centroids"]= generate_from_ckpt('outputs/GANs/images_1_I-64x64_G-Pixels-n=128_D-DCGAN_L-WGANLoss_Z-64xconst=128_B-128-128_test')
    # named_batches["FC_DC_centroids"]= generate_from_ckpt('outputs/GANs/FC_DC')

    data = get_data(data_dir, im_size, False, gray_scale, None).to(device)
    other_data = get_data(other_data_dir, im_size, center_crop, gray_scale,  limit_data=len(data)).to(device)
    named_refs = {
        'data': data,
        'other_data': other_data
    }

    metric_names = ['MiniBatchLoss-dist=w1',
                    'MiniBatchLoss-dist=swd',
                    # 'MiniBatchPatchLoss-dist=swd-p=16-s=4',
                    # 'MiniBatchPatchLoss-dist=swd-p=4-s=4',
                    'MiniBatchPatchLoss-dist=swd-p=8-s=1',
                    # 'MiniBatchPatchLoss-dist=w1-p=8-s=4-n_samples=10000',
    ]

    plot(metric_names, named_batches, named_refs, f"{output_root}/{output_name}.png")

if __name__ == '__main__':
    device = torch.device("cpu")
    output_root = os.path.join(os.path.dirname(__file__), "outputs", "Compare_centroids")

    # im_size=64
    # c=3
    # center_crop=90
    # gray_scale=False
    # centroids_dir = '/mnt/storage_ssd/datasets/FFHQ/FFHQ_centroids/shifted_crops/centroids'
    # data_dir = '/mnt/storage_ssd/datasets/FFHQ/FFHQ_centroids/shifted_crops/shifted_crops'
    # other_data_dir = '/mnt/storage_ssd/datasets/FFHQ/FFHQ'
    # main("FFHQ_shifts")


    im_size=64
    c=3
    center_crop=None
    gray_scale=False
    centroids_dir = '/mnt/storage_ssd/datasets/FFHQ/FFHQ_centroids/floating_images/centroids'
    data_dir = '/mnt/storage_ssd/datasets/FFHQ/FFHQ_centroids/floating_images/floating_images'
    other_data_dir = '/mnt/storage_ssd/datasets/FFHQ/FFHQ_centroids/floating_images/reference'
    main("FFHQ_floating")

    # im_size=40
    # c=1
    # center_crop=None
    # gray_scale=True
    # centroids_dir = '/mnt/storage_ssd/datasets/MNIST/MNIST_centroids/floating_images/centroids'
    # data_dir = '/mnt/storage_ssd/datasets/MNIST/MNIST_centroids/floating_images/floating_images'
    # other_data_dir = '/mnt/storage_ssd/datasets/MNIST/MNIST_centroids/floating_images/reference'
    # main("MNIST_floating")

    # im_size=64
    # c=3
    # center_crop=450
    # gray_scale=False
    # centroids_dir = '/mnt/storage_ssd/datasets/afhq/train/cat_centroids/shifted_crops/centroids'
    # data_dir = '/mnt/storage_ssd/datasets/afhq/train/cat_centroids/shifted_crops/shifted_crops'
    # other_data_dir = '/mnt/storage_ssd/datasets/afhq/train/cat'
    # main("afhq_shifted")