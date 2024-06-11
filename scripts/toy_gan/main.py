import argparse
import os

from time import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from data import NPDataset, generate_data
from toy_models import FCGenerator, PixelGenerator, Discriminator
from toy_utils import draw_points

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from losses import get_loss_function
from utils.train_utils import calc_gradient_penalty
from logger import PLTLogger


def train_GAN(args):
    logger = PLTLogger(args, saved_image_folder)
    debug_fixed_noise = torch.randn((args.eval_batch_size, args.z_dim)).to(device)
    debug_fixed_reals = torch.from_numpy(dataset.np_data).to(device).float()

    other_metrics = [
        get_loss_function("MiniBatchLoss-dist=w1"),
    ]

    loss_function = get_loss_function(args.loss_function)


    start = time()
    iteration = 0
    while iteration < args.n_iterations:
        for real_images in dataloader:
            real_images = real_images.to(device).float()
            b = real_images.size(0)

            noise = torch.randn((b, args.z_dim)).to(device)
            fake_images = netG(noise)

            # #####  1. train Discriminator #####
            if iteration % args.D_step_every == 0 and args.D_step_every > 0:
                Dloss, debug_Dlosses = loss_function.trainD(netD, real_images, fake_images)
                if args.gp_weight > 0:
                    gp, gradient_norm = calc_gradient_penalty(netD, real_images, fake_images)
                    debug_Dlosses['gradient_norm'] = gradient_norm
                    Dloss += args.gp_weight * gp
                netD.zero_grad()
                Dloss.backward()
                optimizerD.step()

                if args.weight_clipping is not None:
                    for p in netD.parameters():
                        p.data.clamp_(-args.weight_clipping, args.weight_clipping)

                logger.log(debug_Dlosses, step=iteration)

            noise = torch.randn((b, args.z_dim)).to(device)
            fake_images = netG(noise)

            # #####  2. train Generator #####
            if iteration % args.G_step_every == 0:
                Gloss, debug_Glosses = loss_function.trainG(netD, real_images, fake_images)
                netG.zero_grad()
                Gloss.backward()
                optimizerG.step()
                logger.log(debug_Glosses, step=iteration)

            if iteration % 100 == 0:
                it_sec = max(1, iteration) / (time() - start)
                print(f"Iteration: {iteration}: it/sec: {it_sec:.1f}")
                logger.plot()

            if iteration % args.log_freq == 0:
               evaluate(netG, netD, other_metrics, debug_fixed_noise, debug_fixed_reals, iteration, logger)

            iteration += 1


def evaluate(netG, netD, other_metrics, fixed_noise, fixed_reals, iteration, logger):
    netG.eval()
    netD.eval()
    start = time()
    with torch.no_grad():
        fixed_noise_fake_images = netG(fixed_noise)

        for metric in other_metrics:
            logger.log({
                f'{metric.name}_fixed_noise_gen_to_train': metric(fixed_noise_fake_images, fixed_reals),
            }, step=iteration)

        draw_points(fixed_noise_fake_images.detach().cpu().numpy(), f'{saved_image_folder}/{iteration}.png',
                    ref_data=dataset.np_data)

    netG.train()
    netD.train()
    print(f"Evaluation finished in {time() - start} seconds")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    args = argparse.Namespace()
    args.data_path = "scripts/toy_gan/Circular-points-(NC-8_NS=256_R=10_STD=0.3)"
    args.batch_size = 512
    args.eval_batch_size = 1024
    args.z_dim = 8
    args.lrG = 0.0001
    args.lrD = 0.0002
    args.n_iterations = 1000000
    args.num_workers = 0
    args.gp_weight = 0
    args.weight_clipping  = None
    args.spectral_normalization  = False
    args.D_step_every = 1
    args.G_step_every = 5
    args.log_freq = 1000
    args.tag = "test"
    args.loss_function = "WGANLoss"; args.gp_weight  = 10
    # args.loss_function = "CtransformLoss"
    # args.loss_function = "BatchEMD-dist=L1"; args.D_step_every = -1

    saved_image_folder = os.path.join("outputs", f"toyGAN-L-{args.loss_function}_D-{os.path.basename(args.data_path)}_{args.tag}")

    dataset = NPDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # netG = PixelGenerator(args.z_dim, n=args.batch_size, b=args.batch_size).to(device)
    netG = FCGenerator(args.z_dim, out_dim=2, depth=6, nf=1024).to(device)
    netD = Discriminator(input_dim=2, depth=6,  nf=1024).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(0.5, 0.9))

    train_GAN(args)

