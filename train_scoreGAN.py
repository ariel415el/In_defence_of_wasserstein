
from time import time

import torch

import os
import sys

from torch import optim

from models import DCGAN, VQVAE

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.common import dump_images, compose_experiment_name
from utils.train_utils import parse_train_args
from losses import get_loss_function
from utils.data import get_dataloader
from utils.logger import get_dir, PLTLogger, WandbLogger


def save_model(encoder, decoder, optimizerE, optimizerD, saved_model_folder, iteration, args):
    fname = f"{saved_model_folder}/{'last' if not args.save_every else iteration}.pth"
    torch.save({"iteration": iteration,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                "optimizerE": optimizerE.state_dict(),
                "optimizerD": optimizerD.state_dict()
                },
               fname)

def get_models_and_optimizers(args):
    c = 1 if args.gray_scale else 3

    nets = {
        'G' : DCGAN.Generator(output_dim=args.im_size, z_dim=args.z_dim, channels=c, nf=64),
        'D' : DCGAN.Discriminator(input_dim=args.im_size, channels=c, num_outputs=args.z_dim, nf=64),
        'ER' : DCGAN.Generator(output_dim=args.im_size, z_dim=args.z_dim, channels=c, nf=64),
        'EF' : DCGAN.Generator(output_dim=args.im_size, z_dim=args.z_dim, channels=c, nf=64)
    }
    optimizers = dict()
    for k in  ['G', 'D', 'ER', 'EF']:
        nets[k].train().to(device)
        lr = 0.0001
        optimizers[k] = optim.Adam(nets[k].parameters(), lr=lr, betas=(0.5, 0.9))

    start_iteration = 0
    return nets, optimizers, start_iteration


def distribution_matching_loss(nets, x, noise_coeff):
    noisy_x = x + noise_coeff * torch.randn_like(x).to(device)

    with (torch.no_grad()):
        pred_fake_image = nets['EF'](nets['D'](noisy_x))
        pred_real_image = nets['ER'](nets['D'](noisy_x))

    weighting_factor = torch.abs(x - pred_real_image).mean(dim=[1, 2, 3], keepdim=True)  # Eqn. 8
    # print(weighting_factor)
    grad = (pred_fake_image - pred_real_image) / weighting_factor
    diff = (x - grad).detach()  # stop-gradient
    return 0.5 * torch.nn.functional.mse_loss(x, diff)


def train_AE(args):
    logger = WandbLogger(args, plots_image_folder) if args.wandb else PLTLogger(plots_image_folder)

    nets, optimizers, start_iteration = get_models_and_optimizers(args)
    noise_coeff = 0.25
    debug_fixed_reals = next(iter(train_loader)).to(device)
    debug_fixed_reals += noise_coeff * torch.randn_like(debug_fixed_reals).to(device)
    debug_fixed_noises = torch.randn(args.f_bs, args.z_dim).to(device)

    loss_function = torch.nn.MSELoss()

    start = time()
    iteration = start_iteration
    while iteration < args.n_iterations:
        for real_images in train_loader:
            real_images = real_images.to(device)
            fake_images = nets['G'](torch.randn(args.f_bs, args.z_dim).to(device))

            noise = noise_coeff * torch.randn_like(real_images).to(device)

            # train Real DAE
            recons = nets['ER'](nets['D'](real_images + noise))
            real_rec_loss = loss_function(real_images, recons)
            nets['ER'].zero_grad()
            nets['D'].zero_grad()
            real_rec_loss.backward()
            optimizers['ER'].step()
            optimizers['D'].step()

            if iteration % 1 == 0:
                # train Fake DAE
                recons = nets['EF'](nets['D'](fake_images + noise))
                fake_rec_loss = loss_function(fake_images, recons)
                nets['EF'].zero_grad()
                nets['D'].zero_grad()
                fake_rec_loss.backward()
                optimizers['ER'].step()
                optimizers['D'].step()

            if iteration % 5 == 0:
                # train Generator
                nets['G'].zero_grad()
                fake_images = nets['G'](torch.randn(args.f_bs, args.z_dim).to(device))
                dm_loss = distribution_matching_loss(nets, fake_images, noise_coeff)
                dm_loss.backward()
                optimizers['G'].step()

            logger.log({'real_rec_loss': real_rec_loss.item(),
                        'fake_rec_loss': fake_rec_loss.item(),
                        'dm_loss': dm_loss.item()
                        }, iteration)

            if iteration % 100 == 0:
                it_sec = max(1, iteration - start_iteration) / (time() - start)
                print(f"Iteration: {iteration}: it/sec: {it_sec:.1f}")
                logger.plot()

            if iteration % args.log_freq == 0:
                evaluate(nets, debug_fixed_noises, debug_fixed_reals, saved_image_folder, iteration, noise_coeff)

                # save_model(encoder, decoder, optimizerE, optimizerD, saved_model_folder, iteration, args)

            iteration += 1


def evaluate(nets, debug_fixed_noises, debug_fixed_reals, saved_image_folder, iteration, noise_coeff):
    for k in ['G', 'D', 'ER', 'EF']:
        nets[k].eval()

    start = time()
    with torch.no_grad():
        fake_images = nets['G'](debug_fixed_noises)
        dump_images(fake_images,  f'{saved_image_folder}/fakes_{iteration}.png')

        noise = noise_coeff * torch.randn_like(debug_fixed_reals).to(device)
        recons = nets['ER'](nets['D'](debug_fixed_reals + noise))
        dump_images(recons,  f'{saved_image_folder}/real_recons_{iteration}.png')

        recons = nets['EF'](nets['D'](debug_fixed_reals + noise))
        dump_images(recons,  f'{saved_image_folder}/fake_recons_{iteration}.png')

        if iteration == 0:
            dump_images(debug_fixed_reals, f'{saved_image_folder}/debug_fixed_reals.png')

    for k in ['G', 'D', 'ER', 'EF']:
        nets[k].train()

    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    args = parse_train_args()
    if args.train_name is None:
        args.train_name = compose_experiment_name(args)
    saved_model_folder, saved_image_folder, plots_image_folder = get_dir(args)

    device = torch.device(args.device)
    if args.device != 'cpu':
        print(f"Working on device: {torch.cuda.get_device_name(device)}")

    train_loader, _ = get_dataloader(args.data_path, args.im_size, args.r_bs, args.n_workers,
                                               val_percentage=0, gray_scale=args.gray_scale, center_crop=args.center_crop,
                                               load_to_memory=args.load_data_to_memory, limit_data=args.limit_data)


    train_AE(args)