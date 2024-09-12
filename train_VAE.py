
from time import time

import torch

import os
import sys

from torch import optim

from models import DCGAN, VQVAE, EDM, FC

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

    encoder = DCGAN.Discriminator(input_dim=args.im_size, channels=c, num_outputs=2*args.z_dim, nf=256)
    # encoder = FC.Discriminator(input_dim=args.im_size, channels=c, num_outputs=2*args.z_dim, nf=64, normalize='none')
    # encoder = VQVAE.Encoder(args.z_dim * 2)
    # encoder = EDM.DhariwalUNet(args.im_size, 3, 5, model_channels=64)
    optimizerE = optim.Adam(encoder.parameters(), lr=args.lrG, betas=(0.5, 0.9))
    encoder.train().to(device)

    decoder = DCGAN.Generator(output_dim=args.im_size, z_dim=args.z_dim, channels=c, nf=256)
    # decoder = FC.Generator(output_dim=args.im_size, z_dim=args.z_dim, channels=c, nf=64, normalize='none')
    # decoder = VQVAE.Decoder(args.z_dim)
    # decoder = EDM.DhariwalUNet(args.im_size, 5, 3, model_channels=64)
    optimizerD = optim.Adam(decoder.parameters(), lr=args.lrD, betas=(0.5, 0.9))
    decoder.train().to(device)



    start_iteration = 0
    return encoder, decoder, optimizerD, optimizerE, start_iteration

def train_AE(args):
    logger = WandbLogger(args, plots_image_folder) if args.wandb else PLTLogger(plots_image_folder)

    encoder, decoder, optimizerD, optimizerE, start_iteration = get_models_and_optimizers(args)
    noise_coeff = 0
    debug_fixed_reals = next(iter(train_loader)).to(device)
    debug_fixed_reals += noise_coeff * torch.randn_like(debug_fixed_reals).to(device)

    loss_function = torch.nn.MSELoss()

    start = time()
    iteration = start_iteration
    while iteration < args.n_iterations:
        for real_images in train_loader:
            real_images = real_images.to(device)
            mu, log_var = torch.split(encoder(real_images), args.z_dim, dim=1)

            z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
            x_hat = decoder(z)

            # noise = + noise_coeff * torch.randn_like(real_images).to(device)
            # recons = decoder(encoder(real_images + noise))
            rec_loss = loss_function(real_images, x_hat)
            kl_loss = - 0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = rec_loss + 0.0001*kl_loss

            logger.log({'loss': loss.item(), 'rec_loss': rec_loss.item(), 'kl_loss': kl_loss.item()}, iteration)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizerD.step()
            optimizerE.step()

            if iteration % 100 == 0:
                it_sec = max(1, iteration - start_iteration) / (time() - start)
                print(f"Iteration: {iteration}: it/sec: {it_sec:.1f}")
                logger.plot()

            if iteration % args.log_freq == 0:
                evaluate(encoder, decoder, debug_fixed_reals, saved_image_folder, iteration)

                save_model(encoder, decoder, optimizerE, optimizerD, saved_model_folder, iteration, args)

            iteration += 1


def evaluate(encoder, decoder, debug_fixed_reals, saved_image_folder, iteration):
    # encoder.eval()
    # decoder.eval()
    start = time()
    with torch.no_grad():
        mu, log_var = torch.split(encoder(debug_fixed_reals), args.z_dim, dim=1)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        recons = decoder(z)
        dump_images(recons,  f'{saved_image_folder}/recons_{iteration}.png')

        z = torch.randn_like(z)
        recons = decoder(z)
        dump_images(recons,  f'{saved_image_folder}/samples_{iteration}.png')
        if iteration == 0:
            dump_images(debug_fixed_reals, f'{saved_image_folder}/debug_fixed_reals.png')

    decoder.train()
    encoder.train()

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