import argparse
import glob
import os.path
from math import sqrt

import wandb
from time import time

import torch
import torch.optim as optim
from torchvision import utils as vutils

from benchmarking.lap_swd import LapSWD
from benchmarking.neural_metrics import InceptionMetrics
from utils.diffaug import DiffAugment
from models import get_models
from utils.common import copy_G_params, load_params, dump_images
from losses import get_loss_function, calc_gradient_penalty
from utils.data import get_dataloader
from utils.logger import get_dir, PLTLogger, WandbLogger


def get_models_and_optimizers(args):
    netG, netD = get_models(args, device)
    netG.train()
    netD.train()

    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(0.5, 0.9))

    # netG.load_state_dict(torch.load('path')['netG'])
    start_iteration = 0
    if args.resume_last_ckpt:
        ckpts = glob.glob(f'{saved_model_folder}/*.pth')
        if ckpts:
            latest_ckpt = max(ckpts, key = os.path.getctime)
            ckpt = torch.load(latest_ckpt)
            netG.load_state_dict(ckpt['netG'])
            netD.load_state_dict(ckpt['netD'])
            optimizerG.load_state_dict(ckpt['optimizerG'])
            optimizerD.load_state_dict(ckpt['optimizerD'])
            start_iteration = ckpt['iteration']
            print(f"Loaded ckpt of iteration: {start_iteration}")
    return netG, netD, optimizerG, optimizerD, start_iteration


class Prior:
    def __init__(self, prior_type, z_dim):
        self.prior_type = prior_type
        self.z_dim = z_dim
        if "const" in self.prior_type:
            self.z = None
            self.b = int(self.prior_type.split("=")[1])

    def sample(self, b):
        if "const" in self.prior_type:
            if self.z is None:
                self.z = torch.randn((self.b, self.z_dim))
            if b != self.b:
                z = self.z[torch.randint(self.b, (b,))]
            else:
                z = self.z
        elif self.prior_type == "binary":
            z = torch.sign(torch.randn((b, self.z_dim)))
        elif self.prior_type == "uniform":
            z = torch.rand((b, self.z_dim))
        else:
            z = torch.randn((b, self.z_dim))
        return z


def train_GAN(args):
    logger = (WandbLogger if args.wandb else PLTLogger)(args, plots_image_folder)
    prior = Prior(args.z_prior, args.z_dim)
    debug_fixed_noise = prior.sample(args.batch_size).to(device)
    debug_fixed_reals = next(iter(tmp_loader)).to(device)

    inception_metrics = InceptionMetrics([next(iter(train_loader)) for _ in range(args.fid_n_batches)], torch.device("cpu"))
    other_metrics = [
                get_loss_function("BatchEMD-dist=L2"),
                get_loss_function("BatchPatchEMD-dist=L2-p=16-s=16-n_samples=1024"),
                # LapSWD()
              ]

    loss_function = get_loss_function(args.loss_function)

    netG, netD, optimizerG, optimizerD, start_iteration = get_models_and_optimizers(args)

    avg_param_G = copy_G_params(netG)

    start = time()
    iteration = start_iteration
    while iteration < args.n_iterations:
        for real_images in train_loader:
            real_images = real_images.to(device)
            b = real_images.size(0)

            noise = prior.sample(b).to(device)
            fake_images = netG(noise)

            real_images = DiffAugment(real_images, policy=args.augmentation)
            fake_images = DiffAugment(fake_images, policy=args.augmentation)

            # #####  1. train Discriminator #####
            if iteration % args.D_step_every == 0 and args.D_step_every > 0:
                Dloss, debug_Dlosses = loss_function.trainD(netD, real_images, fake_images)
                if args.gp_weight > 0:
                    gp, gradient_norm = calc_gradient_penalty(netD, real_images, fake_images)
                    debug_Dlosses['gradient_norm'] = gradient_norm
                    Dloss += args.gp_weight * gp
                    if "W1" in debug_Dlosses:
                        debug_Dlosses['normalized W1'] = (debug_Dlosses['W1'] /  gradient_norm) if  gradient_norm > 0 else 0
                netD.zero_grad()
                Dloss.backward()
                optimizerD.step()

                if args.weight_clipping is not None:
                    for p in netD.parameters():
                        p.data.clamp_(-args.weight_clipping, args.weight_clipping)

                logger.log(debug_Dlosses, step=iteration)

            if not args.no_fake_resample:
                noise = prior.sample(b).to(device)
                fake_images = netG(noise)
                fake_images = DiffAugment(fake_images, policy=args.augmentation)

            # #####  2. train Generator #####
            if iteration % args.G_step_every == 0:
                Gloss, debug_Glosses = loss_function.trainG(netD, real_images, fake_images)
                netG.zero_grad()
                Gloss.backward()
                optimizerG.step()
                logger.log(debug_Glosses, step=iteration)

            # Update avg weights
            for p, avg_p in zip(netG.parameters(), avg_param_G):
                avg_p.mul_(1 - args.avg_update_factor).add_(args.avg_update_factor * p.data)

            if iteration % 100 == 0:
                it_sec = max(1, iteration - start_iteration) / (time() - start)
                print(f"Iteration: {iteration}: it/sec: {it_sec:.1f}")
                logger.plot()

            if iteration % args.log_freq == 0:
                backup_para = copy_G_params(netG)
                load_params(netG, avg_param_G)

                evaluate(netG, netD, inception_metrics, other_metrics, debug_fixed_noise,
                         debug_fixed_reals, saved_image_folder, iteration, logger, args)
                fname = f"{saved_model_folder}/{'last' if not args.save_every else iteration}.pth"
                torch.save({"iteration": iteration, 'netG': netG.state_dict(), 'netD': netD.state_dict(),
                            "optimizerG":optimizerG.state_dict(), "optimizerD": optimizerD.state_dict()},
                           fname)

                load_params(netG, backup_para)

            iteration += 1


def evaluate(netG, netD, inception_metrics, other_metrics, fixed_noise, debug_fixed_reals,
             saved_image_folder, iteration, logger, args):
    netG.eval()
    netD.eval()
    start = time()
    with torch.no_grad():
        fixed_noise_fake_images = netG(fixed_noise)
        D_fake = netD(fixed_noise_fake_images)
        D_real = netD(debug_fixed_reals)
        logger.log({'D_real': D_real.mean().item(),
                   'D_fake': D_fake.mean().item(),
                   }, step=iteration)

        if args.fid_n_batches > 0 and iteration % args.fid_freq == 0:
            fake_batches = [netG(torch.randn_like(fixed_noise).to(device)) for _ in range(args.fid_n_batches)]
            logger.log(inception_metrics(fake_batches), step=iteration)

        for metric in other_metrics:
            logger.log({
                f'{metric.name}_fixed_noise_gen_to_train': metric(fixed_noise_fake_images, debug_fixed_reals),
            }, step=iteration)

        dump_images(fixed_noise_fake_images,  f'{saved_image_folder}/{iteration}.png')
        if iteration == 0:
            dump_images(debug_fixed_reals, f'{saved_image_folder}/debug_fixed_reals.png')

    netG.train()
    netD.train()
    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ/FFHQ_1000/FFHQ_1000",
                        help="Path to train images")
    parser.add_argument('--augmentation', default='', help="comma separated data augmentation ('color,translation')")
    parser.add_argument('--limit_data', default=None, type=int, help="limit the size of the dataset")
    parser.add_argument('--center_crop', default=None, help='center_crop_data to specified size', type=int)
    parser.add_argument('--gray_scale', action='store_true', default=False, help="Load data as grayscale")

    # Model
    parser.add_argument('--gen_arch', default='DCGAN')
    parser.add_argument('--disc_arch', default='DCGAN')
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--z_dim', default=64, type=int)
    parser.add_argument('--z_prior', default="normal", type=str, help="[normal, binary, uniform]")
    parser.add_argument('--spectral_normalization', action='store_true', default=False)
    parser.add_argument('--weight_clipping', type=float, default=None)

    # Training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--loss_function', default="NonSaturatingGANLoss", type=str)
    parser.add_argument('--gp_weight', default=0, type=float)
    parser.add_argument('--lrG', default=0.0001, type=float)
    parser.add_argument('--lrD', default=0.0001, type=float)
    parser.add_argument('--avg_update_factor', default=1, type=float,
                        help='moving average factor weight of updating generator (1 means none)')
    parser.add_argument('--D_step_every', default=1, type=int, help="D G only evry 'D_step_every' iterations")
    parser.add_argument('--G_step_every', default=1, type=int, help="Update G only evry 'G_step_every' iterations")
    parser.add_argument('--n_iterations', default=1000000, type=int)
    parser.add_argument('--no_fake_resample', default=False, action='store_true')

    # Evaluation
    parser.add_argument('--wandb', action='store_true', default=False, help="Otherwise use PLT localy")
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--save_every', action='store_true', default=False)
    parser.add_argument('--fid_freq', default=10000, type=int)
    parser.add_argument('--fid_n_batches', default=0, type=int, help="How many batches batches for reference FID"
                                                                     " statistics (0 turns off FID)")

    # Other
    parser.add_argument('--project_name', default='GANs')
    parser.add_argument('--tag', default='test')
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--resume_last_ckpt', action='store_true', default=False,
                        help="Search for the latest ckpt in the same folder to resume training")
    parser.add_argument('--load_data_to_memory', action='store_true', default=False)
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()

    args.name = f"{os.path.basename(args.data_path)}_I-{args.im_size}x{args.im_size}_G-{args.gen_arch}_D-{args.disc_arch}" \
                f"{'_GS' if args.gray_scale else ''}{f'_CC-{args.center_crop}' if args.center_crop else ''}" \
                f"_L-{args.loss_function}_Z-{args.z_dim}x{args.z_prior}_B-{args.batch_size}_{args.tag}"

    device = torch.device(args.device)
    if args.device != 'cpu':
        print(f"Working on device: {torch.cuda.get_device_name(device)}")

    saved_model_folder, saved_image_folder, plots_image_folder = get_dir(args)

    train_loader, _ = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers,
                                               val_percentage=0, gray_scale=args.gray_scale, center_crop=args.center_crop,
                                               load_to_memory=args.load_data_to_memory, limit_data=args.limit_data)

    tmp_loader, _ = get_dataloader(args.data_path, args.im_size, 1000, args.n_workers,
                                               val_percentage=0, gray_scale=args.gray_scale, center_crop=args.center_crop,
                                               load_to_memory=args.load_data_to_memory, limit_data=args.limit_data)


    train_GAN(args)



