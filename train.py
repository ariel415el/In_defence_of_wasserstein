import argparse
import glob
import os.path
from math import sqrt

import wandb
from time import time, strftime

import torch
import torch.optim as optim
from torchvision import utils as vutils

from diffaug import DiffAugment
from models import get_models
from utils.common import copy_G_params, load_params
from losses import get_loss_function, calc_gradient_penalty
from utils.data import get_dataloader
from utils.logger import get_dir

from benchmarking.fid import FID_score
from benchmarking.lap_swd import LapSWD
from benchmarking.emd import patchEMD, EMD


def get_models_and_optimizers(args):
    netG, netD = get_models(args, device)
    netG.train()
    netD.train()

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

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


def train_GAN(args):
    wandb.init(project=f"GANs", dir=plots_image_folder, name=args.name)
    debug_fixed_noise = torch.randn((args.batch_size, args.z_dim)).to(device)
    debug_fixed_reals = next(train_loader).to(device)
    debug_fixed_reals_test = next(test_loader).to(device)

    fid_metric = FID_score({"train": train_loader, "test":test_loader}, args.fid_n_batches, torch.device("cpu")) if args.fid_n_batches else None
    other_metrics = [
                EMD(),
                patchEMD(p=9, n=256),
                patchEMD(p=17, n=256),
                patchEMD(p=33, n=256),
                LapSWD(),
              ]

    loss_function = get_loss_function(args.loss_function)

    netG, netD, optimizerG, optimizerD, start_iteration = get_models_and_optimizers(args)

    avg_param_G = copy_G_params(netG)

    start = time()
    for iteration in range(start_iteration, args.n_iterations + 1):
        real_images = next(train_loader).to(device)
        b = real_images.size(0)

        noise = torch.randn((b, args.z_dim)).to(device)
        fake_images = netG(noise)

        real_images = DiffAugment(real_images, policy=args.augmentation)
        fake_images = DiffAugment(fake_images, policy=args.augmentation)

        # #####  1. train Discriminator #####
        for i in range(args.n_D_steps):
            Dloss, debug_Dlosses = loss_function.trainD(netD, real_images, fake_images)
            netD.zero_grad()
            Dloss.backward(retain_graph=args.n_D_steps > 1)
            if args.gp_weight > 0:
                Dloss += args.gp_weight * calc_gradient_penalty(netD, real_images, fake_images)
            optimizerD.step()
            wandb.log(debug_Dlosses, step=iteration)

        # #####  2. train Generator #####
        if iteration % args.G_step_every == 0:
            Gloss, debug_Glosses = loss_function.trainG(netD, real_images, fake_images)
            netG.zero_grad()
            Gloss.backward()
            optimizerG.step()
            wandb.log(debug_Glosses, step=iteration)

        # Update avg weights
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(1 - args.avg_update_factor).add_(args.avg_update_factor * p.data)

        if iteration % 100 == 0:
            it_sec = max(1, iteration - start_iteration) / (time() - start)
            print(f"Iteration: {iteration}: it/sec: {it_sec:.1f}")

        if iteration % args.save_interval == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)

            evaluate(netG, netD, fid_metric, other_metrics, debug_fixed_noise,
                     debug_fixed_reals, debug_fixed_reals_test, saved_image_folder, iteration, args)
            torch.save({"iteration": iteration, 'netG': netG.state_dict(), 'netD': netD.state_dict(),
                        "optimizerG":optimizerG.state_dict(), "optimizerD": optimizerD.state_dict()},
                       saved_model_folder + '/%d.pth' % iteration)

            load_params(netG, backup_para)


def evaluate(netG, netD, fid_metric, other_metrics, fixed_noise, debug_fixed_reals, debug_fixed_reals_test,
             saved_image_folder, iteration, args):
    netG.eval()
    netD.eval()
    start = time()
    with torch.no_grad():
        D_on_fixed_real_train = netD(debug_fixed_reals).mean().item()
        D_on_fixed_real_test = netD(debug_fixed_reals_test).mean().item()
        wandb.log({'D_on_fixed_real_train': D_on_fixed_real_train, 'D_on_fixed_real_test':D_on_fixed_real_test}, step=iteration)

        fixed_noise_fake_images = netG(fixed_noise)
        nrow = int(sqrt(len(fixed_noise_fake_images)))
        vutils.save_image(fixed_noise_fake_images,  f'{saved_image_folder}/{iteration}.jpg', nrow=nrow, normalize=True)
        if fid_metric is not None and iteration % args.fid_freq == 0:
            fixed_fid = fid_metric([fixed_noise_fake_images])
            fid = fid_metric([netG(torch.randn_like(fixed_noise).to(device)) for _ in range(args.fid_n_batches)])
            wandb.log({'fixed_fid_train': fixed_fid['train'], 'fixed_fid_test': fixed_fid['test'],
                       'fid_train': fid['train'], 'fid_test': fid['test'] },
                      step=iteration)

        # fake_images = netG(torch.randn_like(fixed_noise).to(device))
        for metric in other_metrics:
            wandb.log({
                f'{metric}_fixed_noise_gen_to_train': metric(fixed_noise_fake_images, debug_fixed_reals),
                # f'{metric}_fixed_noise_gen_to_test': metric(fixed_noise_fake_images, debug_fixed_reals_test),
                # f'{metric}_gen_to_train': metric(fake_images, debug_fixed_reals),
                # f'{metric}_gen_to_test': metric(fake_images, debug_fixed_reals_test),
            }, step=iteration)

        if iteration == 0:
            vutils.save_image(debug_fixed_reals, f'{saved_image_folder}/debug_fixed_reals.jpg', nrow=nrow, normalize=True)

    netG.train()
    netD.train()
    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ_1000/FFHQ_1000",
                        help="Path to train images")
    parser.add_argument('--center_crop', default=None, help='center_crop_data', type=int)
    parser.add_argument('--augmentation', default='', help="comma separated data augmentation ('color,translation')")

    # Model
    parser.add_argument('--gen_arch', default='DCGAN')
    parser.add_argument('--disc_arch', default='DCGAN')
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--z_dim', default=64, type=int)

    # Training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--loss_function', default="SoftHingeLoss", type=str)
    parser.add_argument('--gp_weight', default=0, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--avg_update_factor', default=1, type=float,
                        help='moving average factor weight of updating generator (1 means none)')
    parser.add_argument('--n_D_steps', default=1, type=int, help="Number of repeated D updates with each batch")
    parser.add_argument('--G_step_every', default=1, type=int, help="Update G only evry 'G_step_every' iterations")
    parser.add_argument('--n_iterations', default=100000, type=int)

    # Evaluation
    parser.add_argument('--outputs_root', default='Outputs')
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--fid_freq', default=10000, type=int)
    parser.add_argument('--fid_n_batches', default=0, type=int, help="How many batches of train/test to compute "
                                                                     "reference FID statistics (0 turns off FID)")

    # Other
    parser.add_argument('--tag', default='test')
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--resume_last_ckpt', action='store_true', default=False,
                        help="Search for the latest ckpt in the same folder to resume training")
    parser.add_argument('--load_data_to_memory', action='store_true', default=False)
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()
    args.name = f"{os.path.basename(args.data_path)}_{args.im_size}x{args.im_size}_G-{args.gen_arch}" \
                f"_D-{args.disc_arch}_L-{args.loss_function}_Z-{args.z_dim}_B-{args.batch_size}_{args.tag}"

    device = torch.device(args.device)
    print(f"Working on device: {torch.cuda.get_device_name(device)}")

    saved_model_folder, saved_image_folder, plots_image_folder = get_dir(args)

    train_loader, test_loader = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers,
                                               val_percentage=0.01,
                                               load_to_memory=args.load_data_to_memory)

    train_GAN(args)



