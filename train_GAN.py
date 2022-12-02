import argparse
import os.path
from math import sqrt

from tqdm import tqdm
from time import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils as vutils

from benchmarking.patch_swd import PatchSWD
from diffaug import DiffAugment
from models import get_models
from utils.common import copy_G_params, load_params
from utils.losses import get_loss_function
from utils.data import get_dataloader
from utils.logger import get_dir, LossLogger

from benchmarking.fid import FID_score
from benchmarking.lap_swd import LapSWD


def train_GAN(args):
    debug_fixed_noise = torch.randn((args.batch_size, args.z_dim)).to(device)
    debug_fixed_reals = next(train_loader).to(device)
    debug_fixed_reals_test = next(test_loader).to(device)

    fid_metric = FID_score({"train": train_loader, "test":test_loader}, args.fid_n_batches, torch.device("cpu")) if args.fid_n_batches else None

    other_metrics = [
                LapSWD(),
                PatchSWD(p=9, n=128)
              ]

    netG, netD = get_models(args, device)
    netG.train()
    netD.train()
    avg_param_G = copy_G_params(netG)

    loss_function = get_loss_function(args.loss_fucntion)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    logger = LossLogger(saved_image_folder)
    start = time()
    for iteration in tqdm(range(args.n_iterations + 1)):
        real_images = next(train_loader).to(device)
        b = real_images.size(0)

        noise = torch.randn((b, args.z_dim)).to(device)
        fake_images = netG(noise)

        real_images = DiffAugment(real_images, policy=args.augmentaion)
        fake_images = DiffAugment(fake_images, policy=args.augmentaion)

        # #####  1. train Discriminator #####
        for i in range(args.n_D_steps):
            Dloss, debug_Dlosses = loss_function.trainD(netD, real_images, fake_images)
            netD.zero_grad()
            Dloss.backward(retain_graph=args.n_D_steps > 1)
            optimizerD.step()

        # #####  2. train Generator #####
        if iteration % args.G_step_every == 0:
            Gloss, debug_Glosses = loss_function.trainG(netD, real_images, fake_images)
            netG.zero_grad()
            Gloss.backward()
            optimizerG.step()

        # Update avg weights
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(1 - args.avg_update_factor).add_(args.avg_update_factor * p.data)

        logger.aggregate_data(debug_Dlosses, group_name="D_train")
        logger.aggregate_data(debug_Glosses, group_name="G_train")
        if iteration % 100 == 0:
            sec_per_kimage = (time() - start) / (max(1, iteration) / 1000)
            print(str({k: f"{v:.6f}" for k, v in debug_Dlosses.items()}) + f"sec/kimg: {sec_per_kimage:.1f}")

        if iteration % args.save_interval == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)

            evaluate(netG, netD,
                     fid_metric, other_metrics,
                     debug_fixed_noise, debug_fixed_reals, debug_fixed_reals_test,
                     logger, saved_image_folder, iteration, args)
            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)

            load_params(netG, backup_para)


def evaluate(netG, netD,
             fid_metric, other_metrics,
             fixed_noise, debug_fixed_reals, debug_fixed_reals_test,
             logger, saved_image_folder, iteration, args):
    netG.eval()
    netD.eval()
    start = time()
    with torch.no_grad():
        D_on_fixed_real_train = netD(debug_fixed_reals).mean().item()
        D_on_fixed_real_test = netD(debug_fixed_reals_test).mean().item()
        logger.add_data({'D_on_fixed_real_train': D_on_fixed_real_train, 'D_on_fixed_real_test':D_on_fixed_real_test}, group_name="D_eval")

        fixed_noise_fake_images = netG(fixed_noise)
        nrow = int(sqrt(len(fixed_noise_fake_images)))
        vutils.save_image(fixed_noise_fake_images, saved_image_folder + '/%d.jpg' % iteration, nrow=nrow, normalize=True)

        fake_images = netG(torch.randn_like(fixed_noise).to(device))

        if fid_metric is not None and iteration % args.fid_freq == 0:
            fixed_fid = fid_metric([fixed_noise_fake_images])
            fid = fid_metric([netG(torch.randn_like(fixed_noise).to(device)) for _ in range(args.fid_n_batches)])
            logger.add_data({
                'fixed_fid_train': fixed_fid['train'], 'fixed_fid_test': fixed_fid['test'], 'fid_train': fid['train'], 'fid_test': fid['test']
            }, group_name="FID")

        for metric in other_metrics:
            logger.add_data({
                f'{metric}_train_fixed': metric(fixed_noise_fake_images, debug_fixed_reals).item(),
                f'{metric}_test_fixed': metric(fixed_noise_fake_images, debug_fixed_reals_test).item(),
                f'{metric}_train': metric(fake_images, debug_fixed_reals).item(),
                f'{metric}_test': metric(fake_images, debug_fixed_reals_test).item(),
            }, group_name=f"{metric}")

        logger.plot()

    netG.train()
    netD.train()
    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ_1000/FFHQ_1000", help="Path to train images")
    parser.add_argument('--Generator_architecture', default='DCGAN')
    parser.add_argument('--Discriminator_architecture', default='DCGAN')
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--z_dim', default=64, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--loss_fucntion', default="SoftHingeLoss", type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--avg_update_factor', default=0.001, type=float, help='moving average factor weight of updating generator (1 means none)')
    parser.add_argument('--n_D_steps', default=1, type=int, help="Number of repeated D updates with each batch")
    parser.add_argument('--G_step_every', default=1, type=int, help="Update G only evry 'G_step_every' iterations")
    parser.add_argument('--n_iterations', default=100000, type=int)
    parser.add_argument('--augmentaion', default='color,translation', help="comma separated data augmentaitons")
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--fid_freq', default=10000, type=int)
    parser.add_argument('--fid_n_batches', default=0, type=int, help="How many batches of train/test to compute "
                                                                     "reference FID statistics (0 turns off FID)")
    parser.add_argument('--outputs_root', default='Outputs')
    parser.add_argument('--tag', default='test')
    parser.add_argument('--n_workers', default=0, type=int)
    args = parser.parse_args()
    args.name = f"{os.path.basename(args.data_path)}_{args.im_size}x{args.im_size}_G-{args.Generator_architecture}" \
                f"_D-{args.Discriminator_architecture}_L-{args.loss_fucntion}_Z-{args.z_dim}_B-{args.batch_size}_{args.tag}"

    device = torch.device(args.device)

    saved_model_folder, saved_image_folder = get_dir(args)

    train_loader, test_loader = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers)

    train_GAN(args)



