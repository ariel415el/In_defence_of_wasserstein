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
from utils.data import get_dataloader
from utils.logger import get_dir, LossLogger

from benchmarking.fid import FID_score
from benchmarking.lap_swd import lap_swd, LapSWD


def train_d(net, data, label="real"):
    """FastGAN train loss"""
    pred = net(data)
    if label == 'real':
        pred *= -1
    D_loss = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
    return D_loss


def train_GAN(args):
    debug_fixed_noise = torch.randn((args.batch_size, args.z_dim)).to(device)
    debug_fixed_reals = next(train_loader).to(device)
    debug_fixed_reals_test = next(test_loader).to(device)

    fid_metric = FID_score({"train": train_loader, "test":test_loader}, args.n_fid_batches, torch.device("cpu")) if args.n_fid_batches else None

    other_metrics = [
                LapSWD(),
                PatchSWD(p=9, n=128)
              ]

    netG, netD = get_models(args, device)

    avg_param_G = copy_G_params(netG)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    logger = LossLogger(saved_image_folder)
    start = time()
    for iteration in tqdm(range(args.n_iterations + 1)):
        real_image = next(train_loader).to(device)
        b = real_image.size(0)

        noise = torch.randn((b, args.z_dim)).to(device)
        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=args.augmentaion)
        fake_images = DiffAugment(fake_images, policy=args.augmentaion)

        # #####  1. train Discriminator #####
        netD.zero_grad()
        Dloss_real = train_d(netD, real_image, label="real")
        Dloss_fake = train_d(netD, fake_images.detach(), label="fake")
        Dloss_real.backward()
        Dloss_fake.backward()
        optimizerD.step()

        # #####  2. train Generator #####
        netG.zero_grad()
        Gloss = train_d(netD, fake_images, label="real")
        Gloss.backward()
        optimizerG.step()

        # Update avg weights
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        logger.aggregate_data({"Dloss_real": Dloss_real.item(), "Dloss_fake":Dloss_fake.item()})
        if iteration % 100 == 0:
            sec_per_kimage = (time() - start) / (max(1, iteration) / 1000)
            print(f"Dloss-real: {Dloss_real.item():.5f}, Dloss-fake {Dloss_fake.item():.5f} sec/kimg: {sec_per_kimage:.1f}")

        if iteration % (args.save_interval) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)

            evaluate(netG, netD,
                     fid_metric, other_metrics,
                     debug_fixed_noise, debug_fixed_reals, debug_fixed_reals_test,
                     logger, saved_image_folder, iteration)
            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)

            load_params(netG, backup_para)


def evaluate(netG, netD,
             fid_metric, other_metrics,
             fixed_noise, debug_fixed_reals, debug_fixed_reals_test,
             logger, saved_image_folder, iteration):
    start = time()
    with torch.no_grad():

        Dloss_fixed_real_train = train_d(netD, debug_fixed_reals, label="real").item()
        Dloss_fixedreal_test = train_d(netD, debug_fixed_reals_test, label="real").item()
        logger.add_data({'Dloss_fixed_reals_train': Dloss_fixed_real_train, 'Dloss_fixed_reals_test':Dloss_fixedreal_test})

        combined_plots = {"D_eval": ["Dloss_fixed_reals_train", "Dloss_fixed_reals_test"],
                          "D_train": ["Dloss_real", "Dloss_fake"],
                         }

        fixed_noise_fake_images = netG(fixed_noise)
        nrow = int(sqrt(len(fixed_noise_fake_images)))
        vutils.save_image(fixed_noise_fake_images.add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration, nrow=nrow)

        fake_images = netG(torch.randn_like(fixed_noise).to(device))

        if fid_metric is not None:
            fixed_fid = fid_metric([fixed_noise_fake_images]).values
            fid = fid_metric([fake_images])
            logger.add_data({
                'fixed_fid_train': fixed_fid['train'], 'fixed_fid_test': fixed_fid['test'], 'fid_train': fid['train'], 'fid_test': fid['test']
            })
            combined_plots['FID'] = ['fixed_fid_train', 'fixed_fid_test', 'fid_train', 'fid_test']

        for metric in other_metrics:
            logger.add_data({
                f'{metric}_train_fixed': metric(fixed_noise_fake_images, debug_fixed_reals).item(),
                f'{metric}_test_fixed': metric(fixed_noise_fake_images, debug_fixed_reals_test).item(),
                f'{metric}_train': metric(fake_images, debug_fixed_reals).item(),
                f'{metric}_test': metric(fake_images, debug_fixed_reals_test).item(),
            })
            combined_plots[f'{metric}'] = [f'{metric}_train_fixed',  f'{metric}_test_fixed', f'{metric}_train', f'{metric}_test']

        logger.plot(combined_plots)

    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    args = argparse.Namespace()
    args.data_path = '/mnt/storage_ssd/datasets/FFHQ_1000_images/images'
    args.Generator_architecture = 'DCGAN'
    args.Discriminator_architecture = 'DCGAN'
    args.im_size = 64
    args.z_dim = 64
    args.batch_size = 64
    args.lr = 0.0001
    args.n_iterations = 1000000
    args.name = f"{os.path.basename(args.data_path)}_{args.im_size}x{args.im_size}_G-{args.Generator_architecture}" \
                f"_D-{args.Discriminator_architecture}_Z-{args.z_dim}_B-{args.batch_size}"
    args.augmentaion = 'color,translation'
    args.n_workers = 0
    args.save_interval = 1000
    args.n_fid_batches = 4

    device = torch.device("cuda")

    saved_model_folder, saved_image_folder = get_dir(args)

    train_loader, test_loader = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers)

    train_GAN(args)



