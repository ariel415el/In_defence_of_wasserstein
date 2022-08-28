import argparse
from tqdm import tqdm
from copy import deepcopy
from time import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils as vutils

from diffaug import DiffAugment
from utils.data import get_dataloader
from utils.logging import get_dir, LossLogger

from benchmarking.patch_swd import patch_swd
from benchmarking.patch_swd import swd
from benchmarking.fid import fid_loss
from benchmarking.lap_swd import lap_swd
policy = 'color,translation'
# policy = 'add_noise'


def train_d(net, data, label="real"):
    """Train function of discriminator"""
    pred = net(data)
    if label == 'real':
        pred *= -1
    D_loss = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
    return D_loss


def get_models(args):
    if args.architecture == 'DCGAN':
        from models.DCGAN import Discriminator, Generator, weights_init
    elif args.architecture == 'StyleGAN':
        from models.StyleGAN import Discriminator, Generator, weights_init
    elif args.architecture == 'ProjectedGAN':
        from models.FastGAN import Generator, weights_init
        from models.ProjectedGAN import Discriminator
    else:
        from models.FastGAN import Discriminator, Generator, weights_init
    netG = Generator(args.z_dim).to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    print("D params: ", sum(p.numel() for p in netD.parameters() if p.requires_grad))
    print("G params: ", sum(p.numel() for p in netG.parameters() if p.requires_grad))

    return netG, netD


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def train(args):
    saved_model_folder, saved_image_folder = get_dir(args)

    train_loader, test_loader = get_dataloader(args.data_path, args.im_size, args.batch_size, n_workers)

    debug_fixed_noise = torch.randn((args.batch_size, args.z_dim)).to(device)
    debug_fixed_reals = next(train_loader).to(device)
    debug_fixed_reals_test = next(test_loader).to(device)

    netG, netD = get_models(args)

    avg_param_G = copy_G_params(netG)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.nbeta1, 0.999))
    logger = LossLogger(saved_image_folder)
    start = time()
    for iteration in tqdm(range(args.n_iterations + 1)):
        real_image = next(train_loader).to(device)
        b = real_image.size(0)

        noise = torch.randn((b, args.z_dim)).to(device)
        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = DiffAugment(fake_images, policy=policy)

        ## 1. train Discriminator
        netD.zero_grad()

        Dloss_real = train_d(netD, real_image, label="real")
        Dloss_fake = train_d(netD, fake_images.detach(), label="fake")
        Dloss_real.backward()
        Dloss_fake.backward()
        optimizerD.step()

        ## 2. train Generator
        netG.zero_grad()
        Gloss = -netD(fake_images).mean()
        Gloss.backward()
        optimizerG.step()

        # Update avg weights
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        logger.aggregate_train_losses(Gloss.item(), Dloss_real.item(), Dloss_fake.item())
        if iteration % 100 == 0:
            sec_per_kimage = (time() - start) / (max(1, iteration) / 1000)
            print(f"G loss: {Gloss:.5f}: Dloss-real: {Dloss_real.item():.5f}, Dloss-fake {Dloss_fake.item():.5f} sec/kimg: {sec_per_kimage}")

        if iteration % (save_interval) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                generated_images = netG(debug_fixed_noise)
                vutils.save_image(generated_images.add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration, nrow=4)

                x = debug_fixed_reals.cpu()
                y = generated_images.cpu()
                logger.log_other_losses({'fid_loss': fid_loss(x, y).item(),
                                       'swd': swd(x, y).item(),
                                       'patch_swd': patch_swd(x, y).item(),
                                       'lap_swd': lap_swd(x, y).item()
                                        })

                Dloss_real_train = train_d(netD, debug_fixed_reals, label="real").item()
                Dloss_real_test = train_d(netD, debug_fixed_reals_test, label="real").item()
                logger.log_eval_losses(Dloss_real_train, Dloss_real_test)

                logger.plot()

            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)

            load_params(netG, backup_para)


if __name__ == "__main__":
    args = argparse.Namespace()
    args.data_path = '/mnt/storage_ssd/datasets/FFHQ_128'
    args.architecture = 'ProjectedGAN'
    args.batch_size = 8
    args.n_iterations = 50000
    args.im_size = 128
    args.z_dim = 64
    args.lr = 0.0002
    args.nbeta1 = 0.5
    args.name = f"FFHQ-5k_{args.architecture}_Z-{args.z_dim}_B-{args.batch_size}"

    n_workers = 0
    save_interval = 1000
    device = torch.device("cuda:0")

    train(args)



