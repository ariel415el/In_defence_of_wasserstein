import argparse
from tqdm import tqdm
from time import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils as vutils

from diffaug import DiffAugment
from utils.common import copy_G_params, load_params
from utils.data import get_dataloader
from utils.logger import get_dir, LossLogger

from benchmarking.patch_swd import patch_swd
from benchmarking.patch_swd import swd
from benchmarking.fid import fid_loss
from benchmarking.lap_swd import lap_swd

policy = 'color,translation'


def get_models(args):
    from models.FastGAN import Discriminator, Generator, weights_init

    netG = Generator(args.z_dim).to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    print("D params: ", sum(p.numel() for p in netD.parameters() if p.requires_grad))
    print("G params: ", sum(p.numel() for p in netG.parameters() if p.requires_grad))

    return netG, netD


def train_d(net, data, label="real"):
    """Train function of discriminator"""
    pred = net(data)
    if label == 'real':
        pred *= -1
    D_loss = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
    return D_loss


def train_GAN(args):
    netG, netD = get_models(args)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.nbeta1, 0.999))

    debug_fixed_noise = torch.randn((args.batch_size, args.z_dim)).to(device)
    debug_fixed_reals = next(train_loader).to(device)
    debug_fixed_reals_test = next(test_loader).to(device)

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
        real_features = netD.features(real_image)
        fake_features = netD.features(fake_images)

        Gloss = swd(real_features, fake_features)

        Gloss.backward()
        optimizerG.step()

        logger.aggregate_data({"Gloss": Gloss.item(), "Dloss_real": Dloss_real.item(), "Dloss_fake": Dloss_fake.item()})
        if iteration % 100 == 0:
            sec_per_kimage = (time() - start) / (max(1, iteration) / 1000)
            print(
                f"G loss: {Gloss:.5f}: Dloss-real: {Dloss_real.item():.5f}, Dloss-fake {Dloss_fake.item():.5f} sec/kimg: {sec_per_kimage:.1f}")

        if iteration % (args.save_interval) == 0:
            with torch.no_grad():
                generated_images = netG(debug_fixed_noise)
                vutils.save_image(generated_images.add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration, nrow=4)

                Dloss_real_train = train_d(netD, debug_fixed_reals, label="real").item()
                Dloss_real_test = train_d(netD, debug_fixed_reals_test, label="real").item()
                x = debug_fixed_reals.cpu()
                y = generated_images.cpu()
                logger.add_data({'Dloss_real_train': Dloss_real_train,
                                 'Dloss_real_test': Dloss_real_test,
                                 'fid_loss': fid_loss(x, y).item(),
                                 'swd': swd(x, y).item(),
                                 'patch_swd': patch_swd(x, y).item(),
                                 'lap_swd': lap_swd(x, y).item()
                                 })

                logger.plot({"D_eval": ["Dloss_real_train", "Dloss_real_test"],
                             "D_train": ["Dloss_real", "Dloss_fake"]})

            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)


if __name__ == "__main__":
    from config import args

    args.name = "SWG-" + args.name

    device = torch.device("cuda:0")

    saved_model_folder, saved_image_folder = get_dir(args)

    train_loader, test_loader = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers)

    train_GAN(args)
