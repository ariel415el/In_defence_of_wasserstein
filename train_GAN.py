from math import sqrt

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

from benchmarking.fid import FID_score
from benchmarking.lap_swd import lap_swd


def train_d(net, data, label="real"):
    """Train function of discriminator"""
    pred = net(data)
    if label == 'real':
        pred *= -1
    D_loss = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
    return D_loss


def get_models(args):
    from models.FastGAN import Discriminator, Generator, weights_init

    netG = Generator(args.z_dim, skip_connections=True).to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    print("D params: ", sum(p.numel() for p in netD.parameters() if p.requires_grad))
    print("G params: ", sum(p.numel() for p in netG.parameters() if p.requires_grad))

    # netG = nn.DataParallel(netG)
    # netD = nn.DataParallel(netD)

    return netG, netD


def train_GAN(args):
    debug_fixed_noise = torch.randn((args.batch_size, args.z_dim)).to(device)
    debug_fixed_reals = next(train_loader).to(device)
    debug_fixed_reals_test = next(test_loader).to(device)

    train_fid_calculator = FID_score([next(train_loader).to(device) for _ in range(16)], device)
    test_fid_calculator = FID_score([next(train_loader).to(device) for _ in range(16)], device)

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

        real_image = DiffAugment(real_image, policy=args.augmentaion)
        fake_images = DiffAugment(fake_images, policy=args.augmentaion)

        ## 1. train Discriminator
        netD.zero_grad()

        Dloss_real = train_d(netD, real_image, label="real")
        Dloss_fake = train_d(netD, fake_images.detach(), label="fake")
        # gp = 10 * calc_gradient_penalty(netD, real_image, fake_images, device)
        Dloss_real.backward()
        Dloss_fake.backward()
        # gp.backward()

        optimizerD.step()

        ## 2. train Generator
        netG.zero_grad()
        Gloss = -netD(fake_images).mean()
        Gloss.backward()
        optimizerG.step()

        # Update avg weights
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        logger.aggregate_data({"Gloss":Gloss.item(), "Dloss_real": Dloss_real.item(), "Dloss_fake":Dloss_fake.item()})
        if iteration % 100 == 0:
            sec_per_kimage = (time() - start) / (max(1, iteration) / 1000)
            print(f"G loss: {Gloss:.5f}: Dloss-real: {Dloss_real.item():.5f}, Dloss-fake {Dloss_fake.item():.5f} sec/kimg: {sec_per_kimage:.1f}")

        if iteration % (args.save_interval) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)

            evaluate(netG, netD, debug_fixed_noise,
                     debug_fixed_reals,
                     debug_fixed_reals_test, logger,
                     train_fid_calculator,
                     test_fid_calculator,
                     saved_image_folder, iteration)
            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)

            load_params(netG, backup_para)


def evaluate(netG, netD, debug_fixed_noise,
             debug_fixed_reals,
             debug_fixed_reals_test, logger,
             train_fid_calculator,
             test_fid_calculator,
             saved_image_folder, iteration):
    start = time()
    with torch.no_grad():
        fixed_noise_fake_images = netG(debug_fixed_noise)
        nrow = int(sqrt(len(fixed_noise_fake_images)))
        vutils.save_image(fixed_noise_fake_images.add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration, nrow=nrow)

        fake_images = [netG(torch.randn_like(debug_fixed_noise).to(device)) for _ in range(16)]
        logger.add_data({
            'fixed_batch_fid_to_train': train_fid_calculator.calc_fid([fixed_noise_fake_images]).item(),
            'fixed_batch_fid_to_test': test_fid_calculator.calc_fid([fixed_noise_fake_images]).item(),
            'full_fid_to_train': train_fid_calculator.calc_fid(fake_images).item(),
            'full_fid_to_test': test_fid_calculator.calc_fid(fake_images).item(),
        })

        logger.add_data({
            'lap_swd_train': lap_swd(fixed_noise_fake_images, debug_fixed_reals).item(),
            'lap_swd_test': lap_swd(fixed_noise_fake_images, debug_fixed_reals_test).item()
        })

        Dloss_real_train = train_d(netD, debug_fixed_reals, label="real").item()
        Dloss_real_test = train_d(netD, debug_fixed_reals_test, label="real").item()
        logger.add_data({'Dloss_real_train': Dloss_real_train, 'Dloss_real_test':Dloss_real_test})
        logger.plot({"D_eval": ["Dloss_real_train", "Dloss_real_test"],
                     "D_train": ["Dloss_real", "Dloss_fake"],
                     "FIDs": ['fixed_batch_fid_to_train', 'fixed_batch_fid_to_test', 'full_fid_to_train', 'full_fid_to_test'],
                     "lap_SWD": ['lap_swd_train', 'lap_swd_test']
                     })

    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    from config import args

    device = torch.device("cuda")

    saved_model_folder, saved_image_folder = get_dir(args)

    train_loader, test_loader = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers)

    train_GAN(args)



