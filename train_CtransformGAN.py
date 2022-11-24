from math import sqrt

from tqdm import tqdm
from time import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils as vutils

from diffaug import DiffAugment
from models.BagNet import BagNet, Bottleneck
from utils.common import copy_G_params, load_params
from utils.data import get_dataloader
from utils.logger import get_dir, LossLogger

from benchmarking.fid import FID_score
from benchmarking.lap_swd import lap_swd

def get_c_transform_scores(reals, fakes, f_reals):
    """
    reals.shape = (b1, ...)
    fakes.shape = (b2, ...)
    f_reals.shape = (b1)
    dist.shape = (b2, b1)
    """
    X = fakes.reshape(fakes.shape[0], -1)
    Y = reals.reshape(reals.shape[0], -1)
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    d = X.shape[1]
    dist = dist / d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    dist = dist - f_reals[None, :, 0]

    return torch.min(dist, dim=1)[0]
    # return dist[:, NNs.long()]



def get_models(args):
    # from models.FastGAN import Discriminator, Generator, weights_init
    from models.FastGAN import Discriminator, Generator, weights_init

    netG = Generator(args.z_dim, skip_connections=False).to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    # netD = BagNet(Bottleneck, [3, 4, 6, 3], strides=[2, 2, 2, 1], kernel3=[1, 1, 1, 1], num_classes=1).to(device)
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

    netG, netC = get_models(args)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.nbeta1, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=args.lr, betas=(args.nbeta1, 0.999))

    logger = LossLogger(saved_image_folder)
    start = time()
    for iteration in tqdm(range(args.n_iterations + 1)):
        real_image = next(train_loader).to(device)
        b = real_image.size(0)

        noise = torch.randn((b, args.z_dim)).to(device)
        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=args.augmentaion)
        fake_images = DiffAugment(fake_images, policy=args.augmentaion)

        # Define c-transform
        real_scores = netC(real_image)
        fake_scores = get_c_transform_scores(real_image, fake_images, real_scores)

        ## 1. train Discriminator
        netC.zero_grad()

        Closs = -(real_scores.mean() + fake_scores.mean().detach())
        Closs.backward()

        optimizerC.step()

        ## 2. train Generator
        if iteration % 5 ==0:
            netG.zero_grad()
            real_scores = netC(real_image)
            fake_scores = get_c_transform_scores(real_image, fake_images, real_scores)
            Gloss = real_scores.mean() + fake_scores.mean()
            Gloss.backward()
            optimizerG.step()

        logger.aggregate_data({"Gloss":Gloss.item(), "Closs": Closs.item()})
        if iteration % 100 == 0:
            sec_per_kimage = (time() - start) / (max(1, iteration) / 1000)
            print(f"G loss: {Gloss:.5f}: Closs: {Closs.item():.5f}, sec/kimg: {sec_per_kimage:.1f}")

        if iteration % (args.save_interval) == 0:
            evaluate(netG, netC, debug_fixed_noise,
                     debug_fixed_reals,
                     debug_fixed_reals_test, logger,
                     train_fid_calculator,
                     test_fid_calculator,
                     saved_image_folder, iteration)
            torch.save({'g': netG.state_dict(), 'c': netC.state_dict()}, saved_model_folder + '/%d.pth' % iteration)


def evaluate(netG, netC, debug_fixed_noise,
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

        # fake_images = [netG(torch.randn_like(debug_fixed_noise).to(device)) for _ in range(16)]
        # logger.add_data({
        #     'fixed_batch_fid_to_train': train_fid_calculator.calc_fid([fixed_noise_fake_images]).item(),
        #     'fixed_batch_fid_to_test': test_fid_calculator.calc_fid([fixed_noise_fake_images]).item(),
        #     'full_fid_to_train': train_fid_calculator.calc_fid(fake_images).item(),
        #     'full_fid_to_test': test_fid_calculator.calc_fid(fake_images).item(),
        # })

        logger.add_data({
            'lap_swd_train': lap_swd(fixed_noise_fake_images, debug_fixed_reals).item(),
            'lap_swd_test': lap_swd(fixed_noise_fake_images, debug_fixed_reals_test).item()
        })

        Dloss_real_train = netC(debug_fixed_reals).mean()
        Dloss_real_test = netC(debug_fixed_reals_test).mean()
        logger.add_data({'real_scores_train': Dloss_real_train, 'real_scores_test':Dloss_real_test})
        logger.plot({"C_eval": ["real_scores_train", "real_scores_test"],
                     "C_train": ["Gloss", "Closs"],
                     # "FIDs": ['fixed_batch_fid_to_train', 'fixed_batch_fid_to_test', 'full_fid_to_train', 'full_fid_to_test'],
                     "lap_SWD": ['lap_swd_train', 'lap_swd_test']
                     })

    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    from config import args
    args.name = args.name.replace('FastGAN', 'C-transfromFastGAN')

    device = torch.device("cuda")

    saved_model_folder, saved_image_folder = get_dir(args)

    train_loader, test_loader = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers)

    train_GAN(args)



