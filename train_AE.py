from tqdm import tqdm
from time import time

import torch
import torch.optim as optim
from torchvision import utils as vutils

from diffaug import DiffAugment
from utils.data import get_dataloader
from utils.logger import get_dir, LossLogger

from benchmarking.patch_swd import patch_swd
from benchmarking.patch_swd import swd
from benchmarking.fid import fid_loss
from benchmarking.lap_swd import lap_swd

from losses.VGGFeatures import VGGPerceptualLoss

policy = 'color,translation'


def get_models(args):
    from models.FastGAN import Discriminator, Generator, weights_init

    netG = Generator(args.z_dim, skip_connections=True).to(device)
    netG.apply(weights_init)

    netD = Discriminator(args.z_dim).to(device)
    netD.apply(weights_init)

    print("D params: ", sum(p.numel() for p in netD.parameters() if p.requires_grad))
    print("G params: ", sum(p.numel() for p in netG.parameters() if p.requires_grad))

    return netG, netD

def L2_loss(x, y):
    return ((x-y)**2).mean()

def train_AE(args):
    Decoder, Encoder = get_models(args)

    optimizer = optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=args.lr, betas=(args.nbeta1, 0.999))

    debug_fixed_reals = next(train_loader).to(device)
    debug_fixed_reals_test = next(test_loader).to(device)

    logger = LossLogger(saved_image_folder)
    start = time()
    for iteration in tqdm(range(args.n_iterations + 1)):
        real_images = next(train_loader).to(device)

        real_images = DiffAugment(real_images, policy=policy)

        Encoder.zero_grad()
        Decoder.zero_grad()

        codes = Encoder(real_images)
        reconstructions = Decoder(codes)
        rec_loss = criteria(real_images, reconstructions)
        prior_loss = swd(codes, torch.randn_like(codes))
        loss = rec_loss + prior_loss

        loss.backward()
        optimizer.step()

        logger.aggregate_data({"Loss": loss.item(), "Prior loss": prior_loss.item(), "Reconstruction loss": rec_loss.item()})
        if iteration % 100 == 0:
            sec_per_kimage = (time() - start) / (max(1, iteration) / 1000)
            print(f"loss: {loss.item():.5f}, sec/kimg: {sec_per_kimage:.1f}")

        if iteration % (args.save_interval) == 0:
            with torch.no_grad():
                if iteration == 0:
                    vutils.save_image(debug_fixed_reals.add(1).mul(0.5),f"{saved_image_folder}/Fixed_train_batch.jpg", nrow=4)
                    vutils.save_image(debug_fixed_reals_test.add(1).mul(0.5),f"{saved_image_folder}/Fixed_test_batch.jpg", nrow=4)
                train_reconstructions = Decoder(Encoder(debug_fixed_reals))
                test_reconstructions = Decoder(Encoder(debug_fixed_reals_test))
                train_loss = criteria(train_reconstructions, debug_fixed_reals).item()
                test_loss = criteria(test_reconstructions, debug_fixed_reals_test).item()
                x = train_reconstructions.cpu()
                y = test_reconstructions.cpu()
                logger.add_data({"train_loss": train_loss,
                                         "test_loss": test_loss,
                                        'fid_loss': fid_loss(x, y).item(),
                                       'swd': swd(x, y).item(),
                                       'patch_swd': patch_swd(x, y).item(),
                                       'lap_swd': lap_swd(x, y).item()
                                        })

                vutils.save_image(train_reconstructions.add(1).mul(0.5), f"{saved_image_folder}/train_reconstructions-{iteration}.jpg", nrow=4)
                vutils.save_image(test_reconstructions.add(1).mul(0.5), f"{saved_image_folder}/test_reconstructions-{iteration}.jpg", nrow=4)
                samples = Decoder(torch.randn((args.batch_size, args.z_dim)).to(device))
                vutils.save_image(samples.add(1).mul(0.5), f"{saved_image_folder}/samples-{iteration}.jpg", nrow=4)


                logger.plot({"Eval": ["train_loss", "test_loss"]})

            torch.save({'E': Encoder.state_dict(), 'D': Decoder.state_dict()}, saved_model_folder + '/%d.pth' % iteration)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    from config import args

    args.name = "AE-FASTGAN+skip-L2+Latent-SWD_" + args.name

    # criteria = VGGPerceptualLoss().to(device)
    criteria = L2_loss

    saved_model_folder, saved_image_folder = get_dir(args)

    train_loader, test_loader = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers)

    train_AE(args)



