from time import time

import torch
from torch import optim
from torch import nn

from models import get_generator
from utils.common import dump_images, compose_experiment_name
from utils.train_utils import copy_G_params, load_params, parse_train_args, save_model
from losses import get_loss_function
from utils.data import get_dataloader
from utils.logger import get_dir, PLTLogger, WandbLogger


class LatentCodesDict(nn.Module):
    def __init__(self, nz, n):
        super(LatentCodesDict, self).__init__()
        self.n = n
        self.emb = nn.Embedding(self.n, nz)
        self.nz = nz
        torch.nn.init.normal_(self.emb.weight, mean=0, std=0.01)

    def force_norm(self):
        wn = self.emb.weight.norm(2, 1).data.unsqueeze(1)
        self.emb.weight.data = self.emb.weight.data.div(wn.expand_as(self.emb.weight.data))

    def forward(self, idx):
        z = self.emb(idx).squeeze()
        return z


def train_GLO(args):
    logger = (WandbLogger if args.wandb else PLTLogger)(args, plots_image_folder)

    netG = get_generator(args, device)
    netG.train()
    latent_codes = LatentCodesDict(args.z_dim, len(train_loader.dataset)).to(device)

    optimizerZ = optim.Adam(latent_codes.parameters(), lr=args.lrZ, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(0.5, 0.9))

    debug_fixed_reals, debug_fixed_indices = next(iter(full_batch_loader))
    debug_fixed_reals = debug_fixed_reals.to(device)
    debug_fixed_indices = debug_fixed_indices.to(device)

    other_metrics = [
                get_loss_function("MiniBatchLoss-dist=w1"),
                get_loss_function("MiniBatchPatchLoss-dist=swd-p=16-s=8"),
                get_loss_function("MiniBatchLocalPatchLoss-dist=swd-p=16-s=8"),
              ]

    avg_param_G = copy_G_params(netG)

    start = time()
    iteration = 0
    while iteration < args.n_iterations:
        for real_images, indices in train_loader:
            real_images = real_images.to(device)
            indices = indices.to(device)

            zs = latent_codes(indices)
            if args.noise_sigma > 0:
                # zs += (iteration/args.n_iterations)*torch.randn_like(zs).to(device)
                zs += args.noise_sigma*torch.randn_like(zs).to(device)
            fake_images = netG(zs)

            rec_loss = torch.nn.MSELoss()(fake_images, real_images)
            optimizerZ.zero_grad()
            optimizerG.zero_grad()
            rec_loss.backward()
            optimizerG.step()

            logger.log({"Reconstruction-Loss": rec_loss.item()}, iteration)
            if iteration % 1 == 0:
                optimizerZ.step()
            if args.force_norm_every > 0 and iteration % args.force_norm_every == 0:
                latent_codes.force_norm()

            # Update avg weights
            for p, avg_p in zip(netG.parameters(), avg_param_G):
                avg_p.mul_(1 - args.avg_update_factor).add_(args.avg_update_factor * p.data)

            if iteration % 100 == 0:
                it_sec = max(1, iteration) / (time() - start)
                print(f"Iteration: {iteration}: it/sec: {it_sec:.1f}")
                logger.plot()

            if iteration % args.log_freq == 0:
                backup_para = copy_G_params(netG)
                load_params(netG, avg_param_G)

                evaluate(latent_codes, netG, other_metrics, debug_fixed_reals, debug_fixed_indices, saved_image_folder, iteration, logger)

                save_model(netG, latent_codes, optimizerG, optimizerZ, saved_model_folder, iteration, args)

                load_params(netG, backup_para)

            iteration += 1

def batch_generation(latent_codes, netG, n, b, device):
    n_batches = n // b
    fake_data = []
    for i in range(n_batches):
        zs = latent_codes(torch.arange(i*b, (i+1)*b).to(device))
        fake_data.append(netG(zs))
    if n_batches * b < n:
        zs = latent_codes(torch.arange(n-n_batches * b, n).to(device))
        fake_data.append(netG(zs))
    fake_data = torch.cat(fake_data)
    return fake_data


def evaluate(latent_codes, netG, other_metrics, debug_fixed_reals, debug_fixed_indices,
             saved_image_folder, iteration, logger):
    netG.eval()
    start = time()
    with torch.no_grad():
        fake_images = batch_generation(latent_codes, netG, len(latent_codes.emb.weight), 64, device)

        print(f"Computing metrics between {len(debug_fixed_reals)} real and {len(fake_images)} fake images")
        for metric in other_metrics:
            logger.log({
                f'{metric.name}_fixed_noise_gen_to_train': metric(fake_images.cpu(), debug_fixed_reals.cpu()),
            }, step=iteration)
        dump_images(fake_images[:64],  f'{saved_image_folder}/{iteration}_recs.png')
        if iteration == 0:
            dump_images(debug_fixed_reals[:64], f'{saved_image_folder}/debug_fixed_reals.png')

    netG.train()
    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    args = parse_train_args()

    device = torch.device(args.device)
    if args.device != 'cpu':
        print(f"Working on device: {torch.cuda.get_device_name(device)}")

    train_loader, _ = get_dataloader(args.data_path, args.im_size, args.r_bs, args.n_workers,
                                               val_percentage=0, gray_scale=args.gray_scale, center_crop=args.center_crop,
                                               load_to_memory=args.load_data_to_memory, limit_data=args.limit_data)

    full_batch_loader, _ = get_dataloader(args.data_path, args.im_size, len(train_loader.dataset), args.n_workers,
                                               val_percentage=0, gray_scale=args.gray_scale, center_crop=args.center_crop,
                                               load_to_memory=args.load_data_to_memory, limit_data=args.limit_data)

    if args.train_name is None:
        args.train_name = compose_experiment_name(args)

    saved_model_folder, saved_image_folder, plots_image_folder = get_dir(args)

    train_GLO(args)



