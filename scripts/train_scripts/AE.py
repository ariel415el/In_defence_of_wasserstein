
from time import time

import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmarking.neural_metrics import InceptionMetrics
from utils.diffaug import DiffAugment
from utils.common import dump_images, compose_experiment_name
from utils.train_utils import copy_G_params, load_params, get_models_and_optimizers, parse_train_args, save_model
from losses import get_loss_function
from utils.data import get_dataloader
from utils.logger import get_dir, PLTLogger, WandbLogger


def train_GAN(args):
    logger = (WandbLogger if args.wandb else PLTLogger)(args, plots_image_folder)

    prior, netG, netD, optimizerG, optimizerD, start_iteration = get_models_and_optimizers(args, device, saved_model_folder)

    debug_fixed_noise = prior.sample(args.f_bs).to(device)
    debug_fixed_reals = next(iter(train_loader)).to(device)
    debug_all_reals = next(iter(full_batch_loader)).to(device)

    inception_metrics = InceptionMetrics([next(iter(train_loader)) for _ in range(args.fid_n_batches)], torch.device("cpu"))
    other_metrics = [
                get_loss_function("MiniBatchLoss-dist=w1"),
                get_loss_function("MiniBatchLoss-dist=nn"),
              ]

    loss_function = torch.nn.MSELoss()

    avg_param_G = copy_G_params(netG)

    start = time()
    iteration = start_iteration
    while iteration < args.n_iterations:
        for real_images in train_loader:
            real_images = real_images.to(device)
            recons = netG(netD(real_images))
            loss = loss_function(real_images, recons)

            netD.zero_grad()
            netG.zero_grad()
            loss.backward()
            optimizerD.step()
            optimizerG.step()

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
                         debug_fixed_reals, debug_all_reals, saved_image_folder, iteration, logger, args)

                save_model(prior, netG, netD, optimizerG, optimizerD, saved_model_folder, iteration, args)

                load_params(netG, backup_para)

            iteration += 1


def evaluate(netG, netD, inception_metrics, other_metrics, fixed_noise, debug_fixed_reals,
             debug_all_reals, saved_image_folder, iteration, logger, args):
    netG.eval()
    netD.eval()
    start = time()
    with torch.no_grad():
        if args.D_step_every > 0 :
            recons = netG(netD(debug_fixed_reals))


        if args.fid_n_batches > 0 and iteration % args.fid_freq == 0:
            fake_batches = [netG(torch.randn_like(fixed_noise).to(device)) for _ in range(args.fid_n_batches)]
            logger.log(inception_metrics(fake_batches), step=iteration)

        for metric in other_metrics:
            logger.log({
                f'{metric.name}_fixed_noise_gen_to_train': metric(recons.cpu(), debug_all_reals.cpu()),
            }, step=iteration)


        dump_images(recons,  f'{saved_image_folder}/{iteration}.png')
        if iteration == 0:
            dump_images(debug_fixed_reals, f'{saved_image_folder}/debug_fixed_reals.png')


    netG.train()
    netD.train()

    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    args = parse_train_args()

    device = torch.device(args.device)
    if args.device != 'cpu':
        print(f"Working on device: {torch.cuda.get_device_name(device)}")

    train_loader, _ = get_dataloader(args.data_path, args.im_size, args.r_bs, args.n_workers,
                                               val_percentage=0, gray_scale=args.gray_scale, center_crop=args.center_crop,
                                               load_to_memory=args.load_data_to_memory, limit_data=args.limit_data)

    data_size = len(train_loader.dataset)
    print(f"eval loader size {data_size}")
    full_batch_loader, _ = get_dataloader(args.data_path, args.im_size, data_size, args.n_workers,
                                               val_percentage=0, gray_scale=args.gray_scale, center_crop=args.center_crop,
                                               load_to_memory=args.load_data_to_memory, limit_data=len(train_loader.dataset))
    full_batch_loader = train_loader
    if args.r_bs == -1:
        args.r_bs = data_size

    args.name = compose_experiment_name(args)

    saved_model_folder, saved_image_folder, plots_image_folder = get_dir(args)

    train_GAN(args)



