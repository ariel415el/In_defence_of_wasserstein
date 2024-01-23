import argparse
import os.path
import sys

from time import time

import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from models import get_models
from losses import get_loss_function
from utils.train_utils import calc_gradient_penalty
from utils.data import get_dataloader
from utils.logger import get_dir, PLTLogger


"""Train a neural discriminator to optimal differentiation between two splits of a dataset
    The final dual W1 distance can be then compared to the primal distance to test the efficacy
    of the neural dual solver.
    See 'Wasserstein GANs Work Because They Fail' Stanczuk et. al 2021 """

def get_models_and_optimizers(args):
    args.gen_arch = "FC"
    args.z_dim =  100
    _, netD = get_models(args, device)
    netD.train()

    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(0.5, 0.9))

    start_iteration = 0
    return netD, optimizerD, start_iteration


def train_dual_function(args):
    logger = PLTLogger(args, plots_image_folder)

    loss_function = get_loss_function("WGANLoss")

    netD, optimizerD, start_iteration = get_models_and_optimizers(args)

    start = time()
    iteration = 0
    while iteration < args.n_iterations :
        batches_1 = iter(loader_1)
        batches_2 = iter(loader_2)
        for _ in range(min(len(loader_1), len(loader_2))):
            batch_1 = next(batches_1).to(device)
            batch_2 = next(batches_2).to(device)

            # #####  1. train Discriminator #####
            Dloss, debug_Dlosses = loss_function.trainD(netD, batch_1, batch_2)
            if args.gp_weight > 0:
                gp, gradient_norm = calc_gradient_penalty(netD, batch_1, batch_2)
                debug_Dlosses['gradient_norm'] = gradient_norm
                Dloss += args.gp_weight * gp
                if "W1" in debug_Dlosses:
                    debug_Dlosses['normalized W1'] = (debug_Dlosses['W1'] / gradient_norm) if gradient_norm > 0 else 0
            netD.zero_grad()
            Dloss.backward()
            optimizerD.step()

            if args.weight_clipping is not None:
                for p in netD.parameters():
                    p.data.clamp_(-args.weight_clipping, args.weight_clipping)

            logger.log(debug_Dlosses, step=iteration)

            if iteration % 100 == 0:
                it_sec = max(1, iteration - start_iteration) / (time() - start)
                print(f"Iteration: {iteration}: it/sec: {it_sec:.1f}")
                logger.plot()

            if iteration > 0 and iteration % args.log_freq == 0:
                evaluate(netD, loader_1, loader_2, iteration, logger, args)

            iteration += 1


def evaluate(netD, loader_1, loader_2, iteration, logger, args):
    print("Evaluating", end="...")
    with torch.no_grad():
        netD.eval()
        start = time()
        exp_1 = 0
        exp_2 = 0
        for batch in tqdm(loader_1):
            exp_1 += netD(batch.to(device)).sum()

        for batch in tqdm(loader_2):
            exp_2 += netD(batch.to(device)).sum()

        w1 = exp_1 / len(loader_1.dataset) - exp_2 / len(loader_2.dataset)

        logger.log({'Full-W1': w1.item()}, step=iteration)
        netD.train()
    print(f"Evaluation finished in {time() - start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ_1000/FFHQ_1000",
                        help="Path to train images")
    parser.add_argument('--center_crop', default=None, help='center_crop_data to specified size', type=int)
    parser.add_argument('--gray_scale', action='store_true', default=False, help="Load data as grayscale")

    # Model
    parser.add_argument('--disc_arch', default='DCGAN')
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--spectral_normalization', action='store_true', default=False)
    parser.add_argument('--weight_clipping', type=float, default=None)

    # Training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gp_weight', default=0, type=float)
    parser.add_argument('--lrD', default=0.0001, type=float)
    parser.add_argument('--n_iterations', default=1000000, type=int)

    # Evaluation
    parser.add_argument('--log_freq', default=10000, type=int)
    # parser.add_argument('--plot_w1', action='store_true', default=False)

    # Other
    parser.add_argument('--project_name', default='Dual-solvers')
    parser.add_argument('--tag', default='test')
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--load_data_to_memory', action='store_true', default=False)
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()
    args.train_name = f"Dual_solver-{os.path.basename(args.data_path)}_{args.im_size}x{args.im_size}" \
                f"_D-{args.disc_arch}_B-{args.batch_size}_{args.tag}"

    device = torch.device(args.device)
    if args.device != 'cpu':
        print(f"Working on device: {torch.cuda.get_device_name(device)}")

    saved_model_folder, saved_image_folder, plots_image_folder = get_dir(args)

    loader_1, loader_2 = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers,
                                               val_percentage=0.5,
                                               load_to_memory=args.load_data_to_memory)

    train_dual_function(args)



