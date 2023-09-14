import torch
import sys
import os

from losses import get_loss_function
from utils.common import batch_generation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses.optimal_transport import MiniBatchPatchLoss


def compare_real_fake_patch_dist(netG,  prior, data, metric_names, outputs_dir=''):
    with torch.no_grad():

        fake_data = batch_generation(netG, prior, len(data), 512, data.device)
        dists = {}
        for metric_name in metric_names:
            metric = get_loss_function(metric_name)
            dists[metric_name] = metric(fake_data, data)

        if outputs_dir:
            with open(os.path.join(outputs_dir, 'OT.txt'), 'w') as f:
                f.write(f'Comparing {len(fake_data)} fake images to {len(data)} real images\n')
                for metric in metric_names:
                    f.write(f"{metric}: {dists[metric]:.4f}\n")

