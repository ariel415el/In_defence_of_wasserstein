import os
import sys
from collections import defaultdict
from random import shuffle

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses import get_loss_function
from scripts.experiment_utils import get_data

COLORS =['r', 'g', 'b', 'k']


def blur_batch(batch, sigma):
    if sigma > 0:
        batch = transforms.GaussianBlur(kernel_size=9, sigma=sigma)(batch)
    return batch


def main(b1, b2, metrics):
    losses = defaultdict(lambda: list())
    sigmas = [0, 0.1, 1, 10, 100, 1000]
    fig, ax = plt.subplots(1, len(sigmas), figsize=(len(sigmas)* 3, 1* 3))
    for i, sigma in enumerate(sigmas):

        b2 = blur_batch(b2.clone(), sigma)
        debug_img = make_grid(b2[:9], nrow=3)
        ax[i].imshow((1+debug_img.permute(1, 2, 0).cpu().numpy())/2)
        ax[i].set_title(f'Sigma={sigma}')
        ax[i].set_axis_off()
        for name, metric in metrics.items():
            print(sigma, name)
            losses[name].append(metric(b1, b2))
    plt.tight_layout()
    plt.savefig("scripts/outputs_/test_blur_batches.png")
    plt.clf()
    plt.figure()
    fig, ax = plt.subplots(1, len(metrics), figsize=(len(metrics)* 3, 1* 3))
    for i, (metric_name, values) in enumerate(metrics.items()):
        dists = np.array(losses[metric_name])
        if normalize:
            dists = dists / dists.max()
        ax[i].plot(sigmas, dists)#, label=metric_name)
        # ax[i].legend()
        ax[i].set_title(metric_name, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"scripts/outputs_/test_blur.png")


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "blur_dists")
    device = torch.device('cpu')
    batch_size = 1024
    im_size = 64
    normalize = False
    data_path = '../../data/FFHQ/FFHQ'
    c = 3
    center_crop = 90
    data = get_data(data_path, im_size, c=c, center_crop=center_crop, flatten=False, limit_data=2*batch_size).to(device)

    # data = data[torch.randperm(len(data))]
    b1 = data[batch_size:]
    b2 = data[:batch_size]
    metric_names = [
                    'MiniBatchLoss-dist=w1',
                    'MiniBatchLoss-dist=full_dim_swd-num_proj=128',
                    'MiniBatchLoss-dist=swd',
                    'MiniBatchLoss-dist=projected_w1-num_proj=8-dim=1',
                    'MiniBatchLoss-dist=projected_w1-num_proj=8-dim=8',
                    # 'MiniBatchNeuralLoss-dist=w1',
                    # 'MiniBatchNeuralLoss-dist=fd',
                    # 'MiniBatchPatchLoss-dist=w1-p=16-s=8',
                    # 'MiniBatchPatchLoss-dist=swd-p=16-s=8',
                    # 'MiniBatchPatchLoss-dist=fd-p=8-s=8',
                    # 'MiniBatchNeuralPatchLoss-dist=fd-device=cuda:0-b=1024',
                    # 'MiniBatchNeuralPatchLoss-dist=swd',
                    # 'MiniBatchNeuralPatchLoss-dist=w1',
    ]

    metrics = {name: get_loss_function(name) for name in metric_names}

    main(b1, b2, metrics)
