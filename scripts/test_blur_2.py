import os
import sys
from collections import defaultdict
from random import shuffle

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from losses import get_loss_function
from scripts.experiments.experiment_utils import get_data

COLORS =['r', 'g', 'b', 'k']


def blur_batch(batch, sigma):
    if sigma > 0:
        batch = transforms.GaussianBlur(kernel_size=5, sigma=sigma)(batch)
    return batch


def main(b1, b2, metrics):
    losses = defaultdict(lambda: list())
    sigmas = [0, 0.1, 1, 10]
    fig, ax = plt.subplots(len(sigmas))
    for i, sigma in enumerate(sigmas):
        b2 = blur_batch(b2, sigma)
        ax[i].imshow((1+b2[0].permute(1, 2, 0).cpu().numpy())/2)
        for name, metric in metrics.items():
            print(sigma, name)
            losses[name].append(metric(b1, b2))
    plt.show()
    plt.tight_layout()
    plt.clf()
    for metric_name, values in metrics.items():
        dists = np.array(losses[metric_name])
        dists = dists / dists.max()
        plt.plot(sigmas, dists, label=metric_name)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "blur_dists")
    device = torch.device('cpu')
    batch_size = 10000
    im_size = 64

    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ_128'
    c = 3
    center_crop = 90
    data = get_data(data_path, im_size, c=c, center_crop=center_crop, flatten=False, limit_data=2*batch_size).to(device)

    shuffle(data)
    b1 = data[batch_size:]
    b2 = data[:batch_size]

    metric_names = [
                    # 'MiniBatchLoss-dist=w1',
                    # 'MiniBatchLoss-dist=swd',
                    # 'MiniBatchNeuralLoss-dist=w1',
                    'MiniBatchNeuralLoss-dist=fd',
                    # 'MiniBatchPatchLoss-dist=w1-p=16-s=8',
                    # 'MiniBatchPatchLoss-dist=swd-p=16-s=8',
                    # 'MiniBatchPatchLoss-dist=fd-p=8-s=8',
                    'MiniBatchNeuralPatchLoss-dist=fd-device=cuda:0-b=1024',
                    'MiniBatchNeuralPatchLoss-dist=swd',
                    # 'MiniBatchNeuralPatchLoss-dist=w1',
    ]

    metrics = {name: get_loss_function(name) for name in metric_names}

    main(b1, b2, metrics)