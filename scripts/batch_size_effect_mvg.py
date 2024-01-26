import argparse
import os
import sys
from collections import defaultdict

import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.common import dump_images
from losses import get_loss_function
from scripts.experiment_utils import get_data, get_centroids
from scripts.ot_means import ot_means, weisfeld_minimization


COLORS =['r', 'g', 'b', 'k']


def main():
    """Compare two batches of real images. Plot the distance as a function of the size of the first batch.
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, metric in metrics.items():
        distances = defaultdict(list)
        for d in ds:
            data = torch.randn((batch_sizes[-1] * 2, d))
            for bs in batch_sizes:
                dist = metric(data[:bs], data[-bs:])
                print(d, metric_name, bs, dist.item())
                distances[d].append(dist)

        plot(metric_name, distances, batch_sizes)


def plot(metric_name, distances, batch_sizes):
    plt.figure()
    for d, values in distances.items():
        plt.plot(batch_sizes, values, label=f"d={d}")
        plt.annotate(f"{values[-1]:.4f}", (batch_sizes[-1], values[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")

    plt.xlabel("Batch-size")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(metric_name)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(output_dir, f'batch_size_effect-{metric_name}.png'))
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ds = [1,20,100]
    batch_sizes = [16, 128, 512, 1024, 5000, 10000]#, 20000, 35000, 70000, 140000, 300000]
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "batch_size_effect_mvg")
    device = torch.device('cpu')

    metric_names = [
        'MiniBatchLoss-dist=swd-num_proj=1024',
        # 'MiniBatchLoss-dist=w1-normalize=True',
        # 'MiniBatchLoss-dist=w1-normalize=False-epsilon=0.1',
        # 'MiniBatchPatchLoss-dist=swd-p=8-s=4',
        # 'MiniBatchNeuralLoss-dist=fd',
        # 'MiniBatchNeuralPatchLoss-dist=fd-device=cuda:0-b=1024',
        # 'MiniBatchNeuralPatchLoss-dist=swd',
    ]

    metrics = {name: get_loss_function(name) for name in metric_names}

    main()