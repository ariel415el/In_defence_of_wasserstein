import os
import sys
from collections import defaultdict

import torch
from matplotlib import pyplot as plt

from utils.common import dump_images

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from losses import MiniBatchPatchLoss, MiniBatchLoss, get_loss_function
from scripts.experiments.experiment_utils import get_data, get_centroids

COLORS =['r', 'g', 'b', 'k']


def main():
    """Compare two batches of real images. Plot the distance as a function of the size of the first batch.
    """
    os.makedirs(output_dir, exist_ok=True)

    named_batches = dict()
    for bs in batch_sizes:
        named_batches[bs] = {
                        "Real": data[:bs],
                        "Means": torch.mean(ref_data, dim=0, keepdim=True).repeat(bs, 1, 1, 1),
                        "KMeans": get_centroids(ref_data, bs, use_faiss=False)
                    }
        dump_images(named_batches[bs]["Real"], os.path.join(output_dir, f"Real-{bs}.png"))
        dump_images(named_batches[bs]["Means"], os.path.join(output_dir, f"Means-{bs}.png"))
        dump_images(named_batches[bs]["KMeans"], os.path.join(output_dir, f"KMeans-{bs}.png"))

    for metric_name, metric in metrics.items():
        distances = defaultdict(list)

        for bs in batch_sizes:
            print(bs)

            for name, batch in named_batches[bs].items():
                distances[name].append(metric(batch, ref_data))

        plot(metric_name, distances, batch_sizes)


def plot(metric_name, distances, batch_sizes):
    plt.figure()
    for name, values in distances.items():
        plt.plot(batch_sizes, values, label=name)
        plt.annotate(f"{values[-1]:.2f}", (batch_sizes[-1], values[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")

    plt.xlabel("Batch-size")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(metric_name)
    plt.savefig(os.path.join(output_dir, f'batch_size_effect-{metric_name}.png'))
    plt.clf()


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "batch_size_effect")
    device = torch.device('cpu')
    batch_sizes = [10, 100, 500, 1000]#, 5000, 10000]#, 20000, 35000]
    im_size = 64

    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ'
    c = 3
    gray_scale = False
    center_crop = 80
    limit_data = 10*batch_sizes[-1]
    data = get_data(data_path, im_size, c=c, center_crop=center_crop,
                    gray_scale=gray_scale, flatten=False, limit_data=limit_data).to(device)

    n = batch_sizes[-1]
    ref_data = data[n:]
    data = data[:n]

    metric_names = ['MiniBatchLoss-dist=w1',
                    'MiniBatchLoss-dist=swd',
                    'MiniBatchPatchLoss-dist=swd-p=8-s=4',
    ]

    metrics = {name: get_loss_function(name) for name in metric_names}

    main()