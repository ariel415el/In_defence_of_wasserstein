import os
import sys
from collections import defaultdict
from random import shuffle

import torch
from matplotlib import pyplot as plt

from scripts.experiments.OT.ot_means import ot_mean, weisfeld_minimization
from utils.common import dump_images

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from losses import get_loss_function
from scripts.experiments.experiment_utils import get_data, get_centroids

COLORS =['r', 'g', 'b', 'k']


def main():
    """Compare two batches of real images. Plot the distance as a function of the size of the first batch.
    """
    os.makedirs(output_dir, exist_ok=True)

    named_batches = dict()
    for bs in batch_sizes:
        named_batches[bs] = {
                        "Real": real_batch1[:bs],
                        "Means": torch.mean(ref_data, dim=0, keepdim=True).repeat(bs, 1, 1, 1),
                        # "KMeans": get_centroids(ref_data, bs, use_faiss=False)
                        "OTMeans": ot_mean(ref_data.clone(), bs, n_iters=4, minimization_method=weisfeld_minimization, verbose=False).reshape(-1, *data.shape[1:])
                    }
        dump_images(named_batches[bs]["Real"], os.path.join(output_dir, f"Real-{bs}.png"))
        dump_images(named_batches[bs]["Means"], os.path.join(output_dir, f"Means-{bs}.png"))
        # dump_images(named_batches[bs]["KMeans"], os.path.join(output_dir, f"KMeans-{bs}.png"))
        dump_images(named_batches[bs]["OTMeans"], os.path.join(output_dir, f"OTMeans-{bs}.png"))

    for metric_name, metric in metrics.items():
        distances = defaultdict(list)

        for bs in batch_sizes:

            for name, batch in named_batches[bs].items():
                dist = metric(batch, ref_data)
                print(metric_name, bs, name, dist)
                distances[name].append(dist)

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
    batch_sizes = [10, 100, 500, 1000]#, 20000, 35000]
    im_size = 64
    n = batch_sizes[-1]

    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ'
    c = 3
    gray_scale = False
    center_crop = 80
    limit_data = 10000 + 1 * n
    data = get_data(data_path, im_size, c=c, center_crop=center_crop,
                    gray_scale=gray_scale, flatten=False, limit_data=limit_data).to(device)

    shuffle(data)
    ref_data = data[n:]
    real_batch1 = data[:n]
    # real_batch2 = data[n:2*n]

    metric_names = ['MiniBatchLoss-dist=w1',
                    # 'MiniBatchLoss-dist=swd',
                    # 'MiniBatchLocalPatchLoss-dist=w1-p=8-s=4-n_samples=2000',
                    # 'MiniBatchLocalPatchLoss-dist=swd-p=8-s=4',
                    'MiniBatchPatchLoss-dist=swd-p=8-s=4',
    ]

    metrics = {name: get_loss_function(name) for name in metric_names}

    main()