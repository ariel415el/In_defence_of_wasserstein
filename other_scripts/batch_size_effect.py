import argparse
import os
import sys
from collections import defaultdict

import torch
from matplotlib import pyplot as plt

from ot_means import ot_means, weisfeld_minimization
from experiment_utils import get_data
from utils.common import dump_images

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses import get_loss_function

COLORS =['r', 'g', 'b', 'k']


def main():
    """Compare two batches of real images. Plot the distance as a function of the size of the first batch.
    """
    os.makedirs(output_dir, exist_ok=True)

    named_batches = dict()
    for bs in args.batch_sizes:
        named_batches[bs] = {
                        "Real": real_batch1[:bs],
                        "Means": torch.mean(ref_data, dim=0, keepdim=True).repeat(bs, 1, 1, 1),
                        "OTMeans": ot_means(ref_data.clone(), bs, n_iters=4).reshape(-1, *data.shape[1:])
                    }
        dump_images(named_batches[bs]["Real"], os.path.join(output_dir, f"Real-{bs}.png"))
        dump_images(named_batches[bs]["Means"], os.path.join(output_dir, f"Means-{bs}.png"))
        dump_images(named_batches[bs]["OTMeans"], os.path.join(output_dir, f"OTMeans-{bs}.png"))

    for metric_name, metric in metrics.items():
        distances = defaultdict(list)

        for bs in args.batch_sizes:

            for name, batch in named_batches[bs].items():
                dist = metric(batch, ref_data)
                print(metric_name, bs, name, dist.item())
                distances[name].append(dist)

        plot(metric_name, distances, args.batch_sizes)


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
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ/FFHQ",
                        help="Path to train images")
    parser.add_argument('--center_crop', default=None, type=int)
    parser.add_argument('--limit_data', default=10000, type=int)
    parser.add_argument('--gray_scale', action='store_true', default=False)
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--batch_sizes', nargs='+', default=[10, 100, 500, 1000], type=int)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "batch_size_effect", os.path.basename(args.data_path))
    device = torch.device('cpu')

    data = get_data(args.data_path, args.im_size, gray_scale=args.gray_scale,
                    limit_data=args.limit_data, center_crop=args.center_crop)


    max_bs = args.batch_sizes[-1]

    ref_data = data[max_bs:]
    real_batch1 = data[:max_bs]

    metric_names = ['MiniBatchLoss-dist=w1',
                    'MiniBatchPatchLoss-dist=swd-p=8-s=4',
    ]

    metrics = {name: get_loss_function(name) for name in metric_names}

    main()