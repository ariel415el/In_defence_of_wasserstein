import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.common import dump_images
from losses import get_loss_function
from scripts.experiment_utils import get_data, get_centroids
from scripts.ot_means import ot_means, weisfeld_minimization, sgd_minimization


COLORS =['r', 'g', 'b', 'k']


def blur_batch(batch, kernel_size, sigma):
    if sigma > 0:
        batch = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(batch)
    return batch


def sample(data, n):
    return data[np.random.randint(len(data), size=n)]

def sample_mean(data, n, k):
    return torch.stack([torch.mean(data[np.random.randint(len(data), size=k)], dim=0) for _ in range(n)])


def main():
    """Compare two batches of real images. Plot the distance as a function of the size of the first batch.
    """
    os.makedirs(output_dir, exist_ok=True)

    named_batches = dict()
    for bs in args.batch_sizes:
        named_batches[bs] = {
                        "Real": sample(train_split, bs),
                        # "OTMeans": sample(cetnroids, bs),
                        # "combMeans": sample_mean(train_split, bs, 10),
                        "Blur": blur_batch(sample(train_split, bs), kernel_size=15, sigma=30),
                        # "Means": torch.mean(train_split, dim=0, keepdim=True).repeat(bs, 1, 1, 1),
                        # "OTMeans": ot_means(train_split.clone(), bs, n_iters=4, minimization_method=sgd_minimization).reshape(-1, *data.shape[1:])
                    }
        dump_images(named_batches[bs]["Real"][:9], os.path.join(output_dir, f"Real-{bs}.png"))
        # dump_images(named_batches[bs]["OTMeans"][:9], os.path.join(output_dir, f"OTMeans-{bs}.png"))
        dump_images(named_batches[bs]["Blur"][:9], os.path.join(output_dir, f"Blur-{bs}.png"))
        # dump_images(named_batches[bs]["Means"], os.path.join(output_dir, f"Means-{bs}.png"))
        # dump_images(named_batches[bs]["combMeans"][:9], os.path.join(output_dir, f"CombMeans-{bs}.png"))
        # dump_images(named_batches[bs]["OTMeans"][:9], os.path.join(output_dir, f"OTMeans-{bs}.png"))

    for metric_name, metric in metrics.items():
        distances = defaultdict(list)

        for bs in args.batch_sizes:
            for name, batch in named_batches[bs].items():
                dist = metric(batch, test_split[:bs])
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
    parser.add_argument('--data_path', default="../../data/FFHQ/FFHQ",
                        help="Path to train images")
    parser.add_argument('--center_crop', default=None, type=int)
    parser.add_argument('--gray_scale', action='store_true', default=False)
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--batch_sizes', nargs='+', default=[16, 128, 512, 1024, 5000], type=int)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "batch_size_effect_blur", os.path.basename(args.data_path))
    device = torch.device('cpu')

    max_bs = args.batch_sizes[-1]

    data = get_data(args.data_path, args.im_size, gray_scale=args.gray_scale, flatten=False,
                    limit_data=2*max_bs, center_crop=args.center_crop)

    test_split = data[:max_bs]
    train_split = data[max_bs:]

    metric_names = [
        'MiniBatchLoss-dist=w1',
        'MiniBatchLoss-dist=full_dim_swd',
        # 'MiniBatchLoss-dist=swd',
        # 'MiniBatchPatchLoss-dist=swd-p=16-s=8',
        # 'MiniBatchNeuralLoss-dist=fd',
        # 'MiniBatchNeuralPatchLoss-dist=fd-device=cuda:0-b=1024',
        # 'MiniBatchNeuralPatchLoss-dist=swd',
    ]

    metrics = {name: get_loss_function(name) for name in metric_names}

    # cetnroids = ot_means(train_split.clone(), max_bs, n_iters=4, minimization_method=sgd_minimization).reshape(-1, *data.shape[1:])

    main()