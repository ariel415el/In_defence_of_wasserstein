import os
import sys
from collections import defaultdict

import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.common import parse_classnames_and_kwargs
from losses.batch_losses import MiniBatchPatchLoss
from scripts.experiment_utils import get_data, get_centroids, batch_to_image

COLORS =['r', 'g', 'b', 'k']


def main():
    """Compare different batches to a real batch with different patch distribution metrics
    and plot the effect of the patch size
    """
    os.makedirs(output_dir, exist_ok=True)
    data = get_data(data_path, im_size, c=c, center_crop=center_crop, gray_scale=gray_scale, flatten=False, limit_data=n_data+2*b).to(device)
    r1 = data[:b]
    r2 = data[b:2*b]
    data = data[2*b:]

    # patch_sizes = [3,5,7,9,11,16,22, 32, im_size]
    patch_sizes = [3, 7, 11, im_size]

    named_batches = [
        ("Real", r1),
        ("Means", torch.mean(data, dim=0, keepdim=True).repeat(b,1,1,1)),
        ("K-Means", get_centroids(data, b, use_faiss=True))
    ]
    results = defaultdict(lambda: defaultdict(list))
    for metric_name in metrics:
        metric, kwargs = parse_classnames_and_kwargs(metric_name)
        for p in patch_sizes:
            for batch_name, batch in named_batches:
                print(metric_name, p)
                dist = MiniBatchPatchLoss(dist=metric, p=p, s=stride, **kwargs)(batch, r2)
                results[metric_name][batch_name].append(dist)

    plot(patch_sizes, results, named_batches)


def plot(patch_sizes, results, named_batches):
    """Compare the plots of different levels (Image/Patch) for each metric"""
    plt.figure()
    s = 4
    n = max(len(results), 3)
    fig, axes = plt.subplots(2, n, figsize=(3 * s, 2*s))
    # fig.suptitle(f"Metric: {dist}")
    for i, (name, batch) in enumerate(named_batches):
        axes[0, i].imshow(batch_to_image(batch, im_size, c))
        axes[0, i].axis('off')
        axes[0, i].set_title(name)
    for i, (metric_name, dists) in enumerate(results.items()):
        for j, (name, vals) in enumerate(dists.items()):
            axes[1, i].plot(range(len(vals)), vals, alpha=0.75, c=COLORS[j], label=name)
            axes[1, i].set_title(f"Metric: {metric_name}")
            plt.xticks(range(len(patch_sizes)), patch_sizes, rotation=0)
            plt.xlabel("patch-size")
    plt.title(f"Comparing {b} and {n_data} images with stride {stride}")
    # plt.legend()
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=n)
    plt.savefig(os.path.join(output_dir, f'patch_size_effect.png'))
    plt.clf()


if __name__ == '__main__':
    device = torch.device('cpu')
    b = 64
    n_data = 1024
    im_size = 64
    metrics = ['w1', 'swd', 'full_dim_swd-num_proj=128'] #'projected_w1-num_proj=8-dim=8', 'projected_w1-num_proj=8-dim=1', 'full_dim_swd-num_proj=128']
    # dist = 'w1'
    stride= 8

    data_path = '../../data/FFHQ/FFHQ'
    c = 1
    gray_scale = False
    center_crop = 90

    output_dir = os.path.join(os.path.dirname(__file__), "outputs_")
    main()