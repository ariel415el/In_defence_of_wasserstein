import os
import sys
import torch
from matplotlib import pyplot as plt

from utils.common import parse_classnames_and_kwargs

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from losses.optimal_transport import MiniBatchPatchLoss
from scripts.experiments.experiment_utils import get_data, get_centroids, batch_to_image

COLORS =['r', 'g', 'b', 'k']


def main():
    """Compare different batches to a real batch with different patch distribution metrics
    and plot the effect of the patch size
    """
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        data = get_data(data_path, im_size, c=c, center_crop=center_crop, gray_scale=gray_scale, flatten=False, limit_data=n_data+b).to(device)
        r1 = data[:b]
        data = data[b:]
        r2 = data[b:2*b]

        # patch_sizes = [3,5,7,9,11,16,22, 32, im_size]
        patch_sizes = [3,7,11,22, im_size]

        named_batches = [
            ("Real", r1),
            ("Means", torch.mean(data, dim=0, keepdim=True).repeat(b,1,1,1)),
            ("K-Means", get_centroids(data, b, use_faiss=True))
        ]
        dists = {name: [] for name, _ in named_batches}

        for p in patch_sizes:
            for name, batch in named_batches:
                print(name, p)
                dist_name, kwargs = parse_classnames_and_kwargs(dist)
                dists[name].append(MiniBatchPatchLoss(dist_name, p=p, s=stride, **kwargs)(batch, r2))

        plot(patch_sizes, dists, named_batches)


def plot(patch_sizes, dists, named_batches):
    """Compare the plots of different levels (Image/Patch) for each metric"""
    plt.figure()
    s = 4
    fig, axes = plt.subplots(1, 1 + len(dists), figsize=(4 * s, 1*s))
    fig.suptitle(f"Metric: {dist}")
    for i, (name, batch) in enumerate(named_batches):
        axes[i].imshow(batch_to_image(batch, im_size, c))
        axes[i].axis('off')
        axes[i].set_title(name)
    for j, (name, vals) in enumerate(dists.items()):
        axes[-1].plot(range(len(vals)), vals, alpha=0.75, c=COLORS[j], label=name)
        plt.xticks(range(len(patch_sizes)), patch_sizes, rotation=0)
        plt.xlabel("patch-size")
    plt.title(f"Comparing {b} and {n_data} images with stride {stride}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'patch_size_effect-{dist}.png'))
    plt.clf()


if __name__ == '__main__':
    device = torch.device('cpu')
    b = 64
    n_data = 10000
    im_size = 64
    dist = 'swd'
    # dist = 'projected_w1-num_proj=8-dim=8'
    # dist = 'w1'
    size = 8
    stride= 8

    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ'
    c = 3
    gray_scale = False
    center_crop = 80

    output_dir = os.path.join(os.path.dirname(__file__), "outputs_")

    main()