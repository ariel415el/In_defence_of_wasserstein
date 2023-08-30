import os
import sys
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from losses.optimal_transport import MiniBatchPatchLoss
from scripts.experiments.OT.utils import get_data

COLORS =['r', 'g', 'b', 'k']




def main():
    """Compare different batches to a real batch with different patch distribution metrics
    and plot the effect of the patch size
    """
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        data = get_data(data_path, im_size, c=c, center_crop=center_crop, gray_scale=gray_scale, flatten=False, limit_data=b*2).to(device)
        b1 = data[:b]
        data = data[b:]

        patch_sizes = [3,5,7,9,11,16,22, 32, im_size]

        patch_dists = []

        for p in patch_sizes:
            patch_dists.append(MiniBatchPatchLoss(dist, p=p, s=stride)(b1, data))

        plot(patch_sizes, patch_dists)


def plot(patch_sizes, patch_dists):
    """Compare the plots of different levels (Image/Patch) for each metric"""
    plt.figure()
    plt.plot(range(len(patch_dists)),patch_dists, label=dist, alpha=0.75)

    plt.xticks(range(len(patch_sizes)), patch_sizes, rotation=0)
    plt.xlabel("patch-size")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'Plot.png'))
    plt.clf()


if __name__ == '__main__':
    device = torch.device('cpu')
    b = 64
    im_size = 64
    size = 5
    dist = 'swd'
    stride=1


    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ'
    c = 3
    gray_scale = False
    center_crop = 80

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "patch_size_effect")

    main()