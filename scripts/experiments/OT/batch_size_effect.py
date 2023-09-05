import os
import sys
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from losses.optimal_transport import MiniBatchPatchLoss
from scripts.experiments.experiment_utils import get_data

COLORS =['r', 'g', 'b', 'k']


def main():
    """Compare two batches of real images. Plot the distance as a function of the size of the first batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        data = get_data(data_path, im_size, c=c, center_crop=center_crop, gray_scale=gray_scale, flatten=False, limit_data=2*max_bs).to(device)
        ref_data = data[max_bs:]

        distances = []

        for bs in batch_sizes:
            print(bs)
            distances.append(MiniBatchPatchLoss(dist, p=p, s=stride)(data[:bs], ref_data))

        plot(distances, batch_sizes)


def plot(distances, batch_sizes):
    plt.figure()
    plt.plot(batch_sizes, distances, label=dist)
    plt.annotate(f"{distances[-1]:.2f}", (batch_sizes[-1], distances[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")

    plt.xlabel("Batch-size")
    plt.ylabel("dist")
    plt.legend()
    plt.title(f"{dist}-{p}-{stride}")
    plt.savefig(os.path.join(output_dir, f'batch_size_effect-{dist}.png'))
    plt.clf()


if __name__ == '__main__':
    device = torch.device('cpu')
    max_bs = 10000
    batch_sizes = [16, 32, 64, 128, 256]#, 512,  1024, max_bs//2, max_bs]
    # batch_sizes = [64, 256, 1024, max_bs]
    im_size = 64
    # dist = 'discrete_dual'
    dist = 'swd'
    size = 8
    p=8
    stride=4

    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'
    c = 3
    gray_scale = False
    center_crop = 80

    output_dir = os.path.join(os.path.dirname(__file__), "outputs_")

    main()