import os
import sys
from collections import defaultdict

import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.common import parse_classnames_and_kwargs
from losses.batch_losses import MiniBatchPatchLoss, MiniBatchLoss
from scripts.experiment_utils import get_data, get_centroids, batch_to_image

COLORS =['r', 'g', 'b', 'k']


def main():
    """Compare different batches to a real batch with different patch distribution metrics
    and plot the effect of the patch size
    """
    os.makedirs(output_dir, exist_ok=True)
    data = get_data(data_path, im_size, c=c, center_crop=center_crop, gray_scale=gray_scale, flatten=False, limit_data=2*b).to(device)
    r1 = data[:b]
    r2 = data[b:]

    dists = []
    for name, metric in metrics.items():
        dists.append(metric(r1,r2))
    plt.bar(range(len(metrics)), dists)
    plt.xticks(range(len(metrics)), metrics.keys())
    plt.savefig(os.path.join(output_dir, "test_w1_bound.png"))
    plt.show()


if __name__ == '__main__':
    device = torch.device('cpu')
    b = 1024
    im_size = 64
    metrics = {'w1' : MiniBatchLoss(dist='w1'),
               'upperBound-w1': MiniBatchLoss(dist='full_dim_swd')}

    data_path = '../../data/FFHQ/FFHQ'
    c = 1
    gray_scale = False
    center_crop = 90

    output_dir = os.path.join(os.path.dirname(__file__), "outputs_")
    main()