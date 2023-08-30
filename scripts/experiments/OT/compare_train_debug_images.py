import os
import sys
import torch
from matplotlib import pyplot as plt

from utils import get_data, batch_to_image, read_grid_batch
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from losses.optimal_transport import MiniBatchLoss, MiniBatchPatchLoss


if __name__ == '__main__':
    device = torch.device('cpu')
    b = 64
    im_size=64
    s = 5
    p =  7
    stride=7
    dists = ['remd', 'swd']

    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ'
    c=3
    data = get_data(data_path, im_size, c=c, center_crop=80, gray_scale=False, flatten=False, limit_data=1064).to(device)
    real_batch = data[:b]
    data = data[b:]

    names_and_paths = [
        ("wgan", 'outputs/GoodGANS/FFHQ_I-64x64_G-DCGAN-normalize=none_D-DCGAN-normalize=none_CC-80_L-SoftHingeLoss_Z-64xnormal_B-64-64_test/images/121000.png')
    ]

    names_and_batchs = [("Real", real_batch)] + [(name,  read_grid_batch(path, im_size, c, flatten=False).to(device)) for name, path in names_and_paths]

    fig, ax = plt.subplots(nrows=1, ncols=len(names_and_batchs), figsize=(len(names_and_batchs)*s, s))
    for i, (name, batch) in enumerate(names_and_batchs):
        title = name
        for dist in dists:

            image_dist = MiniBatchLoss(dist)
            patch_dist = MiniBatchPatchLoss(dist, p, stride)
            title += f"\n Image:{image_dist(batch, data).item():.3f}, Patch: {patch_dist(batch, data).item():.3f}"

        ax[i].imshow(batch_to_image(batch, im_size, c))
        ax[i].axis('off')
        ax[i].set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "compare_train_debug_images.png"))
