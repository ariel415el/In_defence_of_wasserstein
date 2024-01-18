import os
import sys
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_utils import get_data, batch_to_image, read_grid_batch, find_last_file
from losses.batch_losses import MiniBatchLoss, MiniBatchPatchLoss


if __name__ == '__main__':
    device = torch.device('cpu')
    b = 64
    im_size=64
    s = 5
    p = 8
    stride=4
    dists = ['w1', 'discrete_dual', 'swd']

    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'
    c=3
    data = get_data(data_path, im_size, c=c, center_crop=80, gray_scale=False, flatten=False, limit_data=10064).to(device)
    data = data[:-b]
    real_batch = data[-b:]

    names_and_paths = [
        ("GAN", 'outputs/GANs/FFHQ_I-64x64_G-DCGAN-normalize=none-nf=128_D-DCGAN-normalize=none-nf=128_CC-80_L-SoftHingeLoss_Z-64xnormal_B-64-64_test/images'),
        ("W1", 'outputs/TMP-Figure_test_GAN_patch_loss_smaller_priors/FFHQ_I-64x64_G-FC-depth=5-nf=1024_D-FC-depth=5-nf=4096_CC-80_L-WGANLoss_Z-64xconst=256_B-64-64_FC-FC-FFHQ_09-01_T-14:48:20/images')
    ]

    # names_and_paths = [
    #     ("FC-FC", 'outputs/tmp2/FFHQ_I-64x64_G-FC-depth=5-nf=1024_D-FC-depth=5-nf=1024_GS_CC-80_L-WGANLoss_Z-64xconst=512_B-64-64_TMP-FC-FC-depth=5-nf=1024_08-31_T-11:30:11/images'),
    #     ("FC-DC", 'outputs/tmp2/FFHQ_I-64x64_G-FC-depth=5-nf=1024_D-DCGAN-normalize=none_GS_CC-80_L-WGANLoss_Z-64xconst=512_B-64-64_TMP-FC-FC-depth=5-nf=1024_08-31_T-09:23:01/images')
    # ]
    # names_and_paths = [
    #     ("FC-DC", 'outputs/TMP-Figure_test_GAN_patch_lossnormal_prior/FFHQ_I-64x64_G-FC-depth=5-nf=1024_D-DCGAN-normalize=none_CC-80_L-WGANLoss_Z-64xconst=1024_B-64-64_FC-DC-FFHQ_08-31_T-19:02:05/images'),
    #     ("FC-FC", 'outputs/TMP-Figure_test_GAN_patch_lossnormal_prior/FFHQ_I-64x64_G-FC-depth=5-nf=1024_D-FC-depth=5-nf=1024_CC-80_L-WGANLoss_Z-64xconst=1024_B-64-64_FC-FC-FFHQ_08-31_T-19:02:09/images')
    # ]

    names_and_batchs = [("Real", real_batch)]
    for name, path in names_and_paths:
        if not os.path.isfile(path):
            path = find_last_file(path)
        names_and_batchs += [(name,  read_grid_batch(path, im_size, c, flatten=False).to(device)) ]

    fig, ax = plt.subplots(nrows=1, ncols=len(names_and_batchs), figsize=(len(names_and_batchs)*s, s))
    for i, (name, batch) in enumerate(names_and_batchs):
        title = name
        for dist in dists:

            image_dist = MiniBatchLoss(dist)
            patch_dist = MiniBatchPatchLoss(dist, p, stride)
            title += f"\n{dist}: Image:{image_dist(batch, data).item():.3f}, Patch: {patch_dist(batch, data).item():.3f}"

        ax[i].imshow(batch_to_image(batch, im_size, c))
        ax[i].axis('off')
        ax[i].set_title(title)

    # plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    plt.savefig(os.path.join(out_dir, "compare_train_debug_images.png"))
