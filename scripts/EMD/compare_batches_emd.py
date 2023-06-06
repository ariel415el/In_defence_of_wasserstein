from collections import defaultdict


import os
import torch
from matplotlib import pyplot as plt

from scripts.EMD.dists import discrete_dual, swd
from scripts.EMD.dists import emd

from utils import get_data, dump_images, batch_to_image, to_patches, read_grid_batch

if __name__ == '__main__':
    # data_path = '/cs/labs/yweiss/ariel1/data/square_data/7x7'; c=1
    # FC_output_dir = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/WGAN_yweiss/7x7_64x64_G-pixels_D-FC-normalize=none_L-WGANLoss_Z-64_B-64_pixels-FC_7x7_06-04_T-18:47:17'
    # DC_output_dir = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/WGAN_yweiss/7x7_64x64_G-pixels_D-DCGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-DC_7x7_06-04_T-18:47:19'
    # GAP_output_dir = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/WGAN_yweiss/7x7_64x64_G-pixels_D-PatchGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-PatchDisc_7x7_06-04_T-18:47:21'

    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ_128/FFHQ_128'; c = 3
    batches = {
    #     "RealSanity": '/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-DCGAN_L-BatchEMD_Z-64_B-64_test/images/debug_fixed_reals.png',
        "EMD": "/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-DCGAN_L-BatchEMD_Z-64_B-64_test/images/8000.png",
    #     "EMD-5-1": "/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-DCGAN_L-BatchPatchEMD-p=5-s=1_Z-64_B-64_test/images/55000.png",
        "FC": '/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-FC-normalize=none_L-WGANLoss_Z-64_B-64_pixels-FC_FFHQ_128_06-04_T-18:47:41/images/100000.png',
        "DC": '/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-DCGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-DC_FFHQ_128_06-04_T-18:47:44/images/100000.png',
    #     "GAP": '/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-PatchGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-PatchDisc_FFHQ_128_06-04_T-18:47:46/images/100000.png'
    }

    outdir = os.path.join("output_emd", os.path.basename(data_path))
    os.makedirs(outdir, exist_ok=True)
    # metric_name, metric = "EMD", emd
    metric_name, metric = "SWD", swd
    # metric_name, metric = "Dual", lambda x,y: discrete_dual(x,y, verbose=True)
    d = 64
    b = 64
    p, s = 7, 7
    device = torch.device('cuda:0')

    data = get_data(data_path, d, c, limit_data=10000 + 2 * b).to(device)

    r1 = data[:b]
    r2 = data[b:2*b]
    data = data[2*b:]

    for k in batches:
        batches[k] = read_grid_batch(batches[k], d, c).to(device)
    batches['r2'] = r2
    # batches['centroids'] = get_centroids(data, b, use_faiss=True)

    fig, ax = plt.subplots(nrows=2, ncols=len(batches) + 1, figsize=(15,5))
    ax[1,0].imshow(batch_to_image(r1, d, c))
    ax[1,0].axis('off')
    ax[0,0].axis('off')
    ax[1,0].set_title(f'Image-{metric_name}\nPatch-{metric_name}-{p}-{s}')
    for i, (name, batch) in enumerate(batches.items()):
        dump_images(batch, b, d, c, f"{outdir}/{name}-{b}.png")
        image_dist = metric(batch, r1)
        x = to_patches(batch, d, c, p, s)
        y = to_patches(r1, d, c, p, s)
        patch_dist = metric(x,y)

        ax[0, 1+i].imshow(batch_to_image(batch, d, c))
        ax[0, 1+i].axis('off')
        ax[0, 1+i].set_title(name)
        ax[1, 1+i].axis('off')
        # ax[1, 1+i].set_title(f'{metric.name}-{name}: {image_dist:3f}\nPatch-{metric.name}-{p}-{s}-{name}: {patch_dist:3f}', size=12)
        ax[1, 1+i].set_title(f'{image_dist:3f}\n{patch_dist:3f}', size=12)


    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{metric_name}.png"))



