import os
import sys

import torch
from matplotlib import pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.EMD.dists import discrete_dual, swd, emd

from utils import get_data, dump_images, batch_to_image, to_patches, read_grid_batch

if __name__ == '__main__':
    device = torch.device('cpu')
    c = 3
    b = 64
    p, s = 16, 8
    metric_name, metric = "EMD", emd
    # metric_name, metric = "SWD", swd
    # metric_name, metric = "Dual", lambda x,y: discrete_dual(x,y, verbose=False, nnb=16)
    for data_path, batches_dir, d in [
        ('/mnt/storage_ssd/datasets/FFHQ/FFHQ_128/FFHQ_128', '/home/ariel/Downloads/outputs/FFHQ_64',64),
        # ('/mnt/storage_ssd/datasets/FFHQ/FFHQ_128/FFHQ_128', '/home/ariel/Downloads/outputs/FFHQ_128', 128),
        # ('/mnt/storage_ssd/datasets/square_data/7x7', '/home/ariel/Downloads/outputs/squares', 64),
        # ('/mnt/storage_ssd/datasets/MNIST/floating_MNIST/train-128-0', '/home/ariel/Downloads/outputs/floating_mnist_128', 128),
        # ('/mnt/storage_ssd/datasets/MNIST/MNIST/jpgs/training', '/home/ariel/Downloads/outputs/mnist',64),
    ]:
        data = get_data(data_path, d, c, limit_data=1000 + 2 * b).to(device)

        r1 = data[:b]
        r2 = data[b:2*b]
        data = data[2*b:]

        batches = {'r2': r2}

        for name in [
            # "FC",
            # "DC",
            # "GAP",
            "EMD-L2",
            "EMD-L2-16-8-n=1024_from_faces",
            "EMD-L2-16-8-n=1024",
            "Dual-L2",
            # "Dual-L2-16-16_from_faces",
        ]:
            batches[name] = read_grid_batch(os.path.join(batches_dir, f"{name}.png"), d, c).to(device)

        # batches['centroids'] = get_centroids(data, b, use_faiss=True)

        fig, ax = plt.subplots(nrows=2, ncols=len(batches) + 1, figsize=(15,5))
        ax[1,0].imshow(batch_to_image(r1, d, c))
        ax[1,0].axis('off')
        ax[0,0].axis('off')
        ax[1,0].set_title(f'Image-{metric_name}\nPatch-{metric_name}-{p}-{s}')
        for i, (name, batch) in enumerate(batches.items()):
            print(name)
            # dump_images(batch, b, d, c, f"{outdir}/{name}-{b}.png")
            image_dist = metric(batch, r1)
            x = to_patches(batch, d, c, p, s)
            y = to_patches(r1, d, c, p, s)
            patch_dist = metric(x,y)

            ax[0, 1+i].imshow(batch_to_image(batch, d, c))
            ax[0, 1+i].axis('off')
            ax[0, 1+i].set_title(name)
            ax[1, 1+i].axis('off')
            ax[1, 1+i].set_title(f'{image_dist:3f}\n{patch_dist:3f}', size=12)


        plt.tight_layout()
        plt.savefig(os.path.join(batches_dir, f"Plot_{metric_name}.png"))



