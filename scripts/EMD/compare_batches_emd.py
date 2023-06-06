from collections import defaultdict

import numpy as np
import ot
from PIL import Image
import os
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torchvision import transforms as T
import torch.nn.functional as F


class swd:
    def __init__(self, num_proj=1024):
        self.name = f'SWD-{num_proj}'
        self.num_proj = num_proj

    def __call__(self, x, y):
        b, c = x.shape[0], np.prod(x.shape[1:])

        # Sample random normalized projections
        rand = np.random.randn(self.num_proj, c)
        rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)  # noramlize to unit directions

        # Sort and compute L1 loss
        projx = x @ rand.T
        projy = y @ rand.T

        projx = np.sort(projx, axis=0)
        projy = np.sort(projy, axis=0)

        loss = np.abs(projx - projy).mean()

        return loss


class emd:
    def __init__(self):
        self.name = f'EMD'

    def __call__(self, x, y):
        uniform_x = np.ones(len(x)) / len(x)
        uniform_y = np.ones(len(y)) / len(y)
        M = ot.dist(x, y) / x.shape[1]
        # from utils import compute_distances_batch
        # M = compute_distances_batch(x, y, b=1024)
        return ot.emd2(uniform_x, uniform_y, M)


def get_centroids(data, n_centroids, use_faiss=False):
    if use_faiss:
        import faiss
        kmeans = faiss.Kmeans(data.shape[1], n_centroids, niter=100, verbose=False, gpu=True)
        kmeans.train(data)
        centroids = kmeans.centroids
    else:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_centroids, random_state=0, verbose=0).fit(data)
        centroids = kmeans.cluster_centers_

    return centroids


def read_grid_batch(path, d, c):
    img = Image.open(path)
    img = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).float() / 255
    img = img * 2 - 1
    if c == 1:
        img = torch.mean(img, dim=1, keepdim=True)
    batch = F.unfold(img[..., 2:,2:], kernel_size=d, stride=66)  # shape (b, c*p*p, N_patches)
    batch = batch[0].permute(1,0).reshape(-1, c, d, d).permute(0,2,3,1).reshape(-1, c*d*d).numpy()
    return batch


def get_data(data_path, im_size=None, c=3, limit_data=10000):
    print("Loading data...", end='')
    if os.path.isdir(data_path):
        image_paths = sorted([os.path.join(data_path, x) for x in os.listdir(data_path)])[:limit_data]
    else:
        image_paths = [data_path]

    data = []
    for i, path in enumerate(tqdm(image_paths)):
        im = np.array(Image.open(path).resize((im_size, im_size))) / 255
        im = im * 2 -1
        data.append(im)

    data = np.stack(data, axis=0)
    if c == 1:
        data = np.mean(data, axis=-1, keepdims=True)

    data = data.reshape(len(data), -1)

    print("done")

    return data


def to_patches(x, d, c, p=8, s=4):
    xp = x.reshape(-1, c, d, d)  # shape  (b,c,d,d)
    xp = torch.from_numpy(xp)
    patches = F.unfold(xp, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)               # shape (b, N_patches, c*p*p)
    patches = patches.reshape(-1, patches.shape[-1]) # shape (b * N_patches, c*p*p))
    patches = patches.numpy()
    return patches


def dump_images(imgs, b, d, c, fname):
    save_image(torch.from_numpy(imgs).reshape(b, d, d, c).permute(0,3,1,2), fname, normalize=True, nrow=int(np.sqrt(b)))


def batch_to_image(batch, n=9):
    t_batch = torch.from_numpy(batch).permute(0,3,1,2)
    grid = make_grid(t_batch[:n], normalize=True, nrow=int(np.sqrt(n)))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

if __name__ == '__main__':
    # data_path = '/cs/labs/yweiss/ariel1/data/square_data/7x7'; c=1
    # FC_output_dir = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/WGAN_yweiss/7x7_64x64_G-pixels_D-FC-normalize=none_L-WGANLoss_Z-64_B-64_pixels-FC_7x7_06-04_T-18:47:17'
    # DC_output_dir = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/WGAN_yweiss/7x7_64x64_G-pixels_D-DCGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-DC_7x7_06-04_T-18:47:19'
    # GAP_output_dir = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/WGAN_yweiss/7x7_64x64_G-pixels_D-PatchGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-PatchDisc_7x7_06-04_T-18:47:21'

    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_128'; c = 3
    batches = {
        "RealSanity": '../outputs/WGAN_yweiss/FFHQ_128_64x64_G-pixels_D-FC-normalize=none_L-WGANLoss_Z-64_B-64_pixels-FC_FFHQ_128_06-04_T-18:47:41/images/debug_fixed_reals.png',
        "EMD": "../outputs/GANs/FFHQ_128_64x64_G-pixels_D-DCGAN_L-BatchEMD_Z-64_B-64_test/images/8000.png",
        "EMD-5-1": "../outputs/GANs/FFHQ_128_64x64_G-pixels_D-DCGAN_L-BatchPatchEMD-p=5-s=1_Z-64_B-64_test/images/10000.png",
        "FC": '../outputs/WGAN_yweiss/FFHQ_128_64x64_G-pixels_D-FC-normalize=none_L-WGANLoss_Z-64_B-64_pixels-FC_FFHQ_128_06-04_T-18:47:41/images/100000.png',
        "DC": '../outputs/WGAN_yweiss/FFHQ_128_64x64_G-pixels_D-DCGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-DC_FFHQ_128_06-04_T-18:47:44/images/100000.png',
        "GAP": '../outputs/WGAN_yweiss/FFHQ_128_64x64_G-pixels_D-PatchGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-PatchDisc_FFHQ_128_06-04_T-18:47:46/images/100000.png'
    }

    outdir = os.path.join("output_emd", os.path.basename(data_path))
    os.makedirs(outdir, exist_ok=True)
    metric = emd()
    d = 64
    b = 64
    p, s = 5, 5

    data = get_data(data_path, d, c, limit_data=10000 + 2 * b)

    r1 = data[:b]
    r2 = data[b:2*b]
    data = data[2*b:]

    for k in batches:
        batches[k] = read_grid_batch(batches[k], d, c)
    batches['r2'] = r2
    batches['centroids'] = get_centroids(data, b, use_faiss=False)

    fig, ax = plt.subplots(nrows=2, ncols=len(batches) + 1, figsize=(15,5))
    ax[1,0].imshow(batch_to_image(r1.reshape(-1, d, d, c)))
    ax[1,0].axis('off')
    ax[0,0].axis('off')
    ax[1,0].set_title(f'Image-{metric.name}\nPatch-{metric.name}-{p}-{s}')
    for i, (name, batch) in enumerate(batches.items()):
        dump_images(batch, b, d, c, f"{outdir}/{name}-{b}.png")
        image_dist = metric(batch, r1)
        x = to_patches(batch, d, c, p, s)
        y = to_patches(r1, d, c, p, s)
        patch_dist = metric(x,y)

        ax[0, 1+i].imshow(batch_to_image(batch.reshape(-1, d, d, c)))
        ax[0, 1+i].axis('off')
        ax[0, 1+i].set_title(name)
        ax[1, 1+i].axis('off')
        # ax[1, 1+i].set_title(f'{metric.name}-{name}: {image_dist:3f}\nPatch-{metric.name}-{p}-{s}-{name}: {patch_dist:3f}', size=12)
        ax[1, 1+i].set_title(f'{image_dist:3f}\n{patch_dist:3f}', size=12)


    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot.png"))



