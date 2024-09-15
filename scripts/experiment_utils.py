import glob
import os

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import torch.nn.functional as F

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data import get_transforms


def get_data(data_path, im_size=None, c=3, center_crop=None, gray_scale=False, flatten=True, limit_data=None):
    if os.path.isdir(data_path):
        image_paths = sorted([os.path.join(data_path, x) for x in os.listdir(data_path)])[:limit_data]
    else:
        image_paths = [data_path]

    transform = get_transforms(im_size, center_crop,  gray_scale)

    data = []
    for i, path in enumerate(tqdm(image_paths)):
        img = Image.open(path).convert('RGB')
        img = transform(img)
        data.append(img)

    data = torch.stack(data)
    if c == 1:
        data = torch.mean(data, dim=1, keepdim=True)

    if flatten:
        data = data.reshape(len(data), -1)

    return data


def get_centroids(data, n_centroids, use_faiss=False):
    print("Computing centroids", end='...')
    np_data = data.cpu().numpy().reshape(data.shape[0], -1).copy(order='C')
    if use_faiss:
        import faiss
        kmeans = faiss.Kmeans(np_data.shape[1], n_centroids, verbose=True, gpu=True)
        kmeans.train(np_data)
        centroids = kmeans.centroids

    else:

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_centroids, random_state=0, verbose=0, max_iter=50).fit(np_data)
        centroids = kmeans.cluster_centers_


    centroids = torch.from_numpy(centroids).to(data.device)
    print("Got centroids")
    return centroids.reshape(n_centroids,*data.shape[1:])


def read_grid_batch(path, d, c, flatten=True):
    img = get_data(path, None, c, flatten=False)[0].unsqueeze(0)
    batch = F.unfold(img[..., 2:,2:], kernel_size=d, stride=d+2)  # shape (b, c*p*p, N_patches)
    batch = batch[0].permute(1,0).reshape(-1, c, d, d)
    if flatten:
        batch = batch.reshape(-1, c*d*d)
    return batch


def to_patches(x, d, c, p=8, s=4, limit_patches=None):
    xp = x.reshape(-1, c, d, d)  # shape  (b,c,d,d)
    patches = F.unfold(xp, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)               # shape (b, N_patches, c*p*p)
    patches = patches.reshape(-1, patches.shape[-1]) # shape (b * N_patches, c*p*p))
    if limit_patches is not None and limit_patches < len(patches):
        samples = np.random.choice(len(patches), size=min(len(x), limit_patches), replace=False)
        patches = patches[samples]
    return patches


def batch_to_image(batch, d, c, n=16):
    t_batch = batch.reshape(-1, c, d, d)
    grid = make_grid(t_batch[:n], normalize=True, nrow=int(np.sqrt(n)))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).cpu().numpy()


def find_last_file(dir, ext='.png'):
    files = glob.glob(f'{dir}/*{ext}')
    if files:
        return max(files, key=os.path.getctime)
