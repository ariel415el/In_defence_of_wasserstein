import itertools
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.test_utils import sample_patch_centers


def crop_center(img,d):
    y,x,c = img.shape
    startx = x//2-(d//2)
    starty = y//2-(d//2)
    return img[starty:starty+d,startx:startx+d]


def crop(img,center, d):
    startx = center[1]-(d//2)
    starty = center[0]-(d//2)
    return img[starty:starty+d,startx:startx+d]

def get_centers(img_dim):
    centers = []
    h = img_dim // 2
    for i in [-3 ,-2,-1,0,1,2,3]:
        for j in [-3 ,-2,-1,0,1,2,3]:
            centers.append((h-i,h-j))
    return centers

def ver1(data_path, out_path, crop_size=90):
    images_dir = f'{out_path}/ver1/images'
    centroids_dir = f'{out_path}/ver1/true_centroids'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(centroids_dir, exist_ok=True)

    centers = get_centers(n_centroids)
    data_paths = os.listdir(data_path)
    for i in range(n_centroids):
        fpath = os.path.join(data_path, data_paths[i])
        img_org = np.array(Image.open(fpath))
        mean = 0
        for j, center in enumerate(centers):
            print(i,j, center)
            img = crop(img_org, center, crop_size)
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(images_dir, f"img-{i}-{j}.png"))
            mean += img.astype(float)
        mean /= len(centers)
        Image.fromarray(mean.astype(np.uint8)).save(os.path.join(centroids_dir, f"mean-{i}.png"))


def ver2(data_path, out_path, crop_size=90, offset=16):
    images_dir = f'{out_path}/ver2/images'
    centroids_dir = f'{out_path}/ver2/true_centroids'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(centroids_dir, exist_ok=True)

    h = crop_size // 2


    data_paths = os.listdir(data_path)
    for i in range(n_centroids):
        fpath = os.path.join(data_path, data_paths[i])
        img_org = np.array(Image.open(fpath))
        d = img_org.shape[0]
        center_crop = crop(img_org, (d//2,d//2), crop_size)
        centers = np.arange(h + offset, d - h - offset + 1)
        centers = list(itertools.product(centers, repeat=2))
        mean = 0
        for j, center in enumerate(centers):
            print(i,j, center)
            output = np.zeros((d,d,3))
            output[center[0]-h: center[0]+h, center[1]-h: center[1]+h] = center_crop
            Image.fromarray(output.astype(np.uint8)).save(os.path.join(images_dir, f"img-{i}-{j}.png"))
            mean += output.astype(float)
        mean /= len(centers)
        Image.fromarray(mean.astype(np.uint8)).save(os.path.join(centroids_dir, f"mean-{i}.png"))

if __name__ == '__main__':
    root = '/mnt/storage_ssd/datasets'
    data_path = f'{root}/FFHQ/FFHQ'
    n_centroids = 128
    out_path =  f'{root}/FFHQ/FFHQ_centroids'
    # ver1(data_path, out_path, crop_size=90)
    ver2(data_path, out_path, crop_size=90)
