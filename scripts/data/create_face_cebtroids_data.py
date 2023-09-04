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
if __name__ == '__main__':
    n_centroids = 128
    crop_size = 90
    img_size = 64
    root = '/cs/labs/yweiss/ariel1/data'
    # root = '/mnt/storage_ssd/datasets/'

    FFHQ = f'{root}/FFHQ/FFHQ'

    out_path = f'{root}/FFHQ/FFHQ_centroids'
    os.makedirs(out_path, exist_ok=True)

    # centers = sample_patch_centers(crop_size-20, img_size, 10000, stride=1, offset=20)
    centers = get_centers(128)
    print(len(centers))
    # exit()
    for i in range(n_centroids):
        fpath = os.path.join(FFHQ, os.listdir(FFHQ)[i])
        img_org = np.array(Image.open(fpath))
        # img_org = crop_center(img_org, crop_size)
        for j, center in enumerate(centers):
            print(i,j, center)
            img = crop(img_org, center, 90)
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(out_path, f"img-{i}-{j}.png"))

