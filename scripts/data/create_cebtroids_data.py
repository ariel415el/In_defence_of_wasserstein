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

def get_centers(img_dim, steps=2, strid=1):
    centers = []
    h = img_dim // 2
    for i in range(-steps,steps+1):
        for j in range(-steps,steps+1):
            centers.append((h-i*strid,h-j*strid))
    return centers


def read_img(path):
    fpath = os.path.join(data_path, path)
    return np.array(Image.open(fpath))


def shifted_crops(data_path, out_path, crop_size=90, steps=3, stride=1):
    images_dir = f'{out_path}/shifted_crops/shifted_crops'
    centroids_dir = f'{out_path}/shifted_crops/centroids'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(centroids_dir, exist_ok=True)

    img_paths = os.listdir(data_path)
    img_size = read_img(img_paths[0]).shape[0]
    centers = get_centers(img_size, steps=steps, strid=stride)
    for i in range(n_centroids):
        mean = 0
        for j, center in enumerate(centers):
            print(i,j, center)
            img_org = read_img(img_paths[i])
            img = crop(img_org, center, crop_size)
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(images_dir, f"img-{i}-{j}.png"))
            mean += img.astype(float)
        mean /= len(centers)
        Image.fromarray(mean.astype(np.uint8)).save(os.path.join(centroids_dir, f"mean-{i}.png"))


def floating_images(data_path, out_path, canvas_size, steps=3, stride=1, crop_size=None):
    images_dir = f'{out_path}/floating_images/floating_images'
    centroids_dir = f'{out_path}/floating_images/centroids'
    reference_dir = f'{out_path}/floating_images/reference'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(centroids_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)

    h = crop_size // 2

    img_paths = os.listdir(data_path)
    img_size = read_img(img_paths[0]).shape[0]
    centers = get_centers(canvas_size, steps=steps, strid=stride)
    for i in range(n_centroids):
        mean = 0
        img_org = read_img(img_paths[i])
        if crop_size is not None:
            img_org = crop(img_org, (img_size//2, img_size//2), crop_size)
        for j, center in enumerate(centers):
            print(i,j, center)

            output = np.zeros((canvas_size, canvas_size,3) if len(img_org.shape) == 3 else (canvas_size, canvas_size))
            output[center[0]-h: center[0]+h, center[1]-h: center[1]+h] = img_org

            Image.fromarray(output.astype(np.uint8)).save(os.path.join(images_dir, f"img-{i}-{j}.png"))
            mean += output.astype(float)
        mean /= len(centers)
        Image.fromarray(mean.astype(np.uint8)).save(os.path.join(centroids_dir, f"mean-{i}.png"))

    for j in range(n_centroids*len(centers)):
        img_org = read_img(img_paths[j+i])
        img_org = crop(img_org, (img_size // 2, img_size // 2), crop_size)
        output = np.zeros((canvas_size, canvas_size, 3) if len(img_org.shape) == 3 else (canvas_size, canvas_size))
        c = canvas_size // 2
        center = centers[np.random.randint(len(centers))]
        output[center[0] - h: center[0] + h, center[1] - h: center[1] + h] = img_org
        Image.fromarray(output.astype(np.uint8)).save(os.path.join(reference_dir, f"img-{j}.png"))

if __name__ == '__main__':
    n_centroids = 128
    root = '/mnt/storage_ssd/datasets'
    data_path, out_path = (f'{root}/MNIST/MNIST/jpgs/training', f'{root}/MNIST/MNIST_centroids')
    # shifted_crops(data_path, out_path, crop_size=22, steps=3)
    # floating_images(data_path, out_path, canvas_size=64, steps=3, crop_size=28)
    # data_path, out_path = (f'{root}/FFHQ/FFHQ', f'{root}/FFHQ/FFHQ_centroids')
    # shifted_crops(data_path, out_path, crop_size=90, steps=3)
    # floating_images(data_path, out_path, canvas_size=128, steps=3, crop_size=90)

    data_path, out_path = (f'{root}/afhq/train/cat', f'{root}/afhq/train/cat_centroids')
    shifted_crops(data_path, out_path, crop_size=450, steps=3, stride=3)
    # floating_images(data_path, out_path, canvas_size=128, steps=3, crop_size=90)