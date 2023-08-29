import os
from random import shuffle

import numpy as np
from PIL import Image
from tqdm import tqdm

root = '/cs/labs/yweiss/ariel1/data'
# root = '/mnt/storage_ssd/datasets/'

FFHQ_1000 = f'{root}/FFHQ/FFHQ_1000'

out_path = f'{root}/FFHQ/FFHQ_flat'
os.makedirs(out_path, exist_ok=True)

fpath = os.path.join(FFHQ_1000, os.listdir(FFHQ_1000)[0])
img_org = np.array(Image.open(fpath))


def get_prod():
    r = np.random.rand(1)[0]
    g = np.random.uniform(0, 1-r)
    b = np.random.uniform(0, 1-r-g)
    rgb = [r,g,b]
    np.random.shuffle(rgb)
    return tuple(rgb)


if __name__ == '__main__':
    for i in range(1000):
        img = img_org.copy().astype(float)

        rand_mat = np.random.randn(3,3)
        it = 0
        while True:
            if it % 100 ==0 :
                print(i, it)
            x = img @ np.random.randn(3,3)
            x -= x.min()
            if x.max() < 255:
                break
            it += 1
        img = x
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(out_path, f"{i}.png"))

