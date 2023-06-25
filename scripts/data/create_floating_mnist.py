import os

import numpy as np
from PIL import Image
from tqdm import tqdm

root = '/cs/labs/yweiss/ariel1/data'

mnist_path = f'{root}/MNIST/MNIST/jpgs/training'

offset = 2
imsize = 64
out_path = f'{root}/floating_MNIST/black-{imsize}-{offset}'
os.makedirs(out_path, exist_ok=True)


for fname in tqdm(os.listdir(mnist_path)):
    fpath = os.path.join(mnist_path, fname)
    # print(fpath)
    img = np.array(Image.open(fpath))
    # img = 255 - img

    #new_image = np.ones((imsize,imsize)) * 255
    new_image = np.zeros((imsize,imsize))

    x = np.random.randint(offset, imsize - 28 - offset)
    y = np.random.randint(offset, imsize - 28 - offset)

    new_image[y:y+28, x: x+28] = img

    Image.fromarray(new_image).convert("L").save(os.path.join(out_path, fname))

