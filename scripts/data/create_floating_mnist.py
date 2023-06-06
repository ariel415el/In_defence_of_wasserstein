import os

import numpy as np
from PIL import Image


root = '/cs/labs/yweiss/ariel1/data'

mnist_path = f'{root}/MNIST/jpgs/training'

offset = 0
imsize = 80
out_path = f'{root}/floating_MNIST/train-{imsize}-{offset}'
os.makedirs(out_path, exist_ok=True)


for  fname in os.listdir(mnist_path):
    fpath = os.path.join(mnist_path, fname)
    print(fpath)
    img = 255 - np.array(Image.open(fpath))

    new_image = np.ones((imsize,imsize)) * 255

    x = np.random.randint(8, imsize - 28 - offset - 1)
    y = np.random.randint(8, imsize - 28 - offset - 1)

    new_image[y:y+28, x: x+28] = img

    Image.fromarray(new_image).convert("L").save(os.path.join(out_path, fname))

