import os

import cv2
import numpy as np

def get_locs(d, size, n, mode="rand"):
    if mode == "rand":
        locs = [ (np.random.randint(0, d - size), np.random.randint(0, d - size)) for _ in range(n)]
    else:
        locs = [(i,j) for i in range(d - size) for j in range(d - size)]
    return locs

def create_random(im_size, square_range, n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        img = np.ones((im_size, im_size, 3), dtype=np.uint8) * bg_color

        size = np.random.randint(*square_range)
        x_loc = np.random.randint(0, im_size-size)
        y_loc = np.random.randint(0, im_size-size)
        color = np.random.randint(0, 1, size=3)
        img[y_loc:y_loc+size, x_loc:x_loc+size] = color[None, None, :]

        # print(img.shape, img); exit()
        cv2.imwrite(f"{out_dir}/{i}.png", img)

def create_all(im_size, square_size, offset, s, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    for y_loc in range(offset, im_size - square_size - offset, s):
        for x_loc in range(offset, im_size - square_size - offset, s):
            img = np.ones((im_size, im_size, 3), dtype=np.uint8) * bg_color
            img[y_loc:y_loc+square_size, x_loc:x_loc+square_size] = np.zeros((square_size, square_size, 3))

            # print(img.shape, img); exit()
            cv2.imwrite(f"{out_dir}/{i}.png", img)
            i += 1

    print(f"Created {i} images")

if __name__ == '__main__':
    n = 256
    im_size = 64
    bg_color = 255
    create_all(im_size, square_size=7, offset=7, s=2, out_dir="/cs/labs/yweiss/ariel1/data/square_data/7x7")
    # create_random(im_size, square_range=(16,32), n=256, out_dir="square_data/random")

