import os

import cv2
import numpy as np

def get_locs(d, size, n, mode="rand"):
    if mode == "rand":
        locs = [ (np.random.randint(0, d - size), np.random.randint(0, d - size)) for _ in range(n)]
    else:
        locs = [(i,j) for i in range(d - size) for j in range(d - size)]
    return locs

def create_all(im_size, square_size, offset, s, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    for y_loc in range(offset, im_size - square_size - offset, s):
        for x_loc in range(offset, im_size - square_size - offset, s):
            img = np.ones((im_size, im_size, 3), dtype=np.uint8) * bg_color
            img[y_loc:y_loc+square_size, x_loc:x_loc+square_size] = fg_color

            # print(img.shape, img); exit()
            cv2.imwrite(f"{out_dir}/{i}.png", img)
            i += 1

    print(f"Created {i} images")

if __name__ == '__main__':
    im_size = 64
    size = 14
    bg_color = 0
    fg_color = 255
    create_all(im_size, square_size=size, offset=7, s=2, out_dir=f"/mnt/storage_ssd/datasets/square_data/{size}x7_black")
    # create_random(im_size, square_range=(16,32), n=256, out_dir="square_data/random")

