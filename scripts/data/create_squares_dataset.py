import os

import cv2
import numpy as np

def create_all(im_size, square_size, offset, stride, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    for y_loc in range(offset, im_size - square_size - offset, stride):
        for x_loc in range(offset, im_size - square_size - offset, stride):
            img = np.ones((im_size, im_size, 3), dtype=np.uint8) * bg_color
            img[y_loc:y_loc+square_size, x_loc:x_loc+square_size] = fg_color

            # print(img.shape, img); exit()
            cv2.imwrite(f"{out_dir}/{i}.png", img)
            i += 1

    print(f"Created {i} images")

if __name__ == '__main__':
    im_size = 64
    size = 14
    offset = 1
    stride = 1
    bg_color = 0
    fg_color = 255
    create_all(im_size, square_size=size, offset=offset, stride=stride, out_dir=f"/mnt/storage_ssd/datasets/square_data/black_S-{size}_O-{offset}_S-{stride}")

