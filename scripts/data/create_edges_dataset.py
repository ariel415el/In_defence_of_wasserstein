import os

import cv2
import numpy as np

def create_all(im_size, n_images, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    for i in range(n_images):
        img = np.zeros((im_size, im_size, 3), dtype=np.uint8)
        img[:, :im_size//2] = np.random.randint(0,255,3)
        img[:, im_size//2:] = np.random.randint(0,255,3)
        cv2.imwrite(f"{out_dir}/{i}.png", img)

    print(f"Created {i+1} images")

if __name__ == '__main__':
    im_size = 64
    create_all(im_size, 1000, out_dir=f"/cs/labs/yweiss/ariel1/data/edges_data")
    # create_random(im_size, square_range=(16,32), n=256, out_dir="square_data/random")

