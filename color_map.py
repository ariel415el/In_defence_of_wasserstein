import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def map_colors(image_path, mask_path, target_rgb):
    image = cv2.imread(image_path)
    new_image = image.copy().astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) == 1
    # print(image.dtype, mask.dtype)

    target_rgb = np.array(target_rgb, dtype=np.float32)

    mean_color = image[mask == True].mean(0)
    new_image[mask == True] += target_rgb - mean_color
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    h,w = image.shape[:2]
    ar = h / w
    S = 15 * h / 1024

    f, axs = plt.subplots(3, 1, figsize=(S, 3 * S * ar))

    target_image = np.zeros_like(new_image)
    target_image[..., 0] = target_rgb[..., 0]
    target_image[..., 1] = target_rgb[..., 1]
    target_image[..., 2] = target_rgb[..., 2]

    axs[0].imshow(image)
    axs[1].imshow(new_image)
    axs[2].imshow(target_image)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()


# Example usage
root = '/mnt/storage_ssd/big_files/archive/car-segmentation'
for fname in os.listdir(os.path.join(root, 'images'))[:3]:
    target_rgb = np.random.randint(0, 255, 3, dtype=np.uint8)
    image_path = f'/mnt/storage_ssd/big_files/archive/car-segmentation/images/{fname}'
    mask_path = f'/mnt/storage_ssd/big_files/archive/car-segmentation/masks/{fname}'

    result_image = map_colors(image_path, mask_path, target_rgb)
