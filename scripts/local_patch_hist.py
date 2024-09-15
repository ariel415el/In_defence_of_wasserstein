import os
import sys

from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.common import dump_images
from evaluate.test_utils import get_data

import numpy as np
import torch
from matplotlib import pyplot as plt

from losses import to_patches

def get_random_mat(p=11):
    rand_kernel = torch.randn((p * p * 3))
    rand_kernel /= rand_kernel.norm()
    return rand_kernel

def run():
    data = get_data(data_path, im_size, 90, None, limit_data=1)
    patches = to_patches(data, p=11, s=11, remove_locations=False)
    random_locs_and_mats = []
    for i in range(n_iters):
        get_random_mat(p=11)
        loc = np.random.randint(0, patches.shape[0])
        mat = get_random_mat(p=11)
        random_locs_and_mats += [(loc, mat)]

    projs = dict()
    fake = get_data(fake_path, im_size, False, None, limit_data=N)
    patches = to_patches(fake, p=11, s=11, remove_locations=False)
    projs['fake'] = [patches[loc] @ mat for loc, mat in random_locs_and_mats]
    dump_images(fake[:4], "fake.png")
    del fake
    del patches

    data = get_data(data_path, im_size, 90, None, limit_data=N)
    patches = to_patches(data, p=11, s=11, remove_locations=False)
    projs['data'] = [patches[loc] @ mat for loc, mat in random_locs_and_mats]
    dump_images(data[:4], "data.png")
    # del data
    del patches

    blured_real = transforms.GaussianBlur(kernel_size=9, sigma=100)(data)
    del data
    patches = to_patches(blured_real, p=11, s=11, remove_locations=False)
    projs['blured_real'] = [patches[loc] @ mat for loc, mat in random_locs_and_mats]
    dump_images(blured_real[:4], "blured_real.png")
    del blured_real
    del patches

    for i in range(n_iters):
        for name in projs:
            counts, bins = np.histogram(projs[name][i], bins=100)
            plt.plot(bins[:-1], counts, label=name)
        plt.legend()
        plt.save_fig(f"plot{i}.png")
        plt.clf()


if __name__ == '__main__':
    N = 10000
    data_path = '/mnt/storage_ssd/data/FFHQ/FFHQ_128'
    im_size = 128
    n_iters = 5
    fake_path = 'outputs/good_old_fastgan/test_outputs/fake_images'
    run()

# blured_fake_images = transforms.GaussianBlur(kernel_size=9, sigma=100)(fake_images)
#blured_real_images = transforms.GaussianBlur(kernel_size=9, sigma=100)(data)

# dump_images(otmeans[:4], "otmeans.png")
# dump_images(blured_fake_images[:4], "blured_fake_images.png")
#dump_images(blured_real_images[:4], "blured_real_images.png")
