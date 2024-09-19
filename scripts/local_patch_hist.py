import os
import sys

from debian.debtags import output
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.common import dump_images
from evaluate.test_utils import get_data

import numpy as np
import torch
from matplotlib import pyplot as plt

from losses import to_patches


def get_random_mat(p, c):
    rand_kernel = torch.randn((p * p * c))
    rand_kernel /= rand_kernel.norm()
    return rand_kernel


def swd(projx, projy):
    projx, _ = torch.sort(projx)
    projy, _ = torch.sort(projy)

    return (projx - projy).abs().mean()


def run():
    data = get_data(list(paths.values())[0], im_size, crop, gray_scale, limit_data=1)
    patches = to_patches(data, p=p, s=s, remove_locations=False)
    random_locs_and_mats = []
    for i in range(n_iters):
        loc = np.random.randint(0, patches.shape[0])
        mat = get_random_mat(p, data.shape[1])
        random_locs_and_mats += [(loc, mat)]

    projs = dict()

    for name, path in paths.items():
        images = get_data(path, im_size, crop, gray_scale, limit_data=N)
        patches = to_patches(images, p=p, s=s, remove_locations=False)
        projs[name] = [patches[loc] @ mat for loc, mat in random_locs_and_mats]
        dump_images(images[:4], f"{outputs_dir}/{name}.png")
        del patches

        # if blur:
        #     b = 128
        #     n_batches = len(images) // b
        #     for i in tqdm(range(n_batches)):
        #         images[i*b:(i+1)*b] = transforms.GaussianBlur(kernel_size=9, sigma=100)(images[i*b: (i+1)*b])
        #     if len(images) % b != 0:
        #         images[n_batches * b:] = transforms.GaussianBlur(kernel_size=9, sigma=100)(images[n_batches * b:])
        #     patches = to_patches(images, p=p, s=s, remove_locations=False)
        #     projs[f'blured_{name}'] = [patches[loc] @ mat for loc, mat in random_locs_and_mats]
        #     dump_images(images[:4], f"{outputs_diir}/blured_{name}.png")
        #     del patches

        del images

    # Plot
    for i in range(n_iters):
        title = 'SWD: '
        keys = list(projs.keys())
        # for j, name1 in enumerate(keys[:-1]):
            # for name2 in keys[j+1:]:
            #     title += f"{name1} vs {name2}: {swd(projs[name1][i], projs[name2][i]):.2f}; "
        for j, name1 in enumerate(keys[1:]):
            title += f"data vs {name1}: {swd(projs['data'][i], projs[name1][i]):.2f}; "
        for name in projs:
            counts, bins = np.histogram(projs[name][i], bins=100,density=True)
            plt.plot(bins[:-1], counts, label=name)
        plt.title(title, fontsize=10)
        plt.legend()
        plt.savefig(f"{outputs_dir}/plot{i}.png")
        plt.clf()


if __name__ == '__main__':

    im_size = 128
    N = 10000
    crop = None
    gray_scale=False
    outputs_dir = 'outputs/trainDCGAN-16-9-2024local_stats'
    paths = {'data': '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_HQ_cropped',
             # 'fake_m=1K': 'outputs/trainDCGAN-16-9-2024/ffhq_hq_const=1000_I-128_Z-128_Reg-GP_G-DCGAN-nf=256_D-DCGAN-nf=256',
             'fake_m=10K': 'outputs/trainDCGAN-16-9-2024/ffhq_hq_const=10000_I-128_Z-128_Reg-GP_G-DCGAN-nf=256_D-DCGAN-nf=256/test_outputs/fake_images',
             'fake_m=70K': 'outputs/trainDCGAN-16-9-2024/ffhq_hq_const=70000_I-128_Z-128_Reg-GP_G-DCGAN-nf=256_D-DCGAN-nf=256/test_outputs/fake_images',
             'fake_normal': 'outputs/trainDCGAN-16-9-2024/ffhq_hq_normal_I-128_Z-128_Reg-GP_G-DCGAN-nf=256_D-DCGAN-nf=256/test_outputs/fake_images',
             }

    # im_size = 64
    # N = 10000
    # crop = None
    # gray_scale=True
    # paths = {'data': '/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training',
    #          # 'fake_m=1K': 'outputs/trainDCGAN-17-9-2024_mnist/mnist_const=1000_I-64_Z-64_Reg-GP_G-DCGAN_D-DCGAN/test_outputs/fake_images',
    #          'fake_m=10K': 'outputs/trainDCGAN-17-9-2024_mnist/mnist_const=10000_I-64_Z-64_Reg-GP_G-DCGAN_D-DCGAN/test_outputs/fake_images',
    #          'fake_Normal': 'outputs/trainDCGAN-17-9-2024_mnist/mnist_normal_I-64_Z-64_Reg-GP_G-DCGAN_D-DCGAN/test_outputs/fake_images',
    #          }
    #
    # im_size = 128
    # N = 10000
    # crop = None
    # gray_scale=False
    # outputs_dir = 'outputs/trainDCGAN-18-9-2024_color_mnist_hq/local_stats'
    # paths = {'data': '/cs/labs/yweiss/ariel1/data/MNIST/Color_MNIST_hq',
    #          # 'fake_m=1K': 'outputs/trainDCGAN-17-9-2024_mnist/mnist_const=1000_I-64_Z-64_Reg-GP_G-DCGAN_D-DCGAN/test_outputs/fake_images',
    #          'fake_m=10K': 'outputs/trainDCGAN-18-9-2024_color_mnist_hq/color_mnist_hq_const=10000_I-128_Z-128_Reg-GP_G-DCGAN_D-DCGAN/test_outputs/fake_images',
    #          'fake_Normal': 'outputs/trainDCGAN-18-9-2024_color_mnist_hq/color_mnist_hq_normal_I-128_Z-128_Reg-GP_G-DCGAN_D-DCGAN/test_outputs/fake_images',
    #          }

    blur = False
    n_iters = 5
    p = 11
    s = 11

    os.makedirs(outputs_dir, exist_ok=True)
    run()

