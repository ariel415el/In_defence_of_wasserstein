import os
import pickle
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.experiment_utils import find_last_file

COLORS=['r', 'g', 'b', 'c', 'm', 'y', 'k']


def find_dir(root, names, disallowed_names=[]):
    valid_dirs = []
    for dname in os.listdir(root):
        is_good = True
        for name in names:
            if name not in dname:
                is_good = False
                break
        for name in disallowed_names:
            if name in dname:
                is_good = False
        if is_good:
            valid_dirs.append(dname)

    assert len(valid_dirs) == 1, (f"Dir description is not unique:)"
                                  f"\t-\nValid dirs: {valid_dirs}"
                                  f"\t-\nnames: {names}"
                                  f"\t-\ndisallowed_names: {disallowed_names}")
    return valid_dirs[0]


def plot_batch(ax, batch_image_path, n, im_dim=64):
    images = np.array(Image.open(batch_image_path))
    f = (n * im_dim)
    images = images[-f:, :f]

    ax.imshow(images)
    ax.axis('off')


def plot_sampe_prior(data_name, z_priors, gen_arch, out_name):
    s = 3
    fig, axes = plt.subplots(2, len(ms), figsize=(len(ms) * s, 1 * s))
    for i, m in enumerate(ms):
        z_prior = z_priors[i]
        dir_path = os.path.join(root_dir, f"{data_name}_Z-{z_prior}_G-{gen_arch}_DirectW1_fbs-{m}")
        batch_image_path = find_last_file(os.path.join(dir_path, "images"))
        axes[0, i].set_title(f"Z-{z_prior}_fbs={m}", fontsize=3 * s)
        print(dir_path, batch_image_path)
        plot_batch(axes[0, i], batch_image_path, 3)

    for i, m in enumerate(ms):
        z_prior = z_priors[i]
        dir_path = os.path.join(root_dir, f"{data_name}_Z-{z_prior}_G-{gen_arch}_WGAN-FC-nf=1024_fbs-{m}")
        batch_image_path = find_last_file(os.path.join(dir_path, "images"))
        axes[1, i].set_title(f"WGAN-FC_Z-{z_prior}_fbs={m}", fontsize=3 * s)
        print(dir_path, batch_image_path)
        plot_batch(axes[1, i], batch_image_path, 3)

    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, f"{out_name}.png"))


if __name__ == '__main__':
    # ms = [64, 250, 500, 1000]
    ms = [16, 1000]
    root_dir = 'outputs/effect_of_batchsize_on_prior_5'
    dataset_names = [
            'ffhq',
            'mnist',
            'squares'
        ]
    for dataset_name in dataset_names:
        for gen_arch in ['FC']:
            # plot_sampe_prior(dataset_name, ['const=1000'] * len(ms), gen_arch, f"{dataset_name}-'const=1000'_G-{gen_arch}.png")
            plot_sampe_prior(dataset_name, ['normal'] * len(ms), gen_arch, f"{dataset_name}-normal_G-{gen_arch}.png")
            plot_sampe_prior(dataset_name, [f'const={m}' for m in ms], gen_arch, f"{dataset_name}-{'const={m}'}_G-{gen_arch}.png")

