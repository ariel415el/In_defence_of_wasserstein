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


def plot_batch(ax, batch_image_path, nx, ny, im_dim=64):
    images = np.array(Image.open(batch_image_path))
    f = (nx * im_dim)
    images = images[- nx * im_dim:, :ny * im_dim]

    ax.imshow(images)
    ax.axis('off')


def load_plot_and_anotate(ax, val_path, label, line_type='-', color='b'):
    vals = pickle.load((open(val_path, "rb")))
    vals = np.nan_to_num(vals)
    # vals = vals / vals.max()
    xs = np.arange(len(vals)) / (len(vals) - 1)
    ax.plot(xs, vals, line_type, label=label, alpha=0.5, color=color)
    ax.annotate(f"{vals[-1]:.3f}", (0.9, vals[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")
    ax.set_yscale('log')
    # ax.set_xscale('log')


def create_plot(dataset_name):
    plot_metrics = False
    width = len(raw_train_names[0]) + (1 if plot_metrics else 0)
    height = len(raw_train_names)
    s = 5
    COLORS = ['r', 'g', 'b']
    fig, axes = plt.subplots(height, width, figsize=(width * s, height * s - s))
    for i in range(len(raw_train_names)):
        for j in range(len(raw_train_names[0])):
            ax1 = axes[i, j]
            if raw_train_names[i][j] is None:
                ax1.axis("off")
                continue
            model_name, filename = raw_train_names[i][j]
            dir_path = os.path.join(root_dir, f"{dataset_name}-{filename}")
            batch_image_path = find_last_file(os.path.join(dir_path, "images"))
            ax1.set_title(f"{model_name}", fontsize=3 * s)
            print(dir_path, batch_image_path)
            plot_batch(ax1, batch_image_path, 2,3)

            if plot_metrics:
                for k, (metric_name, metric_file_name, line_type) in enumerate(metrics):
                    ax2 = axes[i, len(raw_train_names[0])]
                    patch_plot = os.path.join(dir_path, "plots", f"{metric_file_name}_fixed_noise_gen_to_train.pkl")
                    load_plot_and_anotate(ax2, patch_plot, label=f'{model_name}-{metric_name}', line_type=line_type, color=COLORS[j])
                    ax2.set_ylabel(metric_name, fontsize=3 * s)

                    ax2.set_xlabel(f"Relative train time", fontsize=3 * s)
                    # ax2.set_xticks([])
                    ax2.legend(prop={'size': 2.5 * s})

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, f"{dataset_name}.png"))


def create_table(dataset_name):
    height = len(raw_train_names)
    width = len(metrics[0])
    data = np.zeros((2, height, width))
    for r in range(height):
        for c in range(width):
            metric_name, metric_file_name, _ = metrics[0][c]
            for i in range(2):
                gan_model_name, filename = raw_train_names[r][i]
                dir_path = os.path.join(root_dir, f"{dataset_name}-{filename}")
                patch_plot = os.path.join(dir_path, "plots", f"{metric_file_name}_fixed_noise_gen_to_train.pkl")
                data[i, r, c] = pickle.load(open(patch_plot, "rb"))[-1]

    import pandas as pd
    df = pd.DataFrame(data[0], index=np.array(raw_train_names)[:, 0, 0], columns=np.array(metrics)[0, :, 0])
    df.to_csv(os.path.join(root_dir, f"{dataset_name}-WGAN.csv"), sep=',', encoding='utf-8')
    df = pd.DataFrame(data[1], index=np.array(raw_train_names)[:, 1, 0], columns=np.array(metrics)[0, :, 0])
    df.to_csv(os.path.join(root_dir, f"{dataset_name}-Direct.csv"), sep=',', encoding='utf-8')


if __name__ == '__main__':
    root_dir = 'outputs/old/paper_figures/paper_figures_27-1'
    prior = 'const=64'
    raw_train_names = [
        [('WGAN_FC', f'{prior}_WGAN-GP-FC-nf=1024'), ('Direct_Image-W1', f'{prior}_DirectW1')],
        [('WGAN_CNN-GAP', f'{prior}_WGAN-GP-CNN-GAP=True'), ('Direct_Patch-SWD', f'{prior}_DirectPatchSWD')],
        [('WGAN_CNN-FC', f'{prior}_WGAN-GP-CNN-GAP=False'), ('Direct_LocalPatch-SWD', f'{prior}_DirectLocalPatchSWD')],
    ]
    metrics = [
        ('Image-W1', 'MiniBatchLoss-dist=w1', '-'),
        ('Patch-swd', 'MiniBatchPatchLoss-dist=swd-p=16-s=8', '--'),
        ('PatchLocal-swd', 'MiniBatchLocalPatchLoss-dist=swd-p=16-s=8', '--') ,
        # [('Image-W1', 'MiniBatchLoss-dist=w1', '-'), ('PatchLocal-SWD', 'MiniBatchLocalPatchLoss-dist=swd-p=16-s=8', '--') ],
        # [('Patch-W1', 'MiniBatchLocalPatchLoss-dist=w1-p=16-s=8', '-'),],
        # [('patch-SWD', 'MiniBatchPatchLoss-dist=swd-p=16-s=8', '--'),],
        # [('patch-SWD', 'MiniBatchPatchLoss-dist=swd-p=16-s=8', '--'),],
        # [('PatchLocal-SWD', 'MiniBatchLocalPatchLoss-dist=swd-p=16-s=8', '--')],
    ]
    dataset_names = [
            'ffhq',
            'mnist',
            'squares'
        ]
    for dataset_name in dataset_names:
        create_plot(dataset_name)
        # create_table(dataset_name)
