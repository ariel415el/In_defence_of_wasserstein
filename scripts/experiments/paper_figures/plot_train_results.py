import os
import pickle
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiment_utils import find_dir, find_last_file

COLORS=['r', 'g', 'b', 'c', 'm', 'y', 'k']


def get_dir_paths(root, titles_and_name_lists):
    named_dirs = dict()
    for i, (name, names_list, non_names) in enumerate(titles_and_name_lists):
        found_path = find_dir(root, names_list, non_names)
        if not found_path:
            continue
        named_dirs[name] = os.path.join(root, found_path)
    return named_dirs


def plot_batch(ax, batch_image_path, n):
    images = np.array(Image.open(batch_image_path))
    d = images.shape[0] // 2
    f = n * d // 2
    images = images[d - f:d + f, d - f:d + f]

    ax.imshow(images)
    ax.axis('off')


def load_plot_and_anotate(ax, val_path, color, label, line_type='-'):
    vals = pickle.load((open(val_path, "rb")))
    ax.plot(np.arange(len(vals)), vals, line_type, color=color, label=label)
    ax.annotate(f"{vals[-1]:.2f}", (len(vals) - 1, vals[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")


def plot(named_dirs, out_path, plot_loss=None, s=4,  n=5):
    """
    s: base unit for the size of plot
    n: #images in row of displayed images
    h: Ratio of image height to plot height is (h-1)/h
    """

    width = len(named_dirs)
    if plot_loss == "common":
        width += 1
    h=2 if plot_loss == "separate" else 1
    fig, axes = plt.subplots(h, width, figsize=(width * s, h*s), squeeze=False, sharey='row' if plot_loss == "separate" else 'none')
    for i, (name, dir_path) in enumerate(named_dirs.items()):
        axes[0, i].set_title(f"{name}", fontsize=4 * s)

        batch_image_path = find_last_file(os.path.join(dir_path, "images"))
        plot_batch(axes[0, i], batch_image_path, n)

        image_level_plot = os.path.join(dir_path, "plots", "MiniBatchLoss-dist=swd_fixed_noise_gen_to_train.pkl")
        if plot_loss is not None:
            if plot_loss == "separate":
                ax2 = axes[1, i]
                load_plot_and_anotate(ax2, image_level_plot, COLORS[0], f"Image SWD", line_type='-')

                names_and_plot_paths = [
                    ("Patch-4-swd", "MiniBatchPatchLoss-dist=swd-p=4-s=4_fixed_noise_gen_to_train.pkl", COLORS[1], '--'),
                    ("Patch-8-swd", "MiniBatchPatchLoss-dist=swd-p=8-s=4_fixed_noise_gen_to_train.pkl", COLORS[2], '--'),
                    ("Patch-16-swd", "MiniBatchPatchLoss-dist=swd-p=16-s=8_fixed_noise_gen_to_train.pkl", COLORS[3], '--'),
                ]

                for j, (name, path, color, line_type) in enumerate(names_and_plot_paths):
                    patch_plot = os.path.join(dir_path, "plots", path)
                    load_plot_and_anotate(ax2, patch_plot, color, name, line_type)

                handles, labels = ax2.get_legend_handles_labels()
                fig.legend(handles, labels, loc='center', ncol=1+len(names_and_plot_paths), prop={'size': s*width})
            else:
                ax2 = axes[0, -1]
                load_plot_and_anotate(ax2, image_level_plot, COLORS[i], name)
                ax2.legend()

            ax2.set_yscale('log')
            ax2.set_ylabel(f"BatchW1")
        else:
            axes[0, i].set_title(f"{name}  W1: {plot[-1]:.3f}", fontsize=4 * s)

        plt.tight_layout()
        plt.savefig(out_path)
        plt.cla()
