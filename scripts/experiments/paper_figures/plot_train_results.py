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


def plot_batch(ax, batch_image_path, n, im_dim=64):
    images = np.array(Image.open(batch_image_path))
    f = (n * im_dim)
    images = images[:f, :f]

    ax.imshow(images)
    ax.axis('off')


def load_plot_and_anotate(ax, val_path, color, label, line_type='-'):
    vals = pickle.load((open(val_path, "rb")))
    vals = np.nan_to_num(vals)
    xs = np.arange(len(vals)) / (len(vals) - 1)
    ax.plot(xs, vals, line_type, color=color, label=label, alpha=0.5)
    ax.annotate(f"{vals[-1]:.3f}", (0.9, vals[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")


def plot(named_dirs, out_path, plot_type=None, s=4,  n=5):
    """
    s: base unit for the size of plot
    n: #images in row of displayed images
    h: Ratio of image height to plot height is (h-1)/h
    """

    width = len(named_dirs)
    if plot_type == "common":
        width += 1
    h=2 if plot_type == "separate" else 1
    fig, axes = plt.subplots(h, width, figsize=(width * s, h*s), squeeze=False, sharey='row' if plot_type == "separate" else 'none')
    for i, (name, dir_path) in enumerate(named_dirs.items()):
        ax1 = axes[0, i]
        ax1.set_title(f"{name}", fontsize=3 * s)
        batch_image_path = find_last_file(os.path.join(dir_path, "images"))
        plot_batch(ax1, batch_image_path, n)

        if plot_type is not None:
            if plot_type == "separate":
                ax2 = axes[1, i]

                names_and_plot_paths = [
                    # ("Image-W1", "MiniBatchLoss-dist=w1_fixed_noise_gen_to_train.pkl", COLORS[0], '-'),
                    ("Patch-SWD", "MiniBatchPatchLoss-dist=swd-p=16-s=8_fixed_noise_gen_to_train.pkl", COLORS[1], '--'),
                    # ("Patch-SWD", "MiniBatchPatchLoss-dist=swd-p=8-s=4_fixed_noise_gen_to_train.pkl", COLORS[1], '--'),
                    ("LocalPatch-SWD", "MiniBatchLocalPatchLoss-dist=swd-p=16-s=8_fixed_noise_gen_to_train.pkl", COLORS[2], '--'),
                    # ("LocalPatch-SWD", "MiniBatchLocalPatchLoss-dist=swd-p=8-s=4_fixed_noise_gen_to_train.pkl", COLORS[2], '--'),
                    # ("LocalPatch-W1", "MiniBatchLocalPatchLoss-dist=w1-p=16-s=8_fixed_noise_gen_to_train.pkl", COLORS[3], '--'),
                    # ("LocalPatch-W1", "MiniBatchLocalPatchLoss-dist=w1-p=8-s=4_fixed_noise_gen_to_train.pkl", COLORS[3], '--'),
                ]

                for j, (name, path, color, line_type) in enumerate(names_and_plot_paths):
                    patch_plot = os.path.join(dir_path, "plots", path)
                    load_plot_and_anotate(ax2, patch_plot, color, name, line_type)

                # ax2.set_yscale('log')

            else:
                ax2 = axes[0, -1]
                print(dir_path)
                # image_level_plot = os.path.join(dir_path, "plots", "MiniBatchLoss-dist=w1_fixed_noise_gen_to_train.pkl")
                # ax2.set_ylabel(f"Wasserstein-1")
                # load_plot_and_anotate(ax2, image_level_plot, COLORS[i], name)
                #
                image_level_plot = os.path.join(dir_path, "plots", "MiniBatchPatchLoss-dist=swd-p=16-s=2_fixed_noise_gen_to_train.pkl")
                ax2.set_ylabel(f"Patch SWD")
                load_plot_and_anotate(ax2, image_level_plot, COLORS[i], name, line_type='--')
                # ax2.set_yscale('log')
                ax2.yaxis.tick_right()

                ax2.legend(prop={'size': 2.5*s})
                ax2.set_yticks([])

        else:
            ax1.set_title(name, fontsize=4 * s)

    ax2.set_xlabel(f"Relative train time")
    # ax2.set_xticks([])

    ax2.legend(prop={'size': 2.5 * s})

    # if plot_type == "separate":
    #     handles, labels = ax2.get_legend_handles_labels()
    #     fig.legend(handles, labels, loc='center', ncol=1 + len(names_and_plot_paths), prop={'size': s * h})

    plt.tight_layout()
    plt.savefig(out_path)
    plt.clf()
