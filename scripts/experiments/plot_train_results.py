import glob
import os
import pickle

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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

    assert len(valid_dirs) == 1, f"Dir description is not unique {len(valid_dirs)}, {valid_dirs}, {names}, {disallowed_names}"
    return valid_dirs[0]

def find_last_image(dir):
    files = glob.glob(f'{dir}/*.png')
    if files:
        return os.path.basename(max(files, key=os.path.getctime))


def plot(root, titles_and_name_lists_dict, plot_loss=None, s=4,  n=5):
    """
    s: base unit for the size of plot
    n: #images in row of displayed images
    h: Ratio of image height to plot height is (h-1)/h
    """
    for plot_name, titles_and_name_lists in titles_and_name_lists_dict.items():
        width = len(titles_and_name_lists)
        if plot_loss == "common":
            width += 1
        h=2 if plot_loss == "separate" else 1
        # fig = plt.figure(figsize=(width * s, h*s))
        fig, axes = plt.subplots(h, width, figsize=(width * s, h*s), squeeze=False, sharey='row')
        # gs = fig.add_gridspec(h, width)
        all_axs = []
        for i, (name, names_list, non_names) in enumerate(titles_and_name_lists):
            found_path = find_dir(root, names_list, non_names)
            if not found_path:
                continue
            dir = os.path.join(root, found_path)
            img_name = find_last_image(os.path.join(dir, "images"))
            images = os.path.join(dir, "images", img_name)
            images = np.array(Image.open(images))
            d = images.shape[0] // 2
            f = n * d // 2
            images = images[d - f:d + f, d - f:d + f]

            ax = axes[0, i]
            ax.imshow(images)
            ax.axis('off')

            plot = os.path.join(dir, "plots", "MiniBatchLoss-dist=w1_fixed_noise_gen_to_train.pkl")
            plot = pickle.load((open(plot, "rb")))
            ax.set_title(f"{name}  W1: {plot[-1]:.3f}", fontsize=4 * s)

            if plot_loss is not None:
                if plot_loss == "separate":
                    ax2 = axes[-1, i]
                    ax2.plot(np.arange(len(plot)), plot, color=COLORS[0], label=f"Image W1")
                    ax2.annotate(f"{plot[-1]:.2f}", (len(plot)-1, plot[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")

                    for j, (name, path) in enumerate([
                        ("Patch-11-W1", "MiniBatchPatchLoss-dist=w1-p=11-s=4-n_samples=1024_fixed_noise_gen_to_train.pkl"),
                        ("Patch-22-W1", "MiniBatchPatchLoss-dist=w1-p=22-s=8-n_samples=1024_fixed_noise_gen_to_train.pkl"),
                        ("Patch-48-W1", "MiniBatchPatchLoss-dist=w1-p=48-s=16-n_samples=1024_fixed_noise_gen_to_train.pkl"),
                    ]):
                        patch_plot = os.path.join(dir, "plots", path)
                        patch_plot = pickle.load((open(patch_plot, "rb")))

                        ax2.plot(np.arange(len(patch_plot)), patch_plot, color=COLORS[j], label=name)
                        ax2.annotate(f"{patch_plot[-1]:.2f}", (len(patch_plot) - 1, patch_plot[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")

                    ax2.set_yscale('log')
                    all_axs.append(ax2)

                    handles, labels = ax2.get_legend_handles_labels()
                    fig.legend(handles, labels, loc='center', ncol=4, prop={'size': 10})

                else:
                    ax2 = axes[:, -1]
                    ax2.plot(np.arange(len(plot)), plot, color=COLORS[i], label=f"{name}")
                    plt.annotate(f"{plot[-1]:.2f}", (len(plot)-1, plot[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")
                    ax2.legend()

                ax2.set_ylabel(f"BatchW1")

        plt.tight_layout()
        plt.savefig(os.path.join(root, plot_name))
        plt.cla()
