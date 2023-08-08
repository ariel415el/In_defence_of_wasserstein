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


def plot(root, datasets, titles_and_name_lists_dict, seperate_plots=False, s=4,  n=5, h=3):
    """
    s: base unit for the size of plot
    n: #images in row of displayed images
    h: Ratio of image height to plot height is (h-1)/h
    """
    for plot_name, titles_and_name_lists in titles_and_name_lists_dict.items():
        width = len(titles_and_name_lists)
        for dataset in datasets:
            dataset_name = os.path.basename(dataset)
            fig = plt.figure(figsize=(width * s, 1.5 * s))
            gs = fig.add_gridspec(h, width)
            for i, (name, names_list, non_names) in enumerate(titles_and_name_lists):
                found_path = find_dir(root, [dataset_name + "_I"] + names_list, non_names)
                if not found_path:
                    continue
                dir = os.path.join(root, found_path)
                img_name = find_last_image(os.path.join(dir, "images"))
                images = os.path.join(dir, "images", img_name)
                images = np.array(Image.open(images))
                d = images.shape[0] // 2
                f = n * d // 2
                images = images[d - f:d + f, d - f:d + f]

                ax = fig.add_subplot(gs[:-1, i])
                ax.imshow(images)
                ax.axis('off')
                plot = os.path.join(dir, "plots", "MiniBatchLoss-dist=w1_fixed_noise_gen_to_train.pkl")
                plot = pickle.load((open(plot, "rb")))
                ax.set_title(f"{name}  W1: {plot[-1]:.3f}", fontsize=4 * s)
                # if iteration is not None:
                #     plot = plot[:iteration // 1000]
                #     print(len(plot))

                if seperate_plots:
                    ax2 = fig.add_subplot(gs[-1, i])
                    patch_plot = os.path.join(dir, "plots", "MiniBatchPatchLoss-dist=w1-p=16-s=16-n_samples=1024_fixed_noise_gen_to_train.pkl")
                    patch_plot = pickle.load((open(patch_plot, "rb")))
                    ax2.plot(np.arange(len(plot)), plot, color=COLORS[0], label=f"Image W1")
                    ax2.plot(np.arange(len(patch_plot)), patch_plot, color=COLORS[1], label=f"Patch W1")
                else:
                    ax2 = fig.add_subplot(gs[-1, :])
                    # ax2.plot(np.arange(len(plot)), plot, color=COLORS[i], label=f"{name}: Last:{plot[-1]:.3f}")
                    ax2.plot(np.arange(len(plot)), plot, color=COLORS[i], label=f"{name}")

                ax2.legend()
                ax2.set_ylabel(f"BatchW1")

            plt.tight_layout()
            plt.savefig(os.path.join(root, f"{os.path.basename(dataset)}_{plot_name}"))
            plt.clf()
            plt.cla()
