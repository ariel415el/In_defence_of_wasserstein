import glob
import os
import pickle

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
"""
from scripts.send_tasks_to_cluster import run_sbatch

def test_if_patch_optimization_reaches_zero_loss(datasets):
    hours = 2
    for dataset in datasets:
        for loss_function in ["BatchEMD-dist=L2",
                              "BatchPatchEMD-dist=L2-p=16-s=16-n_samples=1024",
                              "BatchPatchEMD-dist=L2-p=16-s=8-n_samples=1024",
                              "BatchPatchEMD-dist=L2-p=16-s=1-n_samples=1024"]:
            name = os.path.basename(dataset)
            base = f"python3 train.py  --data_path {dataset}  --D_step_every -1 --lrG 0.001 --gen_arch pixels --batch_size 64  --n_iterations 10000"
            base += " --n_workers 0 --load_data_to_memory --limit_data 64 --project_name EMD_64_all_datasets"

            run_sbatch(base + f" --loss_function {loss_function} ", f"{loss_function}-{name}", hours)
"""


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

    assert len(valid_dirs) == 1, f"Dir description is not unique {len(valid_dirs)}, {valid_dirs}"
    return valid_dirs[0]

def find_last_image(dir):
    files = glob.glob(f'{dir}/*.png')
    if files:
        return max(files, key=os.path.getctime)



def plot(root, datasets, titles_and_name_lists_dict):
    s = 4
    h = 4
    hight=4
    for plot_name, titles_and_name_lists in titles_and_name_lists_dict.items():
        width = len(titles_and_name_lists)
        for dataset in datasets:
            dataset_name = os.path.basename(dataset)
            fig = plt.figure(figsize=(width*s, 1.5*s))
            gs = fig.add_gridspec(hight, width)
            for i, (name, names_list, non_names) in enumerate(titles_and_name_lists):
                dir = os.path.join(root, find_dir(root, [dataset_name] + names_list, non_names))
                images = os.path.join(dir, "images", find_last_image(os.path.join(dir, "images")))
                images = np.array(Image.open(images))
                m = images.shape[0]//2
                images = images[m-64*h:m+64*h, m-64*h:m+64*h]

                ax = fig.add_subplot(gs[:-1, i])
                ax.imshow(images)
                ax.set_title(f"16 / {(len(images)/64)**2} images")
                ax.axis('off')
                ax.set_title(name, fontsize=6)

                plot = os.path.join(dir, "plots", "BatchEMD-dist=L2_fixed_noise_gen_to_train.pkl")
                plot = pickle.load((open(plot, "rb")))

                ax2 = fig.add_subplot(gs[-1, :])
                ax2.plot(np.arange(len(plot)), plot, color=['r', 'g', 'b', 'k', 'y'][i], label=f"{name}: Last:{plot[-1]:.3f}")
                ax2.legend()
                ax2.set_ylabel(f"W1 distance")

            plt.tight_layout()
            plt.savefig(os.path.join(root, plot_name))
            plt.clf()
            plt.cla()

if __name__ == '__main__':
    plot('/home/ariel/university/repos/DataEfficientGANs/outputs/GANs',
        [
            '/mnt/storage_ssd/datasets/FFHQ/FFHQ_1000/FFHQ_1000',
        ],
         {
             "DiscreteGAN.png":  [
                ("DiscreteWGAN-Pixels", ["G-Pixels", "L-WGANLoss"], []),
                ("DiscreteCTGAN-Pixels", ["G-Pixels", "L-CtransformLoss"], []),
                 # ("DiscreteWGAN-FC", ["G-FC", "L-WGANLoss"], []),
                 # ("DiscreteCTGAN-FC", ["G-FC", "L-CtransformLoss"], []),
                 # ("OT-means", ["OTMEANS"], []),

                 # ("DiscreteCT-GAN", ["G-FC", "L-CtransformLoss"], [])
                ]

         }
    )