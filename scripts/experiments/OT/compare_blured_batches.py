import os
import sys
import torch
from matplotlib import pyplot as plt
import numpy as np
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from losses.optimal_transport import MiniBatchPatchLoss
from utils.common import parse_classnames_and_kwargs
from scripts.experiments.experiment_utils import get_data, batch_to_image
from torchvision.transforms import transforms

COLORS =['r', 'g', 'b', 'k', 'y', 'm', 'c']


def main(noise=False):
    """Compare batch of blurred images with increasing sigma to batch of sharp data
        Comparison is done with different metrics (W1, SWD) and on image and patch level
    """
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        data = get_data(data_path, im_size, c=c, center_crop=center_crop, gray_scale=gray_scale, flatten=False, limit_data=b+n_images).to(device)
        b1 = data[:b]
        data = data[-n_images:]

        names_and_batches = [("sigma=0", b1)]
        # names_and_batches = []
        if noise:
            names_and_batches += [
                (f"sigma={sigma}", b1 + torch.randn_like(b1) * 0.1*sigma)
                for sigma in sigmas
            ]
        else:
            names_and_batches += [
                (f"sigma={sigma}", transforms.GaussianBlur(kernel_size=15, sigma=sigma)(b1))
                for sigma in sigmas
            ]

        plot_images(names_and_batches)

        data.share_memory_()
        patch_dists_means = {dist: [] for dist in dists}
        patch_dists_stds = {dist: [] for dist in dists}
        for name, batch in names_and_batches:
            batch.share_memory_()
            for dist in dists:
                print(name, dist)
                dist_name, kwargs = parse_classnames_and_kwargs(dist)
                dist_metric = MiniBatchPatchLoss(dist_name, p=p, s=s, **kwargs)
                # vals = [dist_metric(batch.clone(), data.clone()) for _ in range(n_reps)]
                vals = run_distributed(dist_metric, batch, data, n_reps)
                patch_dists_means[dist].append(np.mean(vals))
                patch_dists_stds[dist].append(np.std(vals))

        plot([0] + sigmas, dists, patch_dists_means, patch_dists_stds, normalize=False)
        plot([0] + sigmas, dists, patch_dists_means, patch_dists_stds, normalize=True)


def plot_images(names_and_batches):
    """Plot the blurred images in all sigmas"""
    w = len(names_and_batches)
    fig, ax = plt.subplots(nrows=1, ncols=w, figsize=(w * size, size * 1.1))
    for i, (name, batch) in enumerate(names_and_batches):
        ax[i].imshow(batch_to_image(batch, im_size, c))
        ax[i].axis('off')
        ax[i].set_title(f"{name}:")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'blurred_batches.png'))
    plt.clf()


def plot(sigmas, dists, patch_dists_means, patch_dists_stds, normalize=False):
    """Compare the plots of different metrics on the same level (Image/Patch)"""
    plt.figure()
    for i, dist in enumerate(dists):
        n = len(patch_dists_means[dist])
        label = f"patch({p}-{s})-" + dist

        vals = np.array(patch_dists_means[dist])
        stds = np.array(patch_dists_stds[dist])
        if normalize:
            stds /= vals[0]
            vals /= vals[0]

        plt.plot(sigmas, vals, label=label, alpha=0.4, color=COLORS[i])
        plt.fill_between(sigmas, vals - stds / 2, vals + stds / 2, alpha=0.15, color=COLORS[i])

        plt.xticks(sigmas, [f"{x}" for x in sigmas], rotation=45)
        plt.xlabel("Blur Sigma")
        plt.ylabel("Change factor")
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #           fancybox=True, shadow=True, ncol=5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'blurred_plot{f"-patch({p}-{s})"}{"_normalize" if normalize else ""}.png'))
    plt.clf()


def worker(idx, vals, f, batch, data):
    from datetime import datetime
    torch.manual_seed(datetime.now().timestamp() + idx)
    vals[idx] = f(batch, data)


def run_distributed(f, b1, b2, n):
    ret_vals = mp.Manager().list()
    for _ in range(n):
        ret_vals.append(0)
    ps = []
    torch.set_num_threads(1)
    for idx in range(n):
        pr = mp.Process(target=worker, args=(idx, ret_vals, f, b1, b2))
        pr.start()
        ps.append(pr)
    for pr in ps:
        pr.join()
    return ret_vals


if __name__ == '__main__':
    device = torch.device('cpu')
    b = 64
    n_images = 64
    im_size = 64
    n_proj = 4
    size = 5
    dists = [
            "w1",
             f'swd-num_proj={n_proj}',
             # f'projected_w1-num_proj={n_proj}-dim=1',
             # f'projected_w1-num_proj={n_proj}-dim=4',
             f'projected_w1-num_proj={n_proj}-dim=9',
             ]
    sigmas = [0.5, 1.5, 3]
    s = 8
    n_reps = 12

    c = 1
    gray_scale = True
    for p in [3, 8]:
        for noise in [False]:
            for data_path, center_crop in [
                ('/mnt/storage_ssd/datasets/square_data/black_S-10_O-1_S-1', None),
                ('/mnt/storage_ssd/datasets/MNIST/MNIST/jpgs/training', None),
                ('/mnt/storage_ssd/datasets/FFHQ/FFHQ', 80),
            ]:
                output_dir = os.path.join(os.path.dirname(__file__),
                                          "outputs", os.path.basename(data_path),
                                          f"compare_blured_batches-{p}-{s}-{n_images}-N={noise}")

                main(noise)