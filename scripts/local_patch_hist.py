import os
import sys

from matplotlib import pyplot as plt
import pandas as pd
import psutil

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from losses import MiniBatchLocalPatchLoss, MiniBatchLoss, MiniBatchPatchLoss


def get_random_mat(p, c):
    rand_kernel = torch.randn((p * p * c))
    rand_kernel /= rand_kernel.norm()
    return rand_kernel


def swd(projx, projy):
    projx, _ = torch.sort(projx)
    projy, _ = torch.sort(projy)

    return (projx - projy).abs().mean()


def get_random_locs(n_iters, c, dim):
    random_locs_and_mats = []
    for i in range(n_iters):
        x = np.random.randint(0, dim - p)
        y = np.random.randint(0, dim - p)
        mat = get_random_mat(p, c)
        random_locs_and_mats += [(x, y, mat)]
    return random_locs_and_mats


def get_local_hist(batch, x, y, mat):
    return batch[..., x:x + p, y:y + p].reshape(len(batch), -1) @ mat


def compute_local_patch_w1(plot_name, p):
    with torch.no_grad():
        metrics = [
            # ("w1", MiniBatchLoss(dist='w1')),
            # ("patch_swd-7-4", MiniBatchPatchLoss(dist='swd', p=7, s=4)),
            ("Local_swd-7-4", MiniBatchLocalPatchLoss(dist='swd', p=p, s=4)),
        ]

        data = torch.load(paths['FFHQ'])

        random_locs_and_mats = get_random_locs(n_iters, c=data.shape[1], dim=data.shape[-1])
        metrics_table = []
        projections = {'FFHQ': [get_local_hist(data, x, y, mat) for x, y, mat in random_locs_and_mats]}
        for j, name in enumerate(list(paths.keys())[1:]):
            metrcs_row = []
            fake = torch.load(paths[name])
            from utils.common import dump_images
            dump_images(fake[:4], f"{outputs_dir}/{name}")
            projections[name] = [get_local_hist(fake, x, y, mat) for x, y, mat in random_locs_and_mats]
            for metric_name, metric in metrics:
                dist = metric(data, fake).item()
                metrcs_row += [dist]
                print(f"{metric_name} data vs {name}: {dist:.5f}. CPU: {psutil.cpu_percent()}%")
            del fake
            metrics_table += [metrcs_row]

    # Dump table
    print(metrics_table)
    df = pd.DataFrame(metrics_table, index=list(paths.keys())[1:], columns=np.array(metrics)[:, 0])
    df.to_csv(os.path.join(root, f"{plot_name}.csv"), sep=',', encoding='utf-8')

    # Plot histograms
    for i in range(n_iters):
        # title = 'SWD: '
        # keys = list(projs.keys())
        # for j, name1 in enumerate(keys[1:]):
        # title += f"data vs {name1}: {swd(projs['data'][i], projs[name1][i]):.2f}; "
        # plt.title(title, fontsize=10)
        for name in projections:
            counts, bins = np.histogram(projections[name][i], bins=nbins, density=True)
            plt.plot(bins[:-1], counts, label=name)
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{outputs_dir}/plot{i}.png")
        plt.clf()

if __name__ == '__main__':
    n_iters = 6
    p=7
    nbins = 30

    root = 'outputs/trainDCGAN-16-9-2024'
    outputs_dir = f'{root}/local_stats'
    os.makedirs(outputs_dir, exist_ok=True)
    paths = {'FFHQ': '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_HQ_cropped.pth',
             'otMeans': f'otMeans10K.pth',
             'fake_m=10K': f'{root}/ffhq_hq_const=10000_I-128_Z-128_Reg-GP_G-DCGAN-nf=256_D-DCGAN-nf=256/test_outputs/fake_images.pth',
             'fake_m=70K': f'{root}/ffhq_hq_const=70000_I-128_Z-128_Reg-GP_G-DCGAN-nf=256_D-DCGAN-nf=256/test_outputs/fake_images.pth',
             'fake_normal': f'{root}/ffhq_hq_normal_I-128_Z-128_Reg-GP_G-DCGAN-nf=256_D-DCGAN-nf=256/test_outputs/fake_images.pth',
             'afhq': f'afhq.pth',
             'ImageNet': f'imagenette2.pth',
             }

    compute_local_patch_w1("DCGAN", p)

    root = 'outputs/trainOldFastGAN_30_sep'
    outputs_dir = f'{root}/local_stats'
    os.makedirs(outputs_dir, exist_ok=True)
    paths = {'FFHQ': '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_HQ_cropped.pth',
             'otMeans': f'otMeans10K.pth',
             'fake_m=10K': f'{root}/ffhq_hq_const=10000_I-128_Z-128_Reg-GP_G-OldFastGAN_D-OldFastGAN/test_outputs/fake_images.pth',
             'fake_m=70K': f'{root}/ffhq_hq_const=70000_I-128_Z-128_Reg-GP_G-OldFastGAN_D-OldFastGAN/test_outputs/fake_images.pth',
             'fake_normal': f'{root}/ffhq_hq_normal_I-128_Z-128_Reg-GP_G-OldFastGAN_D-OldFastGAN/test_outputs/fake_images.pth',
             'afhq': f'afhq.pth',
             'ImageNet': f'imagenette2.pth',
             }

    compute_local_patch_w1("FastGAN", p)

    # for dataset_name in ['ffhq', 'mnist', 'squares']:
    #     root = 'outputs/old/paper_figures/paper_figures_27-1'
    #     outputs_dir = f'{root}/{dataset_name}_local_stats'
    #     paths = {'data': f'{root}/{dataset_name}.pth',
    #              'FC': f'{root}/{dataset_name}-const=64_WGAN-GP-FC-nf=1024/test_outputs/fake_images.pth',
    #              'CNN-GAP': f'{root}/{dataset_name}-const=64_WGAN-GP-CNN-GAP=True/test_outputs/fake_images.pth',
    #              'CNN-FC': f'{root}/{dataset_name}-const=64_WGAN-GP-CNN-GAP=False/test_outputs/fake_images.pth',
    #              'OTMeans': f'{root}/{dataset_name}_otMeans.pth',
    #              }
    #
    #     os.makedirs(outputs_dir, exist_ok=True)
    #     compute_local_patch_w1()
    #     # run()
    #
