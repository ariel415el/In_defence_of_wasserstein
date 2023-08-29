import os
import sys
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from losses.optimal_transport import MiniBatchLoss, MiniBatchPatchLoss
from scripts.EMD.utils import get_data, batch_to_image
from torchvision.transforms import transforms
COLORS =['r', 'g', 'b', 'k']
if __name__ == '__main__':
    with torch.no_grad():
        device = torch.device('cpu')
        b = 64
        im_size=64
        s = 5

        data_path = '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'
        c=3
        data = get_data('/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ', im_size, c=c, center_crop=80, gray_scale=False, flatten=False, limit_data=1064).to(device)

        # data_path = '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_frontal_1000'
        # data = get_data(data_path, im_size, c=1, center_crop=80, gray_scale=True, flatten=False, limit_data=1000).to(device)

        b1 = data[:b]
        data = data[b:]

        names_and_batches = [
            ("K=0", b1),
            ("K=15,1", transforms.GaussianBlur(kernel_size=15, sigma=1)(b1)),
            ("K=15,2", transforms.GaussianBlur(kernel_size=15, sigma=2)(b1)),
            ("K=15,3", transforms.GaussianBlur(kernel_size=15, sigma=3)(b1)),
            ("K=15,4", transforms.GaussianBlur(kernel_size=15, sigma=4)(b1)),
        ]

        dists = {"nn": [], 'swd': []}
        patch_dists = {dist: [] for dist in dists.keys()}

        w = len(names_and_batches)
        fig, ax = plt.subplots(nrows=1, ncols=w, figsize=(w * s, s * 1.1))

        for i, (name, batch) in enumerate(names_and_batches):
            ax[i].imshow(batch_to_image(batch, im_size, c))
            ax[i].axis('off')
            ax[i].set_title(f"{name}:")

            for dist in dists.keys():
                print(name, dist)
                dists[dist].append(MiniBatchLoss(dist)(batch, data))
                patch_dists[dist].append(MiniBatchPatchLoss(dist, p=8, s=8)(batch, data))
                print(dists[dist], patch_dists[dist])

        plt.tight_layout()
        plt.savefig('blurred_batches.png')
        plt.clf()

        plt.figure()
        for i, dist in enumerate(dists.keys()):
            plt.plot(range(len(dists[dist])), dists[dist], label=dist, color=COLORS[i], alpha=0.75)
            plt.plot(range(len(patch_dists[dist])), patch_dists[dist], '--', label="patch-"+dist, color=COLORS[i], alpha=0.75)
            plt.annotate(f"{dists[dist][-1]:.2f}", (len(dists[dist]) - 1, dists[dist][-1]), textcoords="offset points", xytext=(-2, 2), ha="center")
            plt.annotate(f"{patch_dists[dist][-1]:.2f}", (len(patch_dists[dist]) - 1, patch_dists[dist][-1]), textcoords="offset points", xytext=(-2, 2), ha="center")

        plt.tight_layout()
        plt.legend()
        plt.savefig('blurred_plot.png')
        plt.clf()