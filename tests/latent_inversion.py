import os
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import optim
from torchvision import utils as vutils
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import torch.nn.functional as F
from tests.test_utils import cut_around_center, compute_dists, sample_patch_centers


def plot_torch(real_images, image_inversions, patch_inversion, crop, out_path):
    resize = torchvision.transforms.Resize(real_images.shape[-2:], interpolation=InterpolationMode.NEAREST)
    out_img = torch.cat([real_images,
                         image_inversions,
                         resize(crop(real_images)),
                         resize(crop(image_inversions)),
                         resize(crop(patch_inversion)),
                         patch_inversion,
                         ]).add(1).mul(0.5)
    vutils.save_image(out_img, out_path, nrow=len(real_images), normalize=True)


def imshow(img, axs, title="img"):
    cmap = 'gray' if img.shape[0] == 1 else None
    axs.imshow((img.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2, cmap=cmap)
    axs.axis('off')
    axs.set_title(title)


def plot_plt(real_images, image_inversions, patch_inversion, crop, out_path, s=5):
    n = len(real_images)
    fig, ax = plt.subplots(nrows=6, ncols=n, figsize=(s * 6, s * n))
    for j in range (n):
        imshow(real_images[j], ax[0,j], "real_images")
        imshow(image_inversions[j], ax[1,j], "image_inversions")
        imshow(crop(real_images[j]), ax[2,j], "real_images_crop")
        imshow(crop(image_inversions[j]), ax[3,j], "image_inversions_crop")
        imshow(crop(patch_inversion[j]), ax[4,j], "patch_inversions_crop")
        imshow(patch_inversion[j], ax[5,j], "patch_inversion")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.clf()


def inverse_image(G, z_dim, real_images, outputs_dir, lr=0.01, n_steps=1000, p=32):
    os.makedirs(f"{outputs_dir}/inverse_z", exist_ok=True)
    center = sample_patch_centers(real_images.shape[-1], p, 1)[0]
    crop = lambda x: cut_around_center(x, center, p)

    image_noise = torch.randn((len(real_images), z_dim), requires_grad=True, device=real_images.device)
    patch_noise = torch.randn((len(real_images), z_dim), requires_grad=True, device=real_images.device)
    optimizer_image = optim.Adam([image_noise], lr=lr)
    optimizer_patch = optim.Adam([patch_noise], lr=lr)

    for iteration in tqdm(range(1000)):
        optimizer_image.zero_grad()
        g_images = G(image_noise)
        rec_loss = F.mse_loss(g_images, real_images)
        rec_loss.backward()
        optimizer_image.step()

        optimizer_patch.zero_grad()
        g_images_patch = G(patch_noise)
        rec_loss = F.mse_loss(crop(g_images_patch), crop(real_images))
        rec_loss.backward()
        optimizer_patch.step()

        if iteration % 100 == 0:
            # plot_torch(real_images, g_images, g_images_patch, crop, out_path=f'{outputs_dir}/inverse_z/rec_{iteration}.jpg')
            plot_plt(real_images, g_images, g_images_patch, crop, out_path=f'{outputs_dir}/inverse_z/rec_{iteration}.jpg')
