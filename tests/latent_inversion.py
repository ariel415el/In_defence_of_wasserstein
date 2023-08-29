import os
import torch
import torchvision
from torch import optim
from torchvision import utils as vutils
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import torch.nn.functional as F
from tests.test_utils import cut_around_center, compute_dists, sample_patch_centers


def inverse_image(G, z_dim, real_images, outputs_dir, lr=0.01):
    os.makedirs(f"{outputs_dir}/inverse_z", exist_ok=True)
    p=22
    center = sample_patch_centers(real_images.shape[-1], p, 1)[0]
    crop = lambda x: cut_around_center(x, center, p)

    image_noise = torch.randn((len(real_images), z_dim), requires_grad=True, device=real_images.device)
    patch_noise = torch.randn((len(real_images), z_dim), requires_grad=True, device=real_images.device)
    optimizer_image = optim.Adam([image_noise], lr=lr)
    optimizer_patch = optim.Adam([patch_noise], lr=lr)

    for iteration in tqdm(range(10000)):
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


        resize = torchvision.transforms.Resize(real_images.shape[-2:], interpolation=InterpolationMode.NEAREST)
        if iteration % 100 == 0:
            print(rec_loss.item())
            out_img = torch.cat([real_images,
                                 g_images,
                                 resize(crop(real_images)),
                                 resize(crop(g_images)),
                                 resize(crop(g_images_patch)),
                                 g_images_patch,
                                 ]).add(1).mul(0.5)
            vutils.save_image(out_img, f'{outputs_dir}/inverse_z/rec_{iteration}.jpg',
                              nrow=len(real_images), normalize=True)

