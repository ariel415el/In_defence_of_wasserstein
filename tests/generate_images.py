import numpy as np
import torch
from torchvision import utils as vutils


def generate_images(G, z_dim, outputs_dir, device):
    with torch.no_grad():
        """Sample n_zs images and linearly interpolate between them in the latent space """
        fake_imgs = G(torch.randn((1000, z_dim)).to(device))


        perm = torch.from_numpy(np.load('/mnt/storage_ssd/datasets/FFHQ/FFHQ64_1000_shuffled/perm.npy'))
        inverser_perm = np.argsort(perm)
        fake_imgs_inv = fake_imgs.reshape(len(fake_imgs), fake_imgs.shape[1], -1)[:, :, inverser_perm].reshape(fake_imgs.shape)


        vutils.save_image(fake_imgs_inv, f'{outputs_dir}/test_samples.png', normalize=True, nrow=int(np.sqrt(len(fake_imgs))))
