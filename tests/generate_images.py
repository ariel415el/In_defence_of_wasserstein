import torch

from utils.common import dump_images


def generate_images(G, z_dim, outputs_dir, device):
    with torch.no_grad():
        """Sample n_zs images and linearly interpolate between them in the latent space """
        z = torch.randn((64, z_dim))

        # for i in range(G.n):
        #     fake_imgs = G.models[i](z)
        #     dump_images(fake_imgs, f'{outputs_dir}/test_samples-{i}.png')

        fake_imgs = G(z).to(device)
        dump_images(fake_imgs, f'{outputs_dir}/test_samples.png')
