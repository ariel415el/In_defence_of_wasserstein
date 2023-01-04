import os
import torch
from torchvision import utils as vutils


def interpolate(G, z_dim, n_zs, steps, outputs_dir, device):
    with torch.no_grad():
        """Sample n_zs images and linearly interpolate between them in the latent space """
        os.makedirs(f'{outputs_dir}/interpolations', exist_ok=True)
        cur_z = torch.randn((1, z_dim)).to(device)
        frame = 0
        f = 1 / (steps - 1)
        for i in range(n_zs):
            next_z = torch.randn((1, z_dim)).to(device)
            for a in range(steps):
                noise = next_z * a * f + cur_z * (1 - a * f)
                fake_imgs = G(noise).add(1).mul(0.5)
                vutils.save_image(fake_imgs, f'{outputs_dir}/interpolations/fakes-{frame}.png', normalize=False)
                frame += 1
            cur_z = next_z