import torch
import sys
import os

from utils.common import batch_generation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses.optimal_transport import MiniBatchPatchLoss


def compare_real_fake_patch_dist(netG,  z_dim, data, dist='swd', p=8, s=4, outputs_dir=''):
    with torch.no_grad():
        patch_dist = MiniBatchPatchLoss(dist, p, s)

        fake_data = batch_generation(netG, z_dim, len(data), 512, data.device)

        distance = patch_dist(fake_data, data)
        if outputs_dir:
            with open(os.path.join(outputs_dir, 'OT.txt'), 'w') as f:
                f.write(f'Comparing {len(fake_data)} fake images to {len(data)} real images\n')
                f.write(f"{dist}: {distance:.4f}")

        print("Fake data generated", fake_data.shape)
        print("Dist:", distance)