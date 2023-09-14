import torch

from utils.common import dump_images, batch_generation


def generate_images(G, prior, outputs_dir, device):
    with torch.no_grad():
        images = batch_generation(G, prior, 1024, 512, device)
        dump_images(images, f'{outputs_dir}/test_samples.png')
