import torch

from utils.common import dump_images


def generate_images(G, prior, outputs_dir, device):
    with torch.no_grad():
        images = G(prior.sample(64).to(device))
        dump_images(images, f'{outputs_dir}/test_samples.png')


def generate_all_const_prior(netG, prior, outputs_dir, device):
    import os
    from tqdm import tqdm
    n = len(prior.z)
    b = 64
    os.makedirs(os.path.join(outputs_dir, "fake_images"), exist_ok=True)
    with torch.no_grad():
        counter = 0
        if "const" in prior.prior_type:  # generate images for all 'm' zs
            n = min(n, len(prior.z))
            n_batches = n // b
            for i in tqdm(range(n_batches)):
                zs = prior.z[i * b: (i + 1) * b].to(device)
                fake_images = netG(zs)
                for i in range(len(fake_images)):
                    dump_images(fake_images[i].unsqueeze(0), os.path.join(outputs_dir, "fake_images", f'{counter}.png'))
                    counter += 1
            if n_batches * b < n:
                zs = prior.z[n_batches * b - n:].to(device)
                fake_images = netG(zs)
                for i in range(len(fake_images)):
                    dump_images(fake_images[i].unsqueeze(0), os.path.join(outputs_dir, "fake_images", f'{counter}.png'))
                    counter += 1