import torch
import os
from tqdm import tqdm
from utils.common import dump_images


def generate_images(G, prior, outputs_dir, device):
    with torch.no_grad():
        images = G(prior.sample(64).to(device))
        dump_images(images, f'{outputs_dir}/test_samples.png')


def generate_all_images(netG, prior, outputs_dir, device, args, save_each=False):
    all_fake_images = []
    with torch.no_grad():
        if "const" in prior.prior_type:
            n = len(prior.z)
        else:
            n = len(os.listdir(args['data_path']))
        b = 64
        counter = 0
        n_batches = n // b
        os.makedirs(os.path.join(outputs_dir, "fake_images"), exist_ok=True)
        for i in tqdm(range(n_batches)):
            if "const" in prior.prior_type:
                zs = prior.z[i * b: (i + 1) * b].to(device)
            else:
                zs = prior.sample(b)
            fake_images = netG(zs)
            if save_each:
                for i in range(len(fake_images)):
                    dump_images(fake_images[i].unsqueeze(0), os.path.join(outputs_dir, "fake_images", f'{counter}.png'))
                    counter += 1
            else:
                all_fake_images += [fake_images]
                counter += len(fake_images)
        if n_batches * b < n:
            if "const" in prior.prior_type:
                zs = prior.z[n_batches * b - n:].to(device)
            else:
                zs = prior.sample(b)
            fake_images = netG(zs)
            if save_each:
                for i in range(len(fake_images)):
                    dump_images(fake_images[i].unsqueeze(0), os.path.join(outputs_dir, "fake_images", f'{counter}.png'))
                    counter += 1
            else:
                all_fake_images += [fake_images]
                counter += len(fake_images)

        if not save_each:
            all_fake_images = torch.cat(all_fake_images, dim=0)
            torch.save(all_fake_images, os.path.join(outputs_dir, "fake_images.pth"))