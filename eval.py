import itertools
import json
import os

import numpy as np
import torch
from torch import optim
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.FastGAN import Generator
from torchvision import utils as vutils

from utils.data import get_datasets


def get_data(data_root):
    T = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images = []
    for fname in os.listdir(data_root):
        im = Image.open(os.path.join(data_root, fname))
        im = T(im)
        images += [im]

    return torch.stack(images)


def interpolate(G, z_dim, n_zs=10, steps=10):
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


def _batch_nns(img, data, row_interval, col_interval):
    if row_interval is not None:
        img = img[..., row_interval[0]:row_interval[1], col_interval[0]: col_interval[1]]
        ref = data[..., row_interval[0]:row_interval[1], col_interval[0]: col_interval[1]]

    dists = (ref - img).reshape(ref.shape[0], -1)
    return torch.sort(torch.norm(dists, dim=1))[1]


def find_patch_nns(G, z_dim, data):
    os.makedirs(f'{outputs_dir}/patch_nns', exist_ok=True)
    with torch.no_grad():
        for i in range(5):
            query_image = G(torch.randn(1, z_dim).to(device))
            # x = _find_crop_nns(fake_img, data)

            NN_images = []
            img_dim = query_image.shape[-1]
            d = img_dim // 4

            all_slices = np.arange(d, img_dim - d, d)
            all_slices = list(zip(all_slices, all_slices + d))
            all_pairs_of_slices = itertools.product(all_slices, repeat=2)
            for i, (row_interval, col_interval) in enumerate(all_pairs_of_slices):
                nn_indices = _batch_nns(query_image, data, row_interval, col_interval)
                nn = data[nn_indices[0].cpu().numpy()].clone() * 0.5
                nn[..., row_interval[0]:row_interval[1], col_interval[0]: col_interval[1]] *= 2
                NN_images.append(nn.unsqueeze(0))

            x = torch.cat([query_image] + NN_images, dim=0)

            vutils.save_image(x.add(1).mul(0.5), f'{outputs_dir}/patch_nns/im-{i}.png', normalize=False)


def find_nns(G, data):
    import lpips
    os.makedirs(f'{outputs_dir}/nns', exist_ok=True)
    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
    results = []
    for i in range(8):
        fake_image = G(torch.randn((1, z_dim), device=device))
        dists = [percept(fake_image, data[i].unsqueeze(0)).sum().item() for i in range(len(data))]
        nn_indices = np.argsort(dists)
        nns = data[nn_indices[:4]]

        results.append(torch.cat([fake_image, nns]))

    vutils.save_image(torch.cat(results, dim=0).add(1).mul(0.5), f'{outputs_dir}/nns/im.png', normalize=False, nrow=5)


def inverse_image(G, z_dim, real_images):
    import torch.nn.functional as F
    os.makedirs(f"{outputs_dir}/inverse_z", exist_ok=True)

    fixed_noise = torch.randn((len(real_images), z_dim), requires_grad=True, device=device)
    optimizerG = optim.Adam([fixed_noise], lr=0.001)

    for iteration in tqdm(range(10000)):
        optimizerG.zero_grad()

        g_images = G(fixed_noise)

        # rec_loss = percept(F.avg_pool2d(g_images, 2, 2), F.avg_pool2d(real_images, 2, 2)).sum() + 0.2 * F.mse_loss(g_images, real_images)
        rec_loss = F.mse_loss(g_images, real_images)

        rec_loss.backward()

        optimizerG.step()

        if iteration % 100 == 0:
            vutils.save_image(torch.cat([real_images, g_images]).add(1).mul(0.5), f'{outputs_dir}/inverse_z/rec_{iteration}.jpg',
                              nrow=len(real_images))


if __name__ == '__main__':
    model_dir = 'train_results/AE-VGG-Obama_FastGAN_Z-128_B-8'
    args = json.load(open(os.path.join(model_dir, "args.txt")))

    paths = [os.path.join(model_dir, 'models', name) for name in os.listdir(os.path.join(model_dir, 'models',))]
    ckpt_path = max(paths, key=os.path.getctime)

    outputs_dir = os.path.join(model_dir, 'test_outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    device = torch.device('cuda:0')

    G = Generator(z_dim=args['z_dim'])
    G.load_state_dict(torch.load(ckpt_path)['D'])
    G.to(device)
    # G.eval()

    with torch.no_grad():
        interpolate(G, z_dim=args['z_dim'], n_zs=15)

    data = get_data(args['data_path']).to(device)

    with torch.no_grad():
        # find_nns(G, data)
        find_patch_nns(G, z_dim=args['z_dim'], data=data)

    inverse_image(G, args['z_dim'], data[:10])