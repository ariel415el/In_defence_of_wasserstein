import itertools
import json
import os

import numpy as np
import torch
from torch import optim
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models import get_generator
from models.FastGAN import Generator
from torchvision import utils as vutils
import torch.nn.functional as F

from utils.data import get_transforms


def get_data(data_root, im_size, center_crop, limit_data=None):
    """Load entire dataset to memory as a single batch"""
    T = get_transforms(im_size, center_crop)

    images = []
    print("Loading data to memory to find NNs")
    img_names = os.listdir(data_root)
    if limit_data is not None:
        img_names = img_names[:limit_data]
    for fname in tqdm(img_names):
        im = Image.open(os.path.join(data_root, fname))
        im = T(im)
        images += [im]

    return torch.stack(images)


def interpolate(G, n_zs=10, steps=10):
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


def _batch_nns(img, data, center, p, s):
    h = p // 2
    query_patch = img[..., center[0]-h:center[0]+h, center[1]-h:center[1]+h]
    refs = data[..., center[0]-h-s:center[0]+h+s, center[1]-h-s:center[1]+h+s]
    refs = F.unfold(refs, kernel_size=p, stride=1) # shape (b, 3*p*p, N_patches)
    refs = refs.permute(0, 2, 1).reshape(-1, 3, p, p)

    dists = (refs - query_patch).reshape(refs.shape[0], -1)
    rgb_nn_index = torch.sort(torch.norm(dists, dim=1, p=1))[1][0]

    dists = (refs.mean(1) - query_patch.mean(1)).reshape(refs.shape[0], -1) # Compare gray scale images
    gs_nn_index = torch.sort(torch.norm(dists, dim=1, p=1))[1][0]

    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).view((1,1,3,3))
    dists = (F.conv2d(torch.mean(refs, dim=1, keepdim=True), a) - F.conv2d(torch.mean(query_patch, dim=1, keepdim=True), a)).reshape(refs.shape[0], -1)
    edge_nn_index = torch.sort(torch.norm(dists, dim=1, p=1))[1][0]

    return refs[rgb_nn_index].clone(), refs[gs_nn_index].clone(), refs[edge_nn_index].clone()

def find_patch_nns(G, data, patch_size=24, stride=4, search_margin=6):
    """
    Search for nearest patch in data to patches from generated images.
    Search is performed in a constrained locality of the query patch location
    """
    out_dir = f'{outputs_dir}/patch_nns(p-{patch_size}_s-{search_margin})'
    os.makedirs(out_dir, exist_ok=True)
    h = patch_size // 2
    img_dim = data.shape[-1]
    centers = np.arange(patch_size, img_dim - patch_size, stride)
    centers = list(itertools.product(centers, repeat=2))

    print(f"Searching for {patch_size}x{patch_size} patch nns in {len(data)} data samples:")
    for j in range(5):
        query_image = G(torch.randn(1, z_dim).to(device))
        vutils.save_image(query_image.add(1).mul(0.5), f'{out_dir}/query_img-{j}.png', normalize=False, nrow=2)

        rgb_NN_patches = []
        gs_NN_patches = []
        edge_NN_patches = []
        q_patches = []
        for i, center in enumerate(tqdm(centers)):
            q_patches.append(query_image[..., center[0]-h:center[0]+h, center[1]-h:center[1]+h])
            rgb_nn_patch, gs_nn_patch, edge_nn_patch = _batch_nns(query_image, data, center, p=patch_size, s=search_margin)
            rgb_NN_patches.append(rgb_nn_patch.unsqueeze(0))
            gs_NN_patches.append(gs_nn_patch.unsqueeze(0))
            edge_NN_patches.append(edge_nn_patch.unsqueeze(0))

        x = torch.cat(q_patches + rgb_NN_patches + gs_NN_patches + edge_NN_patches, dim=0)
        vutils.save_image(x.add(1).mul(0.5), f'{out_dir}/patches-{j}.png', normalize=False, nrow=len(q_patches))


def find_nns(G, z_dim, data):
    import lpips
    os.makedirs(f'{outputs_dir}/nns', exist_ok=True)
    percept = lpips.LPIPS(net='vgg', lpips=False).to(device)
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
    device = torch.device('cpu')
    model_dir = 'Outputs/square_data_64x64_G-DCGAN_D-DCGAN_L-SoftHingeLoss_Z-64_B-16_test'
    ckpt_path = f'{model_dir}/models/100000.pth'  # path to the checkpoint
    outputs_dir = f'{model_dir}/test_outputs'

    os.makedirs(outputs_dir, exist_ok=True)
    args = json.load(open(os.path.join(model_dir, "args.txt")))
    z_dim = args['z_dim']
    data_root = args['data_path']

    G = get_generator(args['gen_arch'], args['im_size'], args['z_dim'])
    weights = torch.load(ckpt_path)['netG']
    # weights = {k.replace('module.', ''): v for k, v in weights.items()}
    # weights = {k.replace('network', 'init.init'): v for k, v in weights.items()}
    G.load_state_dict(weights)
    G.to(device)
    G.eval()

    data = get_data(args['data_path'], args['im_size'], args['center_crop'], limit_data=None)
    data = data.to(device)


    with torch.no_grad():
        interpolate(G, n_zs=15)
        find_nns(G, z_dim, data)
        find_patch_nns(G, data, patch_size=16, stride=4, search_margin=6)
        find_patch_nns(G, data, patch_size=24, stride=4, search_margin=6)

    # inverse_image(G, data[:10])
