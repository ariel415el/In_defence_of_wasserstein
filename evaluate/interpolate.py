import os
import torch
from torchvision import utils as vutils
import cv2
from tqdm import tqdm

def read_img_and_make_video(dist, video_name, fps):
    img_array = []
    for i in tqdm(range(len(os.listdir(dist)))):
        try:
            filename = dist + f'/fakes-{i}.png'
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        except Exception as e:
            print(e, 'error at: %d' % i)

    if '.mp4' not in video_name:
        video_name += '.mp4'
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def interpolate(G, z_dim, n_zs, seconds, fps, outputs_dir, device):
    with torch.no_grad():
        """Sample n_zs images and linearly interpolate between them in the latent space """
        os.makedirs(f'{outputs_dir}/interpolations', exist_ok=True)
        cur_z = torch.randn((1, z_dim)).to(device)
        steps = int(fps * seconds / n_zs)
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

    read_img_and_make_video(f'{outputs_dir}/interpolations', f'{outputs_dir}/interpolations.mp4', fps)