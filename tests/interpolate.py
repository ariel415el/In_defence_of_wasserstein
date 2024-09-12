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


def interpolate(encoder, decoder, data, seconds, fps, outputs_dir):
    with torch.no_grad():
        """Sample n_zs images and linearly interpolate between them in the latent space """
        os.makedirs(f'{outputs_dir}/interpolations', exist_ok=True)
        steps = int(fps * seconds / len(data) - 1)
        frame = 0
        f = 1 / (steps - 1)
        cur_z = encoder(data[0].unsqueeze(0))
        for i in range(len(data) - 1):
            next_z = encoder(data[i+1].unsqueeze(0))
            for a in range(steps):
                noise = next_z * a * f + cur_z * (1 - a * f)
                fake_imgs = decoder(noise).add(1).mul(0.5)
                vutils.save_image(fake_imgs, f'{outputs_dir}/interpolations/fakes-{frame}.png', normalize=False)
                frame += 1
            cur_z = next_z

    read_img_and_make_video(f'{outputs_dir}/interpolations', f'{outputs_dir}/interpolations.mp4', fps)