import os

import torch

from losses import get_loss_function
from utils.common import batch_generation


def compute_metrics(netG, prior, data, outputs_dir, device):
    with torch.no_grad():
        n = len(data) // 2
        fake_images = batch_generation(netG, prior, n, 64,
                                       inference_device=device,
                                       verbose=True)

        f = open(os.path.join(outputs_dir, "distances.txt"), 'w')
        for metric_name in [
            'MiniBatchLoss-dist=w1-b=1024',
            'MiniBatchLoss-dist=swd',
            'MiniBatchPatchLoss-dist=swd-p=16-s=8',
        ]:
            metric = get_loss_function(metric_name)
            print(fake_images.shape, data[:n].shape, data[n:].shape)
            text = (f'fake: {metric_name}: {metric(fake_images.cpu(), data[n:]).item()}'
                    f'\nreal: {metric_name}: {metric(data[:n], data[n:]).item()}')
            print(text)
            f.write(f"{text}\n")

