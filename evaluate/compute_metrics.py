import os

import torch

from losses import get_loss_function
from utils.common import batch_generation


def compute_metrics(netG, prior, data, outputs_dir, device):
    with torch.no_grad():
        fake_images = batch_generation(netG, prior, len(data), 64,
                                       inference_device=device,
                                       verbose=True)

        f = open(os.path.join(outputs_dir, "distances.txt"), 'w')
        for metric_name in [
            'MiniBatchLoss-dist=w1-b=1024',
            'MiniBatchLoss-dist=swd',
            'MiniBatchPatchLoss-dist=swd-p=16-s=8',
            'MiniBatchPatchLoss-dist=swd-p=8-s=4',
            'MiniBatchLocalPatchLoss-dist=swd-p=16-s=8',
            'MiniBatchLocalPatchLoss-dist=swd-p=8-s=4',
        ]:
            metric = get_loss_function(metric_name)
            print(fake_images.shape, data.shape)
            text = (f'fake: {metric_name}: {metric(fake_images.cpu(), data).item()}')
            print(text)
            f.write(f"{text}\n")

