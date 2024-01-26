import torch

from losses import get_loss_function
from utils.common import batch_generation


def compute_metrics(netG, prior, data, device):
    with torch.no_grad():
        fake_images = batch_generation(netG, prior, len(data), 64,
                                       inference_device=device,
                                       verbose=True)

        for metric_name in [
            'MiniBatchLoss-dist=w1-b=1024',
            'MiniBatchLoss-dist=swd',
            # 'MiniBatchPatchLoss-dist=swd-p=16-s=8',
        ]:
            print(f"\t - {metric_name}")
            metric = get_loss_function(metric_name)
            print(f'{metric_name}_fixed_noise_gen_to_train ', metric(fake_images.cpu(), data))

