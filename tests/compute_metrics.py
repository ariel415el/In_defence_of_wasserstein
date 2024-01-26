import torch

from losses import get_loss_function
from utils.common import batch_generation


def generate_images(netG, prior, data):
    with torch.no_grad():
        fake_images = batch_generation(netG, prior, len(data), 64,
                                       inference_device=torch.device("cuda:0"),
                                       org_device=torch.device("cpu"), verbose=True)

        for metric_name in [
            'MiniBatchLoss-dist=w1',
            'MiniBatchLoss-dist=swd',
            'MiniBatchPatchLoss-dist=swd-p=16-s=8',
        ]:
            print(f"\t - {metric_name}")
            metric = get_loss_function(metric_name)
            print(f'{metric_name}_fixed_noise_gen_to_train ', metric(fake_images.cpu(), data))

