import os
from copy import deepcopy

import numpy as np
import torch
from torchvision import utils as vutils


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = deepcopy(model)
        self.forward_relu_outputs = []
        for pos, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.LeakyReLU):
                module.register_backward_hook(self.relu_backward_hook_function)
                module.register_forward_hook(self.relu_forward_hook_function)

    def relu_backward_hook_function(self, module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero
        """
        # Get last forward output
        corresponding_forward_output = self.forward_relu_outputs[-1]
        corresponding_forward_output[corresponding_forward_output > 0] = 1
        modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
        del self.forward_relu_outputs[-1]  # Remove last forward output
        return (modified_grad_out,)

    def relu_forward_hook_function(self, module, ten_in, ten_out):
        """
        Store results of forward pass
        """
        self.forward_relu_outputs.append(ten_out)

    def __call__(self, x):
        return self.model(x)


def get_score_gradient_map(classifier, images, mean=False):
    saliency_maps = []
    for i in range(len(images)):
        image = images[i].unsqueeze(0)
        image.requires_grad = True
        score = classifier(image)
        score.backward()
        saliency_maps.append(image.grad)

    saliency_maps = torch.cat(saliency_maps, dim=0)
    if mean:
        saliency_maps = torch.mean(saliency_maps, dim=1, keepdim=True)

    saliency_maps = torch.sigmoid(saliency_maps)

    return saliency_maps


def saliency_maps(netG, netD, z_dim, data, outputs_dir, device):
    os.makedirs(f"{outputs_dir}/discriminator_visualization", exist_ok=True)
    b = len(data)
    nrow = int(np.sqrt(b))

    with torch.no_grad():
        fixed_noise = torch.randn((b, z_dim), device=device)
        fakes = netG(fixed_noise)

    for name, images in [("Real", data), ("Fake", fakes)]:

        vutils.save_image(data,
                          f'{outputs_dir}/discriminator_visualization/{name}.png', nrow=nrow, normalize=True)
        vutils.save_image(get_score_gradient_map(netD, data),
                          f'{outputs_dir}/discriminator_visualization/{name}_gradients.png', nrow=nrow, normalize=True)

        vutils.save_image(get_score_gradient_map(GuidedBackprop(netD), data),
                          f'{outputs_dir}/discriminator_visualization/{name}_guided_gradients.png', nrow=nrow, normalize=True)
