import copy
import os

import torch
import torchvision.transforms
from torchvision.utils import make_grid


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = copy.deepcopy(model)
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model._modules.items():
            x = module(x)  # Forward
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        return conv_output, x


def generate_cam(model, input_image, target_layer="down_16"):
    extractor = CamExtractor(model, target_layer)
    shape = input_image.shape[-2:]
    # Full forward pass
    # conv_output is the output of convolutions at specified layer
    # model_output is the final output of the model (1, 1000)
    conv_output, model_output = extractor.forward_pass(input_image)

    # Zero grads
    model.zero_grad()
    # Backward pass with specified target
    model_output.backward(retain_graph=True)
    # Get hooked gradients
    guided_gradients = extractor.gradients.data
    # Get convolution outputs
    target = conv_output.data
    # Get weights from gradients
    weights = torch.mean(guided_gradients, axis=(2, 3), keepdims=True)  # Take averages for each gradient
    # Create empty numpy array for cam
    cam = torch.mean(target*weights, dim=1, keepdim=True)
    cam = torchvision.transforms.Resize(shape, antialias=True, interpolation='lanczos')(cam)
    return cam


if __name__ == '__main__':
    import json
    from tests.test_utils import load_pretrained_discriminator, get_data
    from torchvision import utils as vutils

    model_dir = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/Outputs/FFHQ_128_64x64_G-DCGAN_D-DCGAN_L-NonSaturatingGANLoss_Z-64_B-64_test'
    device = torch.device("cpu")

    args = json.load(open(os.path.join(model_dir, "args.txt")))
    z_dim = args['z_dim']
    data_root = args['data_path']

    ckpt_path = f'{model_dir}/models/63000.pth'  # path to the checkpoint
    print("Loading models", end='...')
    netD = load_pretrained_discriminator(args, ckpt_path, device)
    print("Done")

    b = 1
    data = get_data(args['data_path'], args['im_size'], args['center_crop'], limit_data=b)

    vutils.save_image(data, f'reals.png', normalize=True)

    cam = generate_cam(netD, data[0].unsqueeze(0), "down_16")

    cam = 0.5 * make_grid(cam, normalize=True) + 0.5 * make_grid(data, normalize=True)

    vutils.save_image(cam, f'GRAD-CAM.png')
