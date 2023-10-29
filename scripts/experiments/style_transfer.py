import os

import torch
from PIL import Image
import sys
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision import models, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.data import get_transforms


def imload(path, im_size, center_crop):
    transform = get_transforms(im_size, center_crop, False)
    img = Image.open(path).convert('RGB')
    img = transform(img)
    return img


class vgg_dist_calculator:
    def __init__(self, layer_indices=(4, 9, 18), device=torch.device("cpu")):
        self.device = device
        self.layer_indices = layer_indices
        self.vgg_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device)
        self.vgg_features.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def extract(self, X):
        feature_maps = []
        for i, layer in enumerate(self.vgg_features):
            X = layer(X)
            if i in self.layer_indices:
                feature_maps.append(X)
        return feature_maps


# def swd(x, y, num_proj=1024):
#     num_proj = int(num_proj)
#     x = x.reshape(x.shape[0], -1).T
#     y = y.reshape(y.shape[0], -1).T
#     assert (len(x.shape) == len(y.shape)) and x.shape[1] == y.shape[1]
#     _, d = x.shape
#
#     # Sample random normalized projections
#     rand = torch.randn(d, num_proj).to(x.device)  # (slice_size**2*ch)
#     rand = rand / torch.norm(rand, dim=0, keepdim=True)  # noramlize to unit directions
#
#     # Project images
#     projx = torch.mm(x, rand)
#     projy = torch.mm(y, rand)
#
#     # Sort and compute L1 loss
#     projx, _ = torch.sort(projx, dim=1)
#     projy, _ = torch.sort(projy, dim=1)
#
#     SWD = (projx - projy).abs().mean() # This is same for L2 and L1 since in 1d: .pow(2).sum(1).sqrt() == .pow(2).sqrt() == .abs()
#
#     return SWD


def channel_wd(x, y):
    assert (len(x.shape) == len(y.shape)) and x.shape[1] == y.shape[1]
    _, d = x.shape

    # Sample random normalized projections
    rand = torch.eye(d).to(x.device) / d # (slice_size**2*ch)

    # Project images
    projx = torch.mm(x, rand)
    projy = torch.mm(y, rand)

    # Sort and compute L1 loss
    projx, _ = torch.sort(projx, dim=1)
    projy, _ = torch.sort(projy, dim=1)

    SWD = (projx - projy).abs().mean() # This is same for L2 and L1 since in 1d: .pow(2).sum(1).sqrt() == .pow(2).sqrt() == .abs()

    return SWD


def gram_matrix(X):
    c, h, w = X.shape
    features = X.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(h * w)


def Gram_loss(X, Y):
    return torch.nn.MSELoss()(gram_matrix(X), gram_matrix(Y))

def L2(X, Y):
    return torch.nn.MSELoss()(X, Y)


def style_mix_optimization(content_img, style_img, lr, max_iter, style_weight=30, content_weight=1):
    optimized_img = content_img.clone()
    optimized_img.requires_grad_(True)

    optimizer = torch.optim.Adam([optimized_img], lr=lr)
    pbar = tqdm(range(max_iter + 1))

    style_features = extractor.extract(style_img)
    content_features = extractor.extract(content_img)
    for iteration in pbar:
        optimized_features = extractor.extract(optimized_img)
        total_loss = 0
        for opt_f, ref_f, ref_c in zip(optimized_features, style_features, content_features):
            style_loss = style_criteria(opt_f, ref_f)
            content_loss = L2(opt_f, ref_c)
            total_loss += style_loss * style_weight + content_loss * content_weight

        total_loss /= len(optimized_features)

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        pbar.set_description(f"total_loss: {total_loss.item()}")

    res = torch.clip(optimized_img, -1, 1)
    return res


if __name__ == '__main__':
    im_size=256
    from utils.distribution_metrics import swd, w1, sinkhorn
    # lr = 0.1; content_weight=0; style_criteria = Gram_loss
    # lr = 0.1; content_weight=0; style_criteria = lambda x,y: swd(x.reshape(x.shape[0], -1).T,y.reshape(y.shape[0], -1).T,num_proj=1024)[0]
    lr = 1; content_weight=0.001; style_criteria = lambda x,y: channel_wd(x.reshape(x.shape[0], -1).T,y.reshape(y.shape[0], -1).T)
    # lr =0.05; content_weight=0; style_criteria = lambda x,y: sinkhorn(x.reshape(x.shape[0], -1).T,y.reshape(y.shape[0], -1).T, epsilon=1)[0]
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    device = torch.device("cuda:0")
    content_img_path = '/mnt/storage_ssd/datasets/GPDM_images/style_transfer/content/cat1.jpg'
    style_img_path = '/mnt/storage_ssd/datasets/GPDM_images/style_transfer/style/starry_night.jpg'
    content_img = imload(content_img_path, im_size,400).to(device)
    style_img = imload(style_img_path,im_size,300).to(device)
    extractor = vgg_dist_calculator(layer_indices=[3, 8, 17], device=device)

    mix = style_mix_optimization(content_img, style_img, lr=lr, max_iter=100, style_weight=1, content_weight=content_weight)
    vutils.save_image(torch.stack([content_img, style_img, mix]), f"{outputs_dir}/style_transfer_lr_{lr}.png", normalize=True, nrow=3)