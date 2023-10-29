import torch
from torchvision import models, transforms
from utils.distribution_metrics import swd


class PerceptualSWD:
    def __init__(self, layer_indices=(19, ), **kwargs):
        self.device = None
        self.layer_indices = layer_indices
        self.vgg_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_features.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def extract(self, X):
        if self.device is None:
            self.device = X.device
        self.vgg_features.to(self.device)
        feature_maps = []
        for i, layer in enumerate(self.vgg_features):
            X = layer(X)
            if i in self.layer_indices:
                feature_maps.append(X)
        return feature_maps

    def compute(self, x, y):
        loss = 0
        for mapx, mapy in zip(self.extract(x), self.extract(y)):
            loss += swd(mapx.reshape(mapx.shape[0], -1).T, mapy.reshape(mapy.shape[0], -1).T, num_proj=1024)[0]
        return loss, {"Perecptual":loss.item()}

    def trainD(self, netD, real_data, fake_data):
        raise NotImplemented("MiniBatchLosses should be run with --n_D_steps 0")

    def trainG(self, netD, real_data, fake_data):
        return self.compute(real_data, fake_data)
