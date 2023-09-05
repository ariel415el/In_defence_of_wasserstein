import torch

from losses import SoftHingeLoss
from utils.distribution_metrics import w1


def get_features(net, img, patch_wise=False):
    feature_maps = net.convs(img)
    if patch_wise:
        nc = feature_maps.shape[1]
        return feature_maps.permute(0,2,3,1).reshape(-1, nc)
    else:
        return feature_maps.reshape(len(feature_maps), -1)


class AdverserialFeatureMatchingLoss(SoftHingeLoss):
    def trainG(self, netD, real_data, fake_data):
        """Train generator to minimize OT in discriminator features"""
        real_features = get_features(netD, real_data)
        fake_features = get_features(netD, fake_data)

        real_features, _ = torch.sort(real_features.T, dim=1)
        fake_features, _ = torch.sort(fake_features.T, dim=1)

        SWD = (real_features - fake_features).pow(2).sum(1).sqrt().mean()

        return SWD, {"FeatutresSWD": SWD}


class FullAdverserialFeatureMatchingLoss(AdverserialFeatureMatchingLoss):
    def trainD(self, netD, real_data, fake_data):
        Dloss, debug = self.trainG(netD, real_data, fake_data)
        return -1*Dloss, debug


class AdverserialPatchFeatureMatchingLoss(SoftHingeLoss):
    def trainG(self, netD, real_data, fake_data):
        """Train generator to minimize OT in discriminator features"""
        real_features = get_features(netD, real_data, patch_wise=True)
        fake_features = get_features(netD, fake_data, patch_wise=True)
        return w1(real_features, fake_features)


class FullAdverserialPatchFeatureMatchingLoss(AdverserialPatchFeatureMatchingLoss):
    def trainD(self, netD, real_data, fake_data):
        Dloss, debug = self.trainG(netD, real_data, fake_data)
        return -1*Dloss, debug