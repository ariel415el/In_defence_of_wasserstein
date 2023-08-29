import torch
from torch.nn import functional as F

from losses import get_ot_plan, w1, SoftHingeLoss


def get_features(net, img, patch_wise=False):
    feature_maps = net.convs(img)
    if patch_wise:
        nc = feature_maps.shape[1]
        return feature_maps.permuete(0,2,3,1).reshape(-1, nc)
    else:
        return feature_maps.reshape(len(feature_maps), -1)


class AdverserialFeatureMatchingLoss(SoftHingeLoss):
    def trainG(self, netD, real_data, fake_data):
        """Train generator to minimize OT in discriminator features"""
        real_features = get_features(netD, real_data)
        fake_features = get_features(netD, fake_data)
        return w1(real_features, fake_features)


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