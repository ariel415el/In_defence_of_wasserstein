import torch
from torch.nn import functional as F


class NonSaturatingGANLoss:
    def trainD(self, netD, real_data, fake_data):
        preds = torch.cat([netD(real_data), netD(fake_data.detach())], dim=0).to(real_data.device).unsqueeze(1)
        labels = torch.cat([torch.ones(len(real_data), 1), torch.zeros(len(fake_data), 1)], dim=0).to(real_data.device)
        Dloss = F.binary_cross_entropy_with_logits(preds, labels)
        return Dloss, {"Dloss": Dloss.item()}

    def trainG(self, netD, real_data, fake_data):
        # A saturating loss is -1*BCE(fake_preds, 0) the non saturating is BCE(fake_preds, 1)
        preds = netD(fake_data).unsqueeze(1)
        labels = torch.ones(len(real_data), 1).to(real_data.device)
        GLoss = F.binary_cross_entropy_with_logits(preds, labels)
        return GLoss, {"GLoss": GLoss.item()}
