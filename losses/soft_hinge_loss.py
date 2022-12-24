import torch
from torch.nn import functional as F


class SoftHingeLoss:
    """
    The core idea for hinge loss is: D is no longer be optimized when if performs good enough.
    Here we use Relu(-x) instead of min(x,0) since -min(-x,0) == max(x,0) == Relu(x)
    See "Geometric GAN" by Lim et al. for more details
    """
    def trainD(self, netD, real_data, fake_data):
        D_scores_real = netD(real_data)
        D_scores_fake = netD(fake_data.detach())
        D_loss_real = F.relu(torch.rand_like(D_scores_real) * 0.2 + 0.8 - D_scores_real).mean()  # -min(-x,0) = max(x,0) = reul(x) =
        D_loss_fake = F.relu(torch.rand_like(D_scores_fake) * 0.2 + 0.8 + D_scores_fake).mean()
        Dloss = D_loss_real + D_loss_fake
        return Dloss, {"D_scores_real": D_scores_real.mean().item(), "D_scores_fake": D_scores_fake.mean().item()}

    def trainG(self, netD, real_data, fake_data):
        D_scores_fake = netD(fake_data)
        Gloss = -D_scores_fake.mean()
        return Gloss, {"D_scores_fake": D_scores_fake.mean().item()}
