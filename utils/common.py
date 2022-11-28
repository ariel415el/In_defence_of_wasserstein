from copy import deepcopy

import torch
import torch.nn.functional as F
import sys
def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def calc_gradient_penalty(netD, real_data, fake_data, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class NonSaturatingGANLoss:
    def trainD(self, netD, real_data, fake_data):
        preds = torch.cat([netD(real_data), netD(fake_data.detach())], dim=0).to(real_data.device)
        labels = torch.cat([torch.ones(len(real_data), 1), torch.zeros(len(fake_data), 1)], dim=0).to(real_data.device)
        Dloss = F.binary_cross_entropy_with_logits(preds, labels)
        return Dloss

    def trainG(self, netD, real_data, fake_data):
        preds = netD(fake_data)
        labels = torch.ones(len(real_data), 1).to(real_data.device)
        GLoss = F.binary_cross_entropy_with_logits(preds, labels)
        return GLoss

class WGANloss:
    def __init__(self, gp_factor=10):
        self.gp_factor = gp_factor

    def trainD(self, netD, real_data, fake_data):
        real_score = netD(real_data).mean()
        fake_score = netD(fake_data.detach()).mean()
        Dloss = fake_score - real_score
        if self.gp_factor >0:
            gp = calc_gradient_penalty(netD, real_data, fake_data, real_data.device)
            Dloss += self.gp_factor * gp
        return Dloss

    def trainG(self, netD, real_data, fake_data):
        Gloss = -netD(fake_data).mean()
        return Gloss


class SoftHingeLoss:
    """The core idea for hinge loss is: D is no longer be optimized when if performs good enough.
    Here we use Relu(-x) instead of min(x,0) since -min(-x,0) == max(x,0) == Relu(x)
    """
    def trainD(self, netD, real_data, fake_data):
        D_scores_real = netD(real_data)
        D_scores_fake = netD(fake_data.detach())
        D_loss_real = F.relu(torch.rand_like(real_data) * 0.2 + 0.8 - D_scores_real).mean()  # -min(-x,0) = max(x,0) = reul(x) =
        D_loss_fake = F.relu(torch.rand_like(fake_data) * 0.2 + 0.8 + D_scores_fake).mean()
        Dloss = D_loss_real + D_loss_fake
        return Dloss, {"D_scores_real": D_scores_real.mean().item(), "D_scores_fake": D_scores_fake.mean().item()}

    def trainG(self, netD, real_data, fake_data):
        D_scores_fake = netD(fake_data)
        # Gloss = -F.relu(torch.rand_like(fake_data) * 0.2 + 0.8 + D_scores_fake).mean()
        Gloss = -D_scores_fake.mean()
        return Gloss, {"D_scores_fake": D_scores_fake.mean().item()}

def get_loss_function(loss_name):
    return getattr(sys.modules[__name__], loss_name)()
