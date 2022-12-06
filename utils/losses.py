import sys

import torch
from torch.nn import functional as F


class NonSaturatingGANLoss:
    def trainD(self, netD, real_data, fake_data):
        preds = torch.cat([netD(real_data), netD(fake_data.detach())], dim=0).to(real_data.device)
        labels = torch.cat([torch.ones(len(real_data), 1), torch.zeros(len(fake_data), 1)], dim=0).to(real_data.device)
        Dloss = F.binary_cross_entropy_with_logits(preds, labels)
        return Dloss

    def trainG(self, netD, real_data, fake_data):
        # A saturating loss is -1*BCE(fake_preds, 0) the non saturating is BCE(fake_preds, 1)
        preds = netD(fake_data)
        labels = torch.ones(len(real_data), 1).to(real_data.device)
        GLoss = F.binary_cross_entropy_with_logits(preds, labels)
        return GLoss


class WGANloss:
    def __init__(self, gp_factor=10):
        self.gp_factor = gp_factor

    def trainD(self, critic, real_data, fake_data):
        real_score = critic(real_data).mean()
        fake_score = critic(fake_data.detach()).mean()
        Dloss = fake_score - real_score
        debug_dict = {"real_score": real_score.item(), "fake_score": fake_score.item()}
        if self.gp_factor > 0:
            gp = calc_gradient_penalty(critic, real_data, fake_data, real_data.device)
            Dloss += self.gp_factor * gp
            debug_dict['gp'] = gp.item()
        return Dloss, debug_dict

    def trainG(self, netD, real_data, fake_data):
        Gloss = -netD(fake_data).mean()
        return Gloss, {"Gloss": Gloss.item()}


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
        # Gloss = -F.relu(torch.rand_like(fake_data) * 0.2 + 0.8 + D_scores_fake).mean()
        Gloss = -D_scores_fake.mean()
        return Gloss, {"D_scores_fake": D_scores_fake.mean().item()}


class CtransformLoss:
    def __init__(self, search_space='full', c1=0.1, c2=0.1):
        self.search_space = search_space
        self.c1 = c1
        self.c2 = c2

    @staticmethod
    def compute_ot(critic, batch, gen_batch, compute_penalties=False, run_in_batch=True):
        if run_in_batch:
            fs = critic(torch.cat([batch, gen_batch]))
        else:
            fs = torch.cat([critic(batch), critic(gen_batch)])

        C = torch.norm(batch[:, None, ...] - gen_batch[None, ...], p=1, dim=(2, 3, 4))
        f_cs = torch.min(C - fs[:, None], dim=0)[0]
        ot = fs.mean() + f_cs.mean().mean()

        if compute_penalties:
            admisibility_gap = C - fs[:, None] - f_cs[None, :]  # admisibility_gap[i,j] = norm(Bxs[i]-Bys[j]) - f[Bxs[i]] - f_C[Bys[j]]

            penalty1 = torch.mean(admisibility_gap**2)
            penalty2 = torch.mean(torch.clamp(admisibility_gap, max=0)**2)

            return ot, penalty1, penalty2
        return ot

    @staticmethod
    def compute_full_space_ot(f, reals, fakes, compute_penalties=True, run_in_batch=False):
        b = len(reals)
        B = torch.cat([reals, fakes], dim=0)
        # Bys = torch.cat([reals, fakes], dim=0)

        if run_in_batch:
            fs = f(B)
        else:
            fs = torch.cat([f(reals), f(fakes)])

        C = torch.norm(B[:, None, ...] - B[None, ...], p=1, dim=(2, 3, 4))
        f_cs = torch.min(C - fs[:, None], dim=0)[0]      # f_cs[j] = min_i {D[i,j] - f[i]}

        f_reals = fs[:b]
        f_c_fakes = f_cs[b:]

        OT = f_reals.mean() + f_c_fakes.mean()
        if compute_penalties:
            full_admisibility_gap = C - fs[:, None] - f_cs[None, :]  # full_admisibility_gap[i,j] = norm(Bxs[i]-Bys[j]) - f[Bxs[i]] - f_C[Bys[j]]
            couples_admisibility_gap = C[:b, b:] - f_reals[:, None] - f_c_fakes[None, :]

            penalty1 = torch.mean(couples_admisibility_gap**2)
            penalty2 = torch.mean(torch.clamp(full_admisibility_gap, min=0)**2)

            return OT, penalty1, penalty2

        else:
            return OT

    def trainD(self, netD, real_data, fake_data):
        OT, penalty1, penalty2 = CtransformLoss.compute_full_space_ot(netD, real_data, fake_data, compute_penalties=True, run_in_batch=True)
        Dloss = -OT + self.c1 * penalty1 + self.c2 * penalty2  # Maximize OT with penalties
        return Dloss, {"OT": OT.item(), "penalty1": penalty1.item(), "penalty2": penalty2.item()}

    def trainG(self, netD, real_data, fake_data):
        OT = CtransformLoss.compute_full_space_ot(netD, real_data, fake_data, compute_penalties=False, run_in_batch=True)
        Gloss = OT  # Minimize OT
        return Gloss, {"OT": OT.item()}


class AmortizedDualWasserstein:
    """
    Idea used in "Wasserstein GAN With Quadratic Transport Cost" and
    "A Two-Step Computation of the Exact GAN Wasserstein Distance"
    Solve the linear dual problem for each batch and regress the discriminator to one of the potentials
    """
    def trainD(self, netD, real_data, fake_data):
        OT, penalty1, penalty2 = CtransformLoss.compute_full_space_ot(netD, real_data, fake_data, compute_penalties=True, run_in_batch=True)
        Dloss = -OT + self.c1 * penalty1 + self.c2 * penalty2  # Maximize OT with penalties
        return Dloss, {"OT": OT.item(), "penalty1": penalty1.item(), "penalty2": penalty2.item()}

    def trainG(self, netD, real_data, fake_data):
        OT = CtransformLoss.compute_full_space_ot(netD, real_data, fake_data, compute_penalties=False, run_in_batch=True)
        Gloss = OT  # Minimize OT
        return Gloss, {"OT": OT.item()}


def get_loss_function(loss_name):
    return getattr(sys.modules[__name__], loss_name)()


#########################
# #### Utilities ###### #
#########################
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