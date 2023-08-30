import torch

from utils.dists import get_dist_metric


class CtransformLoss:
    """Following c-transform WGAN from:
     "(p,q)-WGAN defined at Mallasto, Anton, et al. "(q, p)-Wasserstein GANs: Comparing Ground Metrics for Wasserstein GANs."
    Code inspired by https://github.com/sverdoot/qp-wgan
    """
    def __init__(self, epsilon=0):
        self.base_metric = get_dist_metric("L2")
        self.epsilon = float(epsilon)

    def compute_ot(self, critic, batch, gen_batch):
        C = self.base_metric(batch.reshape(len(batch), -1), gen_batch.reshape(len(gen_batch), -1))
        fs = critic(batch)
        c_m_f = C - fs[:, None]
        if self.epsilon <= 0:
            f_cs = torch.min(c_m_f, dim=0)[0]
        else:
            from math import log
            f_cs = -1 * self.epsilon * (torch.logsumexp(c_m_f / self.epsilon, dim=0) - log(len(batch)))
            # f_cs = -1 * self.epsilon * torch.log(torch.exp(c_m_f / self.epsilon).mean(0))

        ot = fs.mean() + f_cs.mean().mean()

        return ot

    def trainD(self, netD, real_data, fake_data):
        OT = self.compute_ot(netD, real_data, fake_data.detach())
        Dloss = -OT # Maximize OT with penalties

        debug_dict = {"CT-OT": OT.item()}

        return Dloss, debug_dict

    def trainG(self, netD, real_data, fake_data):
        OT = self.compute_ot(netD, real_data, fake_data)
        Gloss = OT  # Minimize OT

        return Gloss, {"CT-OT": OT.item()}
