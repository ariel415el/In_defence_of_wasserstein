import torch


class CtransformLoss:
    """Following c-transform WGAN from:
     "(p,q)-WGAN defined at Mallasto, Anton, et al. "(q, p)-Wasserstein GANs: Comparing Ground Metrics for Wasserstein GANs."
    Code inspired by https://github.com/sverdoot/qp-wgan
    """
    def __init__(self, search_space='full', c1=0.1, c2=0.1):
        self.search_space = search_space
        self.c1 = float(c1)
        self.c2 = float(c2)

    @staticmethod
    def compute_ot(critic, batch, gen_batch, compute_penalties=False):
        fs = critic(batch)

        # C = torch.mean(torch.abs(batch[:, None, ...] - gen_batch[None, ...]), dim=(2, 3, 4))
        C = torch.mean((batch[:, None] - gen_batch[None, :]) ** 2, dim=(-3, -2,-1))

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
            penalty2 = torch.mean(torch.clamp(full_admisibility_gap, max=0)**2)

            return OT, penalty1, penalty2

        else:
            return OT

    def trainD(self, netD, real_data, fake_data):
        OT, penalty1, penalty2 = CtransformLoss.compute_ot(netD, real_data, fake_data.detach(), compute_penalties=True)
        Dloss = -OT + self.c1 * penalty1 + self.c2 * penalty2  # Maximize OT with penalties
        debug_dict = {"CT-OT": OT.item()}

        from benchmarking.emd import EMD
        debug_dict['Primal-PT'] = EMD()(real_data, fake_data)

        return Dloss, debug_dict

    def trainG(self, netD, real_data, fake_data):
        OT = CtransformLoss.compute_ot(netD, real_data, fake_data, compute_penalties=False)
        Gloss = OT  # Minimize OT

        return Gloss, {"CT-OT": OT.item()}
