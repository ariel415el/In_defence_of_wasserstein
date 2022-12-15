import sys

import torch
from torch.nn import functional as F
import numpy as np
import cvxopt


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


class WGANLoss:
    def __init__(self, gp_factor=10):
        self.gp_factor = gp_factor

    def trainD(self, netD, real_data, fake_data):
        real_score = netD(real_data).mean()
        fake_score = netD(fake_data.detach()).mean()
        WD = real_score - fake_score
        Dloss = -1 * WD  # Maximize WD

        debug_dict = {"W1 ": WD.item()}
        if self.gp_factor > 0:
            gp = calc_gradient_penalty(netD, real_data, fake_data, real_data.device)
            Dloss += self.gp_factor * gp

        from benchmarking.emd import EMD
        debug_dict['Primal-OT'] = EMD()(real_data, fake_data)

        return Dloss, debug_dict

    def trainG(self, netD, real_data, fake_data):
        Gloss = -netD(fake_data).mean()
        return Gloss, {"Gloss": Gloss.item()}


class BatchW2D:
    """
    """
    def __init__(self, search_space='full'):
        self.dist_clc = vgg_dist_calculator()
        self.search_space = search_space

    def trainG(self, netD, real_data, fake_data):
        import ot
        uniform_x = np.ones(len(real_data)) / len(real_data)
        uniform_y = np.ones(len(fake_data)) / len(fake_data)
        with torch.no_grad():
            C = torch.mean((real_data[:, None] - fake_data[None, :]) ** 2, dim=(-3, -2,-1))
            # C = self.dist_clc.get_dist_mat(real_data, fake_data, b=64)
            C = C.cpu().numpy()
            Tmap = ot.emd(uniform_x, uniform_y, C)
            Nns = Tmap.argmax(0)
            OT = np.sum(Tmap * C)

        Gloss = ((fake_data - real_data[Nns])**2).mean()

        return Gloss, {"Gloss": Gloss.item(), "OT": OT.item()}


class CtransformLoss:
    """Following c-transform WGAN from:
     "(p,q)-WGAN defined at Mallasto, Anton, et al. "(q, p)-Wasserstein GANs: Comparing Ground Metrics for Wasserstein GANs."
    Code inspired by https://github.com/sverdoot/qp-wgan
    """
    def __init__(self, search_space='full', c1=0.1, c2=0.1):
        self.search_space = search_space
        self.c1 = c1
        self.c2 = c2

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


class AmortizedDualWasserstein:
    """
    Introduced in "A Two-Step Computation of the Exact GAN Wasserstein Distance"
    and used in "Wasserstein GAN With Quadratic Transport Cost"
    Solve the linear dual problem for each batch and regress the discriminator to one of the potentials
    Code inspired by https://github.com/harryliew/WGAN-QC.git

    dual furmulation is max{f,g} [a | b]^T @ [f | g] s.t f_i+g_j <= C_(i,j)
    and is solved in the primal form as min{f,g} [-a | -b]^T @ [f | g] s.t f_i+g_j <= C_(i,j)
    """
    def __init__(self):
        self.init = False
        self.d = None
        self.fi_gj = None
        self.ab = None
        self.criterion = None

    @staticmethod
    def fast_dist_mat(X, Y):
        dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
        d = X.shape[1]
        dist /= d
        return dist

    def set_foundations(self, d):
        fi_plus_gj_extraction_matrix = np.zeros((d ** 2, 2 * d))
        for i in range(d):
            for j in range(d):
                fi_plus_gj_extraction_matrix[i * d + j, i] = 1
                fi_plus_gj_extraction_matrix[i * d + j, d + j] = 1

        self.d = d
        self.fi_gj = cvxopt.sparse(cvxopt.matrix(fi_plus_gj_extraction_matrix))
        self.ab = cvxopt.matrix(-1 * np.ones(2 * d) / d)
        self.criterion = torch.nn.MSELoss()

        # this seem to make a faster computation
        self.pStart = {}
        self.pStart['x'] = cvxopt.matrix([cvxopt.matrix([1.0] * d), cvxopt.matrix([1.0] * d)])
        self.pStart['s'] = cvxopt.matrix([1.0] * (2 * d))

        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

    def solve_dual(self, reals, fakes):
        assert len(reals) == len(fakes)
        if not self.init:
            self.init = True
            self.set_foundations(len(reals))

        dist = self.fast_dist_mat(reals.reshape(self.d, -1), fakes.reshape(self.d, -1)).cpu().numpy()
        # dist = torch.mean((reals[:, None] - fakes[None, :]) ** 2, dim=(-3, -2,-1)).cpu().numpy()
        constraints = cvxopt.matrix(dist.reshape(-1).astype(np.float64))
        sol = cvxopt.solvers.lp(self.ab, self.fi_gj, constraints, primalstart=self.pStart, solver='glpk')

        x = np.array(sol['x'])[:, 0]
        # x = x - x.mean()  # Since the solution is shift invariant we may as well take take the zero shirt one ( This is true only in the +- formulation not in ++)
        fg = torch.from_numpy(x).to(reals.device).float()
        f = fg[:self.d]
        g = fg[self.d:]

        return f, g, -1 * sol['primal objective']

    def trainD(self, netD, real_data, fake_data):
        with torch.no_grad():
            f, g, WD = self.solve_dual(real_data, fake_data)
        # for i in range(self.n_iters):
        real_score = netD(real_data)
        real_score_mean = real_score.mean()
        fake_score = netD(fake_data.detach())

        L2LossD_fake = self.criterion(fake_score, g)
        L2LossD_real = self.criterion(real_score_mean, f.mean())
        Dloss = 0.5 * L2LossD_real  + 0.5 * L2LossD_fake
        WD_D = real_score_mean + fake_score.mean()

        debug_dict = {"WD_D":WD_D.item(), "Dual-OT": WD, "f+g":(f.mean() - g.mean()).item()}

        from benchmarking.emd import EMD
        debug_dict['Primal-OT'] = EMD()(real_data, fake_data)

        return Dloss, debug_dict

    def trainG(self, netD, real_data, fake_data):
        Gloss = -1*netD(fake_data).mean()
        return Gloss, {"Gloss": Gloss.item()}

#########################
# #### Utilities ###### #
#########################

def get_loss_function(loss_name):
    return getattr(sys.modules[__name__], loss_name)()


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


def get_batche_slices(n, b):
    n_batches = n // b
    slices = []
    for i in range(n_batches):
        slices.append(slice(i * b, (i + 1) * b))

    if n % b != 0:
        slices.append(slice(n_batches, n))
    return slices

class vgg_dist_calculator:
    def __init__(self,  layer=18):
        self.layer = layer  # [4, 9, 18]
        from torchvision import models, transforms
        self.vgg_features = models.vgg19(pretrained=True).features
        self.device = None
        self.vgg_features.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def extract_intermediate_feature_maps(self, x):
        for i, layer in enumerate(self.vgg_features):
            x = layer(x)
            # if i in layer_indices:
            if i == self.layer:
                return x

    def get_dist_mat(self, X, Y, b=64):
        if self.device is None:
            self.device = X.device
            self.vgg_features.to(self.device)
        x_slices = get_batche_slices(len(X), b)
        y_slices = get_batche_slices(len(Y), b)
        dists = -1 * torch.ones(len(X), len(Y))
        for slice_x in x_slices:
            features_x = self.extract_intermediate_feature_maps(X[slice_x])
            for slice_y in y_slices:
                features_y = self.extract_intermediate_feature_maps(Y[slice_x])
                D = features_x[:, None] - features_y[None, ]
                D = D.reshape(slice_x.stop - slice_x.start, slice_y.stop - slice_y.start, -1)
                dists[slice_x, slice_y] = torch.norm(D, dim=-1)

        return dists

if __name__ == '__main__':
    vgg_dist = vgg_dist_calculator(torch.device("cpu"))

    x = torch.ones(16,3,128,128)
    y = torch.zeros(16,3,128,128)

    d = vgg_dist.get_dist_mat(x,y)
    print(d.min())
