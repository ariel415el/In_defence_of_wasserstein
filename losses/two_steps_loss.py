import cvxopt
import numpy as np
import torch


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
                fi_plus_gj_extraction_matrix[i * d + j, d + j] = -1

        self.d = d
        self.fi_gj = cvxopt.sparse(cvxopt.matrix(fi_plus_gj_extraction_matrix))
        self.ab = cvxopt.matrix(-1 * np.ones(2 * d) / d)
        self.ab[d:] *= -1
        self.criterion = torch.nn.MSELoss()

        # this seem to make a faster computation
        self.pStart = {}
        self.pStart['x'] = cvxopt.matrix([cvxopt.matrix([1.0] * d), cvxopt.matrix([-1.0] * d)])
        self.pStart['s'] = cvxopt.matrix([1.0] * (2 * d))

        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

    def solve_dual(self, reals, fakes):
        assert len(reals) == len(fakes)
        if not self.init:
            self.init = True
            self.set_foundations(len(reals))

        # dist = self.fast_dist_mat(reals.reshape(self.d, -1), fakes.reshape(self.d, -1)).cpu().numpy()
        dist = torch.mean((reals[:, None] - fakes[None, :]) ** 2, dim=(-3, -2,-1)).cpu().numpy()
        constraints = cvxopt.matrix(dist.reshape(-1).astype(np.float64))
        sol = cvxopt.solvers.lp(self.ab, self.fi_gj, constraints, primalstart=self.pStart, solver='glpk')

        x = np.array(sol['x'])[:, 0]
        x = x - x.mean()  # Since the solution is shift invariant we may as well take the zero shirt one ( This is true only in the +- formulation not in ++)
        fg = torch.from_numpy(x).to(reals.device).float()
        f = fg[:self.d]
        g = fg[self.d:]

        return f, g, -1 * sol['primal objective']

    def trainD(self, netD, real_data, fake_data):
        with torch.no_grad():
            f, g, WD = self.solve_dual(real_data, fake_data)
        # for i in range(self.n_iters):
        real_score_mean = netD(real_data).mean()
        fake_score = netD(fake_data.detach())

        L2LossD_fake = self.criterion(fake_score, g)
        L2LossD_real = self.criterion(real_score_mean, f.mean())
        Dloss = 0.5 * L2LossD_real + 0.5 * L2LossD_fake
        WD_D = real_score_mean - fake_score.mean()

        debug_dict = {"WD_D":WD_D.item(),  "Dual-OT": WD, "f": f.mean().item(), "g": g.mean().item()}

        return Dloss, debug_dict

    def trainG(self, netD, real_data, fake_data):
        Gloss = -1*netD(fake_data).mean()
        return Gloss, {"Gloss": Gloss.item()}
