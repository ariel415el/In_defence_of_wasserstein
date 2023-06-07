import numpy as np
import ot
import torch
from tqdm import tqdm


def L1_dist_matrix(X, Y):
    return torch.abs(X[:, None] - Y[None, :]).mean(-1)


def emd(x,y):
    uniform_x = np.ones(len(x)) / len(x)
    uniform_y = np.ones(len(y)) / len(y)
    # M = ot.dist(x, y) / x.shape[1]
    # M = efficient_L2_distances(x, y).cpu().numpy()
    M = batch_dist_matrix(x, y, b=512, dist_function=L1_dist_matrix).cpu().numpy()
    dist = ot.emd2(uniform_x, uniform_y, M)
    # dist = ot.sinkhorn2(uniform_x, uniform_y, M, 1)
    return dist


def swd(x, y, num_proj=1024):
    with torch.no_grad():
        b, dim = x.shape[0], np.prod(x.shape[1:])

        # Sample random normalized projections
        rand = torch.randn(num_proj, dim).to(x.device)
        rand = rand / torch.linalg.norm(rand, dim=1, keepdims=True)  # noramlize to unit directions

        # Sort and compute L1 loss
        projx = x @ rand.T
        projy = y @ rand.T

        projx = torch.sort(projx, dim=0)[0]
        projy = torch.sort(projy, dim=0)[0]

        loss = torch.abs(projx - projy).mean()

        return loss.item()


def efficient_L2_dist_matrix(X, Y):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    d = X.shape[1]
    dist /= d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist


def batch_dist_matrix(X, Y, b, dist_function):
    """
    For each x find the best index i s.t i = argmin_i(x,y_i)-f_i
    return the value and the argmin
    """
    dists = torch.ones(len(X), len(Y))
    n_batches = len(X) // b
    for i in range(n_batches):
        s = slice(i * b,(i + 1) * b)
        dists[s] = dist_function(X[s], Y).cpu()
    if len(X) % b != 0:
        s = slice(n_batches * b, len(X))
        dists[s] = dist_function(X[s], Y).cpu()

    return dists

def batch_NN(X, Y, f, b, dist_function):
    """
    For each x find the best index i s.t i = argmin_i(x,y_i)-f_i
    return the value and the argmin
    """
    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    NN_dists = torch.zeros(X.shape[0], device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        s = slice(i * b,(i + 1) * b)
        dists = dist_function(X[s], Y) - f
        NN_dists[s], NNs[s] = dists.min(1)
    if len(X) % b != 0:
        s = slice(n_batches * b, len(X))
        dists = dist_function(X[s], Y) - f
        NN_dists[s], NNs[s] = dists.min(1)

    return NN_dists, NNs


class W1:
    def __init__(self, b=512):
        self.b = b

    def score(self, x, y, f):
        return batch_NN(x, y, f, self.b, L1_dist_matrix)

    def loss(self, x, y):
        return torch.abs(x - y).sum(-1).mean(0)


def discrete_dual(x, y, n_steps=500, batch_size=None, lr=0.001, verbose=False):
    pbar = range(n_steps)
    if verbose:
        print(f"Optimizing duals: {x.shape}, {y.shape}")
        pbar = tqdm(pbar)

    if batch_size is None:
        batch_size = len(x)

    loss_func = W1(b=1024)
    psi = torch.zeros(len(x), requires_grad=True, device=x.device)
    opt_psi = torch.optim.Adam([psi], lr=lr)
    # scheduler = ReduceLROnPlateau(opt_psi, 'min', threshold=0.0001, patience=200)
    for _ in pbar:
        opt_psi.zero_grad()

        mini_batch = y[torch.randperm(len(y))[:batch_size]]

        phi, outputs_idx = loss_func.score(mini_batch, x, psi)

        dual_estimate = torch.mean(phi) + torch.mean(psi)

        loss = -1 * dual_estimate  # maximize over psi
        loss.backward()
        opt_psi.step()
        # scheduler.step(dual_estimate)
        if verbose:
            pbar.set_description(f"dual estimate: {dual_estimate.item()}, LR: {opt_psi.param_groups[0]['lr']}")

    return dual_estimate.item()



