import numpy as np
import torch
from geomloss import SamplesLoss
from matplotlib import pyplot as plt

from scripts.EMD.dists import emd
from scripts.EMD.utils import get_data

if __name__ == '__main__':
    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ/FFHQ'
    # data = get_data(data_path, 64, 3, limit_data=70000)
    epsilon=1
    d = 20
    data = torch.randn(500, d)
    sh = SamplesLoss(loss="sinkhorn", p=1, blur=epsilon)
    losses_emd = []
    losses_sk = []
    losses_sk_2 = []
    # bs = [10, 100, 1000, 10000, len(data)//2]
    bs = np.arange(25,501,25, dtype=int)
    for b in bs:
        # r1 = data[:b]
        # r2 = data[-b:]
        r1 = torch.randn(b, d)
        r2 = torch.randn(b, d)
        losses_emd.append(emd(r1, r2))
        losses_sk.append(emd(r1, r2, sinkhorn=epsilon))
        losses_sk_2.append(sh(r1,r2))

    plt.plot(bs, losses_emd, color='r', label="EMD", alpha=0.5)
    plt.plot(bs, losses_sk, color='g', label=f"Sinkhorn-{epsilon}", alpha=0.5)
    plt.plot(bs, losses_sk_2, color='b', label=f"Sinkhorn_2-{epsilon}", alpha=0.5)
    plt.legend()
    plt.savefig("batch_size_effect.png")