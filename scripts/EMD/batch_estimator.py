from matplotlib import pyplot as plt

from scripts.EMD.dists import emd
from scripts.EMD.utils import get_data

if __name__ == '__main__':
    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ/FFHQ'
    data = get_data(data_path, 64, 3, limit_data=70000)

    losses_emd = []
    losses_sk = []
    bs = [10, 100, 1000, 10000, 35000]
    for b in bs:
        r1 = data[:b]
        r2 = data[-b:]

        losses_emd.append(emd(r1, r2))
        losses_sk.append(emd(r1, r2, sinkhorn=100))

    plt.plot(bs, losses_emd, color='r', label="EMD")
    plt.plot(bs, losses_sk, color='b', label="Sinkhorn-100")
    plt.savefig("batch_size_effect.png")