import torch
from matplotlib import pyplot as plt

from scripts.EMD.dists import emd
from scripts.EMD.utils import get_data
from time import time


def time_metric(x,y, metric, n_reps=3):
    start = time()
    for  _ in range(n_reps):
        _ = metric(x,y)
    return (time() - start) / n_reps


if __name__ == '__main__':
    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_128'
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 10000, 20000, 30000]
    # sizes = [64, 256, 1024]
    data = get_data(data_path, 64, 3, limit_data=sizes[-1] + 64)

    sinkhorns = []
    emds = []
    for s in sizes:
        print(s)
        b1 = data[:s]
        mb = data[s:s+64]

        emds += [time_metric(mb, b1, emd)]
        sinkhorns += [time_metric(mb, b1, lambda a,b : emd(a,b, sinkhorn=True))]
        # b2bs += [time_metric(b1, b2, metric)]


    plt.plot(range(len(sizes)), emds, label="EMD", c='b')
    plt.plot(range(len(sizes)), sinkhorns, label="variable batch sizes", c='r')
    plt.legend()
    plt.xlabel("Batch-size")
    plt.ylabel("Compute time")
    plt.savefig("time_plot.png")