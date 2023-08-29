mport torch
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
    metric, metric_name = emd, "EMD"
    sizes = [64, 256, 1024, 4096, 10000, 30000]
    # sizes = [64, 256, 1024]
    data = get_data(data_path, 64, 3, limit_data=sizes[-1] + 64)

    b2bs = []
    mb2b = []
    for s in sizes:
        print(s)
        mb = data[:64]
        b1 = data[64:64+s]
        # b2 = data[64+s:2*s+s]

        mb2b += [time_metric(mb, b1, metric)]
        # b2bs += [time_metric(b1, b2, metric)]


    plt.plot(sizes, mb2b, label="fixed (64) to variable batch size", c='b')
    # plt.plot(sizes, b2bs, label="variable batch sizes", c='r')
    plt.legend()
    plt.xlabel("Batch-size")
    plt.ylabel("Compute time")
    plt.savefig("time_plot.png")