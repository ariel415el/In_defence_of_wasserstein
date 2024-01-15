import numpy as np
import ot
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_utils import get_data
from utils.metrics import L2, compute_pairwise_distances_in_batches

metric = L2()


data = get_data('/mnt/storage_ssd/datasets/FFHQ/FFHQ_128', 64, c=3, center_crop=90)

b = data.shape[0] // 2
# x = torch.randn(b, 3 * 64 * 64)
# y = torch.randn(b, 3 * 64 * 64)
x = data[:b]
y = data[b:]
uniform_x = np.ones(b) / b
uniform_y = np.ones(b) / b

C = compute_pairwise_distances_in_batches(x, y, metric, bx=1000, by=1000).numpy()

# C = metric(x, y).numpy()
OTplan_sh = ot.bregman.sinkhorn(uniform_x, uniform_y, C, reg=1, verbose=True)
# OTplan_emd = ot.emd(uniform_x, uniform_y, C)
print(np.sum(OTplan_sh * C))

