import glob
import os
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from plot_train_results import find_dir, find_last_image, plot


if __name__ == '__main__':
    plot('/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/discrete_wgan',
        [
                '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'
        ],
         {
             f"{gen_arch}.png": [
                 (f"{gen_arch}WGAN-FC", ["L-WGANLoss", f"G-{gen_arch}", "D-FC-depth=3"], ["FC-depth=3-df=512"]),
                 (f"{gen_arch}WGAN-FC-512", ["L-WGANLoss", f"G-{gen_arch}", "FC-depth=3-df=512"], []),
                 (f"{gen_arch}WGAN-FC-5", ["L-WGANLoss", f"G-{gen_arch}", "D-FC-depth=5"], []),
                 (f"{gen_arch}-W1-FC", ["L-MiniBatchLoss-dist=w1", f"G-{gen_arch}"], []),
                 (f"{gen_arch}-Sinkhorn-100-FC", ["L-MiniBatchLoss-dist=sinkhorn-epsilon=100", f"G-{gen_arch}"], []),
             ]
             for gen_arch in ["Pixels", "FC"]
         },
         seperate_plots=False
    )