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
    eps = 100
    plot('/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/WGAN',
        [
            '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ',
        ],
         {
             f"WGAN.png": [
                 (f"Z-{z_prior}-WGAN-GAP-22", ["L-WGANLoss", "PatchGAN-depth=3-normalize=none-k=4", f"64x{z_prior}"], []),
                 (f"Z-{z_prior}-WGAN-GAP-48", ["L-WGANLoss", "PatchGAN-depth=4-normalize=none-k=4", f"64x{z_prior}"], []),

                 (f"Z-{z_prior}-sinkhorn-epsilon={eps}", [f"MiniBatchLoss-dist=sinkhorn-epsilon={eps}", f"64x{z_prior}"], []),
                 (f"Z-{z_prior}-sinkhorn-epsilon={eps}-p=22-s=8", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=22-s=8", f"64x{z_prior}"], []),
                 (f"Z-{z_prior}-sinkhorn-epsilon={eps}-p=48-s=16", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=48-s=16", f"64x{z_prior}"], []),
             ]
         for z_prior in ["const=64", "normal"]},
         seperate_plots=True
    )