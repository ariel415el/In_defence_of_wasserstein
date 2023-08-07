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
    plot('/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/Exp3-patch-evidence',
        [
                '/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1',
                '/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training',
                '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ',
        ],
         {
             f"Evidence-.png": [
                 ("PixelWGAN-FC", ["L-WGANLoss", "D-FC"], []),
                 # ("PixelWGAN-GAP-22", ["L-WGANLoss", "D-PatchGAN-normalize=none-k=4"], ["PatchGAN-depth=4-normalize=none-k=4"]),
                 ("PixelWGAN-GAP-48", ["L-WGANLoss", "PatchGAN-depth=4-normalize=none-k=4"], []),
                 # ("PixelWGAN-DCGAN", ["L-WGANLoss", "D-DCGAN"], []),

                 ("Pixel-W1", ["MiniBatchLoss-dist=w1"], []),
                 # ("Pixel-W1-p=22-s=8", ["MiniBatchPatchLoss-dist=w1-p=22-s=8"], []),
                 ("Pixel-W1-p=48-s=16", ["MiniBatchPatchLoss-dist=w1-p=48-s=16"], []),

                 (f"Pixel-sinkhorn-epsilon={eps}", [f"MiniBatchLoss-dist=sinkhorn-epsilon={eps}"], []),
                 # (f"Pixel-sinkhorn-epsilon={eps}-p=22-s=8", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=22-s=8"], []),
                 (f"Pixel-sinkhorn-epsilon={eps}-p=48-s=16", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=48-s=16"], []),
             ]
         },
         seperate_plots=True
    )