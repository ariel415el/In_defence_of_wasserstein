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
    plot('/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/W1_patch_evidence',
        [
                '/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1',
                '/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training',
                '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ',
        ],
         {
             f"Evidence-.png": [
                 ("PixelWGAN-FC", ["L-WGANLoss", "D-FC"], []),
                 ("PixelWGAN-GAP-22", ["L-WGANLoss", "D-PatchGAN-normalize=none-k=4"], []),
                 ("Pixel-W1", ["MiniBatchLoss-dist=w1"], []),
                 ("Pixel-W1-p=22-s=8", ["MiniBatchPatchLoss-dist=w1-p=22-s=8"], ["MiniBatchLoss-dist=w1"]),
             ]
         },
         seperate_plots=True
    )