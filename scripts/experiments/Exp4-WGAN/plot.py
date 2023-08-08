import glob
import os
import pickle

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from plot_train_results import find_dir, find_last_image, plot

def create_plot(project_name, dataset):
    plot(f'/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/{project_name}',
        [
            dataset,
        ],
         {
             f"WGAN.png": [
                 # (f"Z-{z_prior}-WGAN-GAP-22", ["L-WGANLoss", "PatchGAN-depth=3-normalize=none-k=4", f"64x{z_prior}", f"G-{gen_arch}"], []),
                 # (f"Z-{z_prior}-WGAN-GAP-48", ["L-WGANLoss", "PatchGAN-depth=4-normalize=none-k=4", f"64x{z_prior}", f"G-{gen_arch}"], []),
                 (f"Z-{z_prior}", ["L-WGANLoss", f"64x{z_prior}"], [])

                 # (f"Z-{z_prior}-sinkhorn-epsilon={eps}", [f"MiniBatchLoss-dist=sinkhorn-epsilon={eps}", f"64x{z_prior}", f"G-{gen_arch}"], []),
                 # (f"Z-{z_prior}-sinkhorn-epsilon={eps}-p=22-s=8", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=22-s=8", f"64x{z_prior}", f"G-{gen_arch}"], []),
                 # (f"Z-{z_prior}-sinkhorn-epsilon={eps}-p=48-s=16", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=48-s=16", f"64x{z_prior}", f"G-{gen_arch}"], []),
             for z_prior in ["const=64", "const=512"] ]
         },
         seperate_plots=False
    )

if __name__ == '__main__':
    create_plot(project_name="WGAN-10K_2", dataset='/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1')
    create_plot(project_name="WGAN-10K_2", dataset='/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training')
    create_plot(project_name="WGAN-10K_2", dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ')