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
    eps = 100
    plot(f'/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs/{project_name}',
        [
            dataset,
        ],
         {
             f"G-{gen_arch}-{z_prior}-WGAN.png": [
                 # (f"Z-{z_prior}-WGAN-GAP-22", ["L-WGANLoss", "PatchGAN-depth=3-normalize=none-k=4", f"64x{z_prior}", f"G-{gen_arch}"], []),
                 (f"Z-{z_prior}-WGAN-GAP-48", ["L-WGANLoss", "PatchGAN-depth=4-normalize=none-k=4", f"64x{z_prior}", f"G-{gen_arch}"], []),

                 (f"Z-{z_prior}-sinkhorn-epsilon={eps}", [f"MiniBatchLoss-dist=sinkhorn-epsilon={eps}", f"64x{z_prior}", f"G-{gen_arch}"], []),
                 # (f"Z-{z_prior}-sinkhorn-epsilon={eps}-p=22-s=8", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=22-s=8", f"64x{z_prior}", f"G-{gen_arch}"], []),
                 (f"Z-{z_prior}-sinkhorn-epsilon={eps}-p=48-s=16", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=48-s=16", f"64x{z_prior}", f"G-{gen_arch}"], []),
             ]
         for z_prior in ["const=64", "normal"] for gen_arch in ["FC","DCGAN"]},
         seperate_plots=True
    )

if __name__ == '__main__':
    create_plot(project_name="WGAN-1K", dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ')
    create_plot(project_name="WGAN-10K", dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ')