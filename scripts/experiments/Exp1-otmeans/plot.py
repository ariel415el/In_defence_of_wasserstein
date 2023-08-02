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
                dataset
        ],
         {
             f"{gen_arch}.png": [
                 (f"WGAN-FC-1024", ["L-WGANLoss", f"G-{gen_arch}", "D-FC-df=1024"], []),
                 (f"CTGAN", ["L-CtransformLoss", f"G-{gen_arch}"], []),
                 # (f"W1", ["L-MiniBatchLoss-dist=w1", f"G-{gen_arch}"], []),
                 #(f"Sinkhorn-100-FC", ["L-MiniBatchLoss-dist=sinkhorn-epsilon=100", f"G-{gen_arch}"], []),
             ]
             for gen_arch in ["Pixels", "FC"]
         },
         seperate_plots=False
    )

if __name__ == '__main__':
    # create_plot(project_name="discreteWGAN-1k", dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ')
    create_plot(project_name="discreteWGAN-10k", dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ')