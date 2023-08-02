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
    plot(f'outputs/GANs',
        [
                dataset
        ],
         {
             f"plot.png": [
                 (f"OTmeans", ["L-MiniBatch"], []),
                 (f"BatchW1", ["L-MiniBatch"], []),
                 (f"CTGAN", ["L-CtransformLoss"], []),
             ]
         },
         seperate_plots=False
    )

if __name__ == '__main__':
    create_plot(project_name="Exp1-Discrete_GM", dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ')