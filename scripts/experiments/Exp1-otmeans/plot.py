import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from plot_train_results import find_dir, find_last_image, plot


def create_plot(project_name, dataset):
    plot(f'outputs/{project_name}',
        [
                dataset
        ],
         {
             f"plot.png": [
                 (f"OTmeans", ["K-64"], []),
                 (f"BatchW1", ["L-MiniBatch"], []),
                 (f"CTGAN", ["L-CtransformLoss"], []),
             ]
         },
         seperate_plots=False, n=8
    )

if __name__ == '__main__':
    create_plot(project_name="Exp1-Discrete_GM", dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ')