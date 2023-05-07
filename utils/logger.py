import os
import pickle
from collections import defaultdict

import numpy as np

from matplotlib import pyplot as plt
import json
import matplotlib as mpl

import wandb

mpl.use('Agg')

class PLTLogger:
    def __init__(self, args, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.data_avgs = defaultdict(list)
        self.data = defaultdict(list)

    def log(self, data_dict, step):
        for k, v in data_dict.items():
            self.data[k].append(float(v))

    def plot(self):
        # Plot single plots
        for k, v in self.data.items():
            if self.data[k]:
                self.data_avgs[k] += [np.mean(self.data[k])]
                self.data[k] = []
                plt.plot(np.arange(len(self.data_avgs[k])), self.data_avgs[k])
                plt.title(k)
                plt.savefig(self.save_dir + f"/{k}.png")
                plt.clf()


class WandbLogger:
    def __init__(self, args, save_dir):
        self.wandb = wandb.init(project=args.project_name, dir=save_dir, name=args.name)

    def log(self, val_dict, step):
        self.wandb.log(val_dict, step=step)

    def plot(self):
        pass

def get_dir(args):
    task_name = os.path.join(f"outputs", args.project_name,   args.name)
    saved_model_folder = os.path.join(task_name, 'models')
    saved_image_folder = os.path.join(task_name, 'images')
    plots_image_folder = os.path.join(task_name, 'plots')

    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)
    os.makedirs(plots_image_folder, exist_ok=True)

    with open(os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder, plots_image_folder


