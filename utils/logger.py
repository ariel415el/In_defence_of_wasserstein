import os
import pickle
from collections import defaultdict

import numpy as np

from matplotlib import pyplot as plt
import json

COLORS=['r','g','b','k','pink', 'yellow']


class LossLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.aggregated_data = defaultdict(list)
        self.data = defaultdict(list)
        self.group_names = dict()
        self.log_file = f"{self.save_dir}/log.pkl"
        if os.path.exists(self.log_file):
            self.data = pickle.load(open(self.log_file, "rb"))

    def aggregate_data(self, data_dict, group_name):
        if group_name not in self.group_names:
            self.group_names[group_name] = list(data_dict.keys())
        for k, v in data_dict.items():
            self.aggregated_data[k].append(v)

    def add_data(self, data_dict, group_name):
        if group_name not in self.group_names:
            self.group_names[group_name] = list(data_dict.keys())
        for k, v in data_dict.items():
            self.data[k].append(v)

    def plot(self):
        # Add averaged aggregated data
        for k, v in self.aggregated_data.items():
            self.data[k].append(np.mean(v))
            self.aggregated_data[k] = []

        # Plot grouped data
        all_names = []
        for title, names in self.group_names.items():
            for i, name in enumerate(names):
                plt.plot(np.arange(len(self.data[name])), self.data[name], label=name, color=COLORS[i])
                all_names.append(name)
            plt.legend()
            plt.savefig(self.save_dir + f"/{title}.png")
            plt.clf()

        # Plot single plots
        for k, v in self.data.items():
            if k not in all_names:
                plt.plot(np.arange(len(self.data[k])), self.data[k])
                plt.title(k)
                plt.savefig(self.save_dir + f"/{k}.png")
                plt.clf()

        pickle.dump(self.data, open(self.log_file, "wb"))
def get_dir(args):
    task_name = os.path.join(args.outputs_root,  args.name)
    saved_model_folder = os.path.join(task_name, 'models')
    saved_image_folder = os.path.join(task_name, 'images')
    plots_image_folder = os.path.join(task_name, 'plots')

    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)
    os.makedirs(plots_image_folder, exist_ok=True)

    with open(os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder, plots_image_folder


