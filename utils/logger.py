import os
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

    def aggregate_data(self, data_dict):
        for k, v in data_dict.items():
            self.aggregated_data[k].append(v)

    def add_data(self, data_dict):
        for k, v in data_dict.items():
            self.data[k].append(v)
    def plot(self, data_group_names):
        # Add averaged aggregated data
        for k, v in self.aggregated_data.items():
            self.data[k].append(np.mean(v))
            self.aggregated_data[k] = []

        # Plot grouped data
        all_names = []
        for title, names in data_group_names.items():
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




def get_dir(args):
    task_name = 'train_results/' + args.name
    saved_model_folder = os.path.join( task_name, 'models')
    saved_image_folder = os.path.join( task_name, 'images')
    
    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    # for f in os.listdir('./'):
    #     if '.py' in f:
    #         shutil.copy(f, task_name+'/'+f)
    
    with open( os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder


