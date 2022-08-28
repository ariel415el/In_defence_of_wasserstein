import os
from collections import defaultdict

import numpy as np

from matplotlib import pyplot as plt
import shutil
import json


class LossLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.other_losses = defaultdict(list)
        self.G_losses = []
        self.D_losses_real = []
        self.D_losses_fake = []
        self.G_losses_avg = []
        self.D_losses_real_avg = []
        self.D_losses_fake_avg = []

        self.D_losses_real_train = []
        self.D_losses_real_test = []

    def aggregate_train_losses(self, G_loss, D_loss_real, D_loss_fake):
        self.G_losses.append(G_loss)
        self.D_losses_real.append(D_loss_real)
        self.D_losses_fake.append(D_loss_fake)

    def log_eval_losses(self, dloss_real_train, dloss_real_test):
        self.D_losses_real_train += [dloss_real_train]
        self.D_losses_real_test += [dloss_real_test]

    def log_other_losses(self, losses):
        for k, v in losses.items():
            self.other_losses[k].append(v)

    def plot(self):
        for name, losses in self.other_losses.items():
            plt.plot(np.arange(len(losses)), losses)
            plt.title(name)
            plt.savefig(self.save_dir + f"/{name}.png")
            plt.clf()

        x_axis = np.arange(len(self.D_losses_real_train))
        plt.plot(x_axis, self.D_losses_real_train, label='D_losses_real_train', c='r')
        plt.plot(x_axis, self.D_losses_real_test, label='D_losses_real_test', c='b')
        plt.legend()
        plt.title("eval_losses")
        plt.savefig(self.save_dir + f"/D_test.png")
        plt.clf()

        self.G_losses_avg.append(np.mean(self.G_losses))
        self.D_losses_real_avg.append(np.mean(self.D_losses_real))
        self.D_losses_fake_avg.append(np.mean(self.D_losses_fake))
        self.G_losses = []
        self.D_losses_real = []
        self.D_losses_fake = []

        x_axis = np.arange(len(self.G_losses_avg))
        # plt.plot(x_axis, self.G_losses_avg, label='G_losses', c='r')
        plt.plot(x_axis, self.D_losses_real_avg, label='D_losses_real', c='b')
        plt.plot(x_axis, self.D_losses_fake_avg, label='D_losses_fake', c='g')
        plt.legend()
        plt.title("D_train")
        plt.savefig(self.save_dir + f"/Train_losses.png")
        plt.clf()


def get_dir(args):
    task_name = 'train_results/' + args.name
    saved_model_folder = os.path.join( task_name, 'models')
    saved_image_folder = os.path.join( task_name, 'images')
    
    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    for f in os.listdir('./'):
        if '.py' in f:
            shutil.copy(f, task_name+'/'+f)
    
    with open( os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder


