import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_plot_and_anotate(val_path, name, real):
    vals = pickle.load((open(val_path, "rb")))
    vals = np.nan_to_num(vals)
    # vals = vals / vals.max()
    xs = np.arange(len(vals)) / (len(vals) - 1)
    plt.plot(xs, vals, label='Fake')
    plt.annotate(f"{vals[-1]:.3f}", (0.9, vals[-1]), textcoords="offset points", xytext=(-2, 2), ha="center")
    plt.plot(xs, real*np.ones(len(vals)), label='Real')
    # plt.yscale('log')
    plt.title(name)
    plt.savefig(f"{name}.png")
    plt.clf()
    # ax.set_xscale('log')


root = 'outputs/26-1-2024/ffhq_const=35000_I-64_Z-64_Reg-GP_G-FastGAN_D-FastGAN'

w1_plot = os.path.join(root, 'plots', 'MiniBatchLoss-dist=w1_fixed_noise_gen_to_train.pkl')
load_plot_and_anotate(w1_plot, 'ImageW1', 39.887)

w1_plot = os.path.join(root, 'plots', 'MiniBatchPatchLoss-dist=swd-p=16-s=8_fixed_noise_gen_to_train.pkl')
load_plot_and_anotate(w1_plot, 'PatchSWD', 0.0015)