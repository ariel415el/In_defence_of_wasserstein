import subprocess
from time import sleep, strftime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch


def send_tasks(datasets):
    project_name = f"discrete_wgan"
    hours = 8
    for dataset, dataset_params in datasets:
        for gen_arch in ["Pixels", "FC-depth=3"]:
            base = f"python3 train.py  --data_path {dataset}  {dataset_params}" \
                   f" --load_data_to_memory --n_workers 0 --project_name {project_name} --z_prior const=64" \
                   f"  --n_iterations 100000 " \
                   f"--gen_arch {gen_arch} "

            # run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrG 0.0001 --lrD 0.0001  --G_step_every 5 --disc_arch FC-depth=3",
            #            f"PixelWGAN-FC", hours)
            #
            # run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrG 0.0001 --lrD 0.0001 --G_step_every 5 --disc_arch FC-depth=3-df=512",
            #            f"PixelWGAN-FC-512", hours)
            #
            # run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrG 0.0001 --lrD 0.0001 --G_step_every 5 --disc_arch FC-depth=5",
            #            f"PixelWGAN-FC-5", hours)
            #
            # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=w1 --lrG 0.0001 --D_step_every -1",
            #            f"Pixel-W1", hours)

            run_sbatch(base + f" --loss_function MiniBatchLoss-dist=sinkhorn-epsilon=100 --lrG 0.001 --D_step_every -1",
                       f"Pixel-sinkhorn100", hours)





if __name__ == '__main__':
    datasets = [
        # ('/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1',' --gray_scale'),
        # ('/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training',' --gray_scale'),
        ('/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ', ''),
    ]

    send_tasks(datasets)