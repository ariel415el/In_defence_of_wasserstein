import subprocess
from time import sleep, strftime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch

def send_tasks(datasets):
    project_name = f"W1_patch_evidence"
    hours = 8
    gen_arch = "Pixels"
    for dataset, dataset_params in datasets:
        base = f"python3 train.py  --data_path {dataset}  {dataset_params}" \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
               f"  --n_iterations 100000 " \
               f"--gen_arch {gen_arch} --lrG 0.001"


        run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch FC-depth=3",
                   f"PixelWGAN-FC", hours)

        run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch PatchGAN-normalize=none-k=4",
                   f"PixelWGAN-GAP-22", hours)

        # run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch PatchGAN-depth=4-normalize=none-k=4",
        #            f"PixelWGAN-GAP-48", hours)
        #
        # run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch DCGAN-normalize=none",
        #            f"PixelWGAN-DC", hours)

        run_sbatch(base + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1",
                   f"Pixel-W1", hours)

        run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=w1-p=22-s=8 --D_step_every -1",
                   f"Pixel-W1-22", hours)

        # run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=w1-p=48-s=16 --D_step_every -1",
        #            f"Pixel-W1-48", hours)



if __name__ == '__main__':
    datasets = [
        ('/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1',' --gray_scale'),
        ('/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training',' --gray_scale'),
        ('/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ', ' --center_crop 90')
    ]

    send_tasks(datasets)