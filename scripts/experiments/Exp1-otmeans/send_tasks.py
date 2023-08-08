import subprocess
from time import sleep, strftime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch


def send_tasks(project_name, dataset, additional_params):
    hours = 1
    gen_arch = "Pixels --lrG 0.001"
    base = f"python3 train.py  --data_path {dataset}  {additional_params}" \
           f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
           f" --n_iterations 10000 --gen_arch {gen_arch}"

    run_sbatch(base + f" --loss_function CtransformLoss", f"Exp1-PixelCTGAN",
               hours, killable, gpu)
    # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1",  f"Exp1-Pixel-W1", hours, killable)

    run_sbatch(f"python3 scripts/EMD/ot_means.py {additional_params} --data_path {dataset} "
               f"--project_name {project_name}",  f"Exp1-OTmeans",
               hours, killable, gpu)

if __name__ == '__main__':
    hours = 2
    killable = True
    gpu = 4
    send_tasks(project_name="Exp1-Discrete_GM",
               dataset='/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1',
               additional_params=' --gray_scale')

    send_tasks(project_name="Exp1-Discrete_GM",
               dataset='/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training',
               additional_params=' --gray_scale')

    send_tasks(project_name="Exp1-Discrete_GM",
               dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ',
               additional_params=' --center_crop 100 --limit_data 10000')