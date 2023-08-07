import subprocess
from time import sleep, strftime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch


def send_tasks(project_name, dataset, additional_params):
    hours = 8
    for gen_arch in [
        "Linear",
        # "MultiLinear-n_linears=1000"
    ]:
        for disc_arch in [
            # "FC-depth=8-nf=1024",
            "DCGAN-normalize=in-nf=128",
            "ResNet"
        ]:
            for z_prior in [
                "normal",
                # "const=64"
            ]:
                base = f"python3 train.py  --data_path {dataset}  {additional_params}" \
                       f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
                       f"  --n_iterations 500000 --gen_arch {gen_arch}  " \
                       f" --loss_function WGANLoss --gp_weight 10 --z_prior {z_prior} --z_dim 10 --n_generators 1000" \
                       f" --lrG 0.001"

                run_sbatch(base + f" --disc_arch {disc_arch}  --batch_size 64 --lrD 0.0001 ",
                               f"{gen_arch}-Z-{z_prior}-{disc_arch}", hours, killable, gpu_memory)



if __name__ == '__main__':
    killable = False
    gpu_memory = 16
    # send_tasks(project_name="WGAN-1K",
    #            dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ',
    #            additional_params=' --center_crop 90 --limit_data 1000')

    send_tasks(project_name="trials",
               dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ',
               additional_params=' --center_crop 100 --limit_data 10000')