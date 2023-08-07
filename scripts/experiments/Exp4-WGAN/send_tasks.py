import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch


def send_tasks(project_name, dataset, additional_params):
    for gen_arch in [
        "FC",
        # "FC-depth=8-nf=1024",
        # "DCGAN-normalize=in-nf=128",
        # "ResNet"
    ]:
        for disc_arch in [
            # "FC-depth=8-nf=1024",
            "DCGAN-normalize=none",
            # "ResNet"
        ]:
            for z_prior in [
                "const=64",
                "const=512"
            ]:
                base = f"python3 train.py  --data_path {dataset}  {additional_params}" \
                       f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
                       f"  --n_iterations 500000 --gen_arch {gen_arch} --lrG 0.00001 " \
                       f" --loss_function WGANLoss --gp_weight 10 --z_prior {z_prior} "

                run_sbatch(base + f" --disc_arch {disc_arch}  --batch_size 64 --lrD 0.0001 --G_step_every 5 ",
                               f"{gen_arch}-Z-{z_prior}-{disc_arch}", hours, killable, gpu_memory)


if __name__ == '__main__':
    killable = True
    hours = 8
    gpu_memory = 8

    send_tasks(project_name="WGAN-10K_2",
               dataset='/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1',
               additional_params=' --gray_scale')

    send_tasks(project_name="WGAN-1K",
               dataset='/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training',
               additional_params=' --gray_scale')

    send_tasks(project_name="WGAN-10K_2",
               dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ',
               additional_params=' --center_crop 100 --limit_data 10000')