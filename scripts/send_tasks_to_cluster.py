import subprocess
from time import sleep, strftime
import os
# ssh phoenix-gw


def run_sbatch(train_command, stdout_name, hours):
    sbatch_text = f"""#!/bin/csh
#SBATCH --time={hours}:0:0
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --mem=64g
#SBATCH -c 5
#SBATCH --mail-type=END
#SBATCH --mail-user=ariel1
#SBATCH --output=/cs/labs/yweiss/ariel1/cluster_runs/{stdout_name}.out


# module load torch
source /cs/labs/yweiss/ariel1/venv/bin/activate.csh
cd /cs/labs/yweiss/ariel1/repos/DataEfficientGANs
{train_command} --tag {stdout_name}_{strftime('%m-%d_T-%H:%M:%S')}
"""
    f = open("send_task.csh", "w")
    f.write(sbatch_text)
    f.close()
    
    subprocess.Popen(["cat", "send_task.csh"])
    subprocess.Popen(["sbatch", "send_task.csh"])
    sleep(2)


def train_bagnet_discriminators():
    im_size, z_dim, batch_size, hours = (128, 128, 32, 24)

    base = f"python3 train.py --data_path /cs/labs/yweiss/ariel1/data/FFHQ_128" \
              f" --im_size {im_size} --z_dim {z_dim} --batch_size {batch_size}  --loss_function SoftHingeLoss " \
              f" --wandb_project BagNetTraining"

    for p in [33, 17, 9]:
        run_sbatch(base + f" --gen_arch FastGAN --disc_arch BagNet-rf={p} ", f"FastGan-BagNet-{p}_stdout", hours)



def run_WGAN(dataset, gen):
    hours = 4
    name = os.path.basename(dataset)
    base = f"python3 train.py --data_path {dataset}  --loss_function WGANLoss --batch_size 64 --gp_weight 10 --project_name WGAN_yweiss_IN --wandb --n_iterations 100000"
    

    run_sbatch(base + f" --gen_arch {gen}  --disc_arch FC-normalize='in'", f"{gen}-FC_{name}", hours)
    run_sbatch(base + f" --gen_arch {gen}  --disc_arch DCGAN-normalize='in'", f"{gen}-DC_{name}", hours)
    run_sbatch(base + f" --gen_arch {gen}  --disc_arch PatchGAN-normalize='in'", f"{gen}-PatchDisc_{name}", hours)
    # run_sbatch(base + f" --gen_arch {gen}  --disc_arch SCNN-ksize=9", f"{gen}-SCNN22_{name}", hours)
    # run_sbatch(base + f" --gen_arch {gen}  --disc_arch SCNN-ksize=22", f"{gen}-SCNN9_{name}", hours)

if __name__ == '__main__':
    for dataset in [
        '/cs/labs/yweiss/ariel1/data/square_data/7x7',
        '/cs/labs/yweiss/ariel1/data/MNIST/floating_MNIST/train-128-0',
        '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_128'
    ]:
        for gen in ["pixels"]:
            run_WGAN(dataset, gen)
