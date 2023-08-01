import subprocess
from time import sleep, strftime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch

def send_tasks(datasets):
    project_name = f"WGAN"
    hours = 8
    gen_arch = "FC-depth=3"
    for dataset, dataset_params in datasets:
        for z_prior in {"const=64", "normal"}:
            base = f"python3 train.py  --data_path {dataset}  {dataset_params}" \
                   f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
                   f"  --n_iterations 500000 --gen_arch {gen_arch} --lrG 0.0001 "

            # WGANs
            run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --z_prior {z_prior} "
                              f"--disc_arch PatchGAN-depth=3-normalize=none-k=4 --lrD 0.0001 --G_step_every 5 ",
                       f"Z-{z_prior}-WGAN", hours)

            run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --z_prior {z_prior} "
                              f"--disc_arch PatchGAN-depth=4-normalize=none-k=4 --lrD 0.0001 --G_step_every 5 ",
                       f"Z-{z_prior}-WGAN", hours)

            # Direct Sinkhorn
            eps=100
            run_sbatch(base + f" --loss_function MiniBatchLoss-dist=sinkhorn-epsilon={eps} --D_step_every -1",
                       f"Z-{z_prior}-SH100", hours)
            run_sbatch(base + f" --lss_function MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=22-s=8 --D_step_every -1",
                       f"Z-{z_prior}-SH{eps}-22", hours)
            run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=48-s=16 --D_step_every -1",
                       f"Z-{z_prior}-SH{eps}-48", hours)




if __name__ == '__main__':
    datasets = [
        ('/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ', ' --center_crop 90')
    ]

    send_tasks(datasets)