import subprocess
from time import sleep, strftime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch


def send_tasks(project_name, dataset, additional_params):
    hours = 8
    for gen_arch in ["FC-depth=3", "DCGAN-normalize=none"]:
        for z_prior in {"const=64", "normal"}:
            base = f"python3 train.py  --data_path {dataset}  {additional_params}" \
                   f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
                   f"  --n_iterations 500000 --gen_arch {gen_arch} --lrG 0.0001 "

            # WGANs
            run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --z_prior {z_prior} "
                              f"--disc_arch PatchGAN-depth=3-normalize=none-k=4 --lrD 0.0001 --G_step_every 5 ",
                       f"{gen_arch}-Z-{z_prior}-WGAN-GAP-22", hours)

            run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --z_prior {z_prior} "
                              f"--disc_arch PatchGAN-depth=4-normalize=none-k=4 --lrD 0.0001 --G_step_every 5 ",
                       f"{gen_arch}-Z-{z_prior}-WGAN-GAP-22", hours)

            run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --z_prior {z_prior} "
                              f"--disc_arch DCGAN-normalize=none --lrD 0.0001 --G_step_every 5 ",
                       f"{gen_arch}-Z-{z_prior}-WGAN-DC", hours)

            # Direct Sinkhorn
            eps=100
            run_sbatch(base + f" --loss_function MiniBatchLoss-dist=sinkhorn-epsilon={eps} "
                              f"--D_step_every -1 --z_prior {z_prior}",
                       f"{gen_arch}-Z-{z_prior}-SH100", hours)
            run_sbatch(base + f" --lss_function MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=22-s=8 "
                              f"--D_step_every -1 --z_prior {z_prior}",
                       f"{gen_arch}-Z-{z_prior}-SH{eps}-22", hours)
            run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=48-s=16 "
                              f"--D_step_every -1 --z_prior {z_prior}",
                       f"{gen_arch}-Z-{z_prior}-SH{eps}-48", hours)


if __name__ == '__main__':
    # send_tasks(project_name="WGAN-1K",
    #            dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ',
    #            additional_params=' --center_crop 90 --limit_data 1000')

    send_tasks(project_name="WGAN-10K",
               dataset='/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ',
               additional_params=' --center_crop 90 --limit_data 10000')