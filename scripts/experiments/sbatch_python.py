import subprocess
from time import sleep, strftime
import os

def run_sbatch(train_command, stdout_name, hours, kilable=True):
    sbatch_text = (f"#!/bin/csh\n#SBATCH --time={hours}:0:0"
                   f"\n#SBATCH --gres=gpu:1,vmem:8g"
                   f"\n#SBATCH --mem=64g"
                   f"\n#SBATCH -c 5"
                   f"\n#SBATCH --mail-type=END" 
                  f"\n#SBATCH --mail-user=ariel1" 
                  f"\n#SBATCH --output=/cs/labs/yweiss/ariel1/cluster_runs/{stdout_name}.out")
    if kilable:
        sbatch_text += "\n#SBATCH --killable"
    sbatch_text += f"\nsource /cs/labs/yweiss/ariel1/venv/bin/activate.csh" \
                   f"\ncd /cs/labs/yweiss/ariel1/repos/DataEfficientGANs" \
                   f"\n{train_command} --tag {stdout_name}_{strftime('%m-%d_T-%H:%M:%S')}" \

    f = open("send_task.csh", "w")
    f.write(sbatch_text)
    f.close()

    subprocess.Popen(["cat", "send_task.csh"])
    subprocess.Popen(["sbatch", "send_task.csh"])
    sleep(2)

#
# def pixel_wgan(datasets):
#     project_name = f"PixelWGAN_all"
#     hours = 4
#     gen_arch = "Pixels"
#     for disc_arc in ["FC-depth=5", "DCGAN-normalize=none"]:
#         for dataset in datasets:
#             base = f"python3 train.py  --data_path {dataset}  --center_crop 90" \
#                    f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
#                    f" --loss_function WGANLoss --gp_weight 10 --n_iterations 100000 " \
#                    f"--gen_arch {gen_arch} --disc_arch {disc_arc}"
#             run_sbatch(base + f" --lrD 0.0001 --lrG 0.001 --G_step_every 5",
#                        f"DWGAN-{gen_arch}-{disc_arc}", hours)
#
# def pixel_wgan_2(datasets):
#     project_name = f"PixelWGAN_5"
#     hours = 2
#     gen_arch = "Pixels"
#     for dataset in datasets:
#         base = f"python3 train.py  --data_path {dataset}  --center_crop 90" \
#                f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
#                f"  --n_iterations 50000 " \
#                f"--gen_arch {gen_arch} --lrG 0.01"
#
#         # run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.001 --G_step_every 5 --disc_arch FC-nf=512-depth=5",
#         #            f"PixelWGAN", hours)
#         #
#         run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.001 --G_step_every 5 --disc_arch FC-depth=3",
#                    f"PixelWGAN", hours)
#
#         # run_sbatch(base + f" --loss_function CtransformLoss  --lrD 0.001 --G_step_every 5 --disc_arch FC-nf=512-depth=5",
#         #            f"Pixel-CT-WGAN", hours)
#
#         # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=w1  --D_step_every -1",
#         #            f"Pixel-SH-WGAN", hours)
#         #
#         # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=sinkhorn-epsilon=1  --D_step_every -1",
#         #            f"Pixel-SH-WGAN", hours)
#         #
#         # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=sinkhorn-epsilon=10  --D_step_every -1",
#         #            f"Pixel-SH-WGAN", hours)
#         #
#         # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=sinkhorn-epsilon=50  --D_step_every -1",
#         #            f"Pixel-SH-WGAN", hours)
#
#         #
#         # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=sinkhorn-epsilon=100  --D_step_every -1",
#         #            f"Pixel-SH-WGAN", hours)
#
#
# def discrete_wgan(datasets):
#     project_name = f"discreteWGAN_all"
#     hours = 4
#     for gen_arch in ["FC-depth=5", "DCGAN-normalize=none"]:
#         for disc_arc in ["FC-depth=5", "DCGAN-normalize=none"]:
#             for z_prior in ["const=64"]:#, "normal"]:
#                 for dataset in datasets:
#                     base = f"python3 train.py  --data_path {dataset}  --center_crop 90" \
#                            f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
#                            f" --loss_function WGANLoss --gp_weight 10 --n_iterations 100000 " \
#                            f"--gen_arch {gen_arch} --disc_arch {disc_arc} --z_prior {z_prior}"
#                     run_sbatch(base + f" --lrD 0.0001 --lrG 0.0001 --G_step_every 5",
#                                f"DWGAN-{gen_arch}-{disc_arc}", hours)
#
#
# def WGAN(datasets):
#     project_name = "WGAN"
#     disc_arch  = "DCGAN-normalize=none"
#     z_prior = "normal"
#     hours = 24
#     for dataset in datasets:
#         for gen_arch in ["FC-depth=3", "DCGAN-normalize=none"]:
#             base = f"python3 train.py  --data_path {dataset}  --center_crop 90" \
#                    f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
#                    f" --loss_function WGANLoss --gp_weight 10 --n_iterations 100000 " \
#                    f" --gen_arch {gen_arch} --disc_arch {disc_arch} --z_prior {z_prior}"
#             run_sbatch(base + f" --lrD 0.0001 --lrG 0.0001 --G_step_every 5",
#                        f"DWGAN-{gen_arch}-{disc_arch}", hours)
#
# def patch_losses(datasets):
#     project_name = "BatchLosses"
#     z_prior = "normal"
#     hours = 4
#     for dataset in datasets:
#         for gen_arch in ["FC-depth=3", "DCGAN-normalize=none"]:
#             base = f"python3 train.py  --data_path {dataset}  --center_crop 90" \
#                    f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
#                    f"  --D_step_every -1 --n_iterations 100000 " \
#                    f" --gen_arch {gen_arch}  --z_prior {z_prior}"
#             run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=w1-p=48-s=24 --lrG 0.001",
#                        f"W1-{gen_arch}", hours)
#             # run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=swd-p=16-s=8 --lrG 1",
#             #            f"swd-{gen_arch}", hours)
#             # run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=sinkhorn-epsilon=100-p=16-s=8  --lrG 0.001",
#             #            f"sinkhorn-{gen_arch}", hours)
#
#
# if __name__ == '__main__':
#     datasets = [
#         '/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1',
#         '/cs/labs/yweiss/ariel1/data/edges_data',
#         # '/cs/labs/yweiss/ariel1/data/MNIST/floating_MNIST/train-128-0',
#         '/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training_1000',
#         '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_1000',
#         '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_flat',
#     ]
#
#     # discrete_wgan(['/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'])
#     # pixel_wgan(['/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'])
#     WGAN(['/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'])
#     patch_losses(['/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'])
#     # pixel_wgan_2(['/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_1000', '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'])
#     # WGAN(['/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ'])