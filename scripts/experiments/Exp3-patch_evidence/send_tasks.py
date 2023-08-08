import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch


def send_tasks(project_name, datasets):
    gen_arch = "FC-nf=1024"
    for dataset, dataset_params in datasets:
        base = f"python3 train.py  --data_path {dataset}  {dataset_params}" \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
               f"  --n_iterations 100000 " \
               f"--gen_arch {gen_arch} --lrG 0.001"

        # WGANs
        run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch FC-nf=1024",
                   f"Exp3-DiscreteWGAN-FC", hours, killable)
        run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch PatchGAN-normalize=none-k=4",
                   f"Exp3-DiscreteWGAN-GAP-22", hours, killable)
        run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch PatchGAN-depth=4-normalize=none-k=4-nf=128",
                   f"Exp3-DiscreteWGAN-GAP-48", hours, killable)
        run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch DCGAN-normalize=none",
                   f"Exp3-DiscreteWGAN-DC", hours, killable)

        # Direct W1
        run_sbatch(base + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1",
                   f"Exp3-Discrete-W1", hours, killable)
        run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=w1-p=22-s=8 --D_step_every -1",
                   f"Exp3-Discrete-W1-22", hours, killable)
        run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=w1-p=48-s=16 --D_step_every -1",
                   f"Exp3-Discrete-W1-48", hours, killable)

        # Direct Sinkhorn
        # eps=100
        # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=sinkhorn-epsilon={eps} --D_step_every -1",
        #            f"Pixel-SH100", hours, killable)
        # run_sbatch(base + f" --lss_function MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=22-s=8 --D_step_every -1",
        #            f"Pixel-SH{eps}-22", hours, killable)
        # run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=48-s=16 --D_step_every -1",
        #            f"Pixel-SH{eps}-48", hours, killable)



if __name__ == '__main__':
    hours = 6
    killable=True
    datasets = [
        ('/cs/labs/yweiss/ariel1/data/square_data/black_S-10_O-1_S-1',' --gray_scale'),
        ('/cs/labs/yweiss/ariel1/data/MNIST/MNIST/jpgs/training',' --gray_scale'),
        ('/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ', ' --center_crop 100')
    ]

    send_tasks(project_name="Exp3-patch-evidence", datasets=datasets)