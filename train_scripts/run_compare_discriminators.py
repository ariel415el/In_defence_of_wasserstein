import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from train_scripts.send_experiment import run_sbatch


if __name__ == '__main__':
    data_root = '/cs/labs/yweiss/ariel1/data/'
    data_map = {
        "ffhq": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 90 --limit_data 10000', 10000),
        # "ffhq_hq": (f'{data_root}/FFHQ/FFHQ_HQ_cropped', '--limit_data 10000', 10000),
        # "squares": (f'{data_root}/square_data/black_S-10_O-1_S-1', ' --gray_scale', 2704),
        # "mnist": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 10000', 10000),
    }
    dim = 64
    M = 64
    for gen_arch in ['FC', 'DCGAN']:
        project_name = f"paper_figures_10-6_G-{gen_arch}"
        commands = dict()
        for data_name, (data_path, args, N) in data_map.items():
            for z_prior in [f'const={M}']:
                base_command = f"python3 train.py --data_path {data_path} {args}" \
                                        f" --project_name {project_name} --log_freq 1000" \
                                        f" --z_dim {dim} --z_prior {z_prior} --im_size {dim}" \
                                        f" --load_data_to_memory --n_workers 0 --n_iterations 300000 " \
                                        f" --gen_arch {gen_arch}"

                for disc_arch in ['FC-nf=1024', 'CNN-GAP=True', 'SCNN-p=16-s=8']:
                    train_name = f"{data_name}-{z_prior}_WGAN-{disc_arch}"
                    train_command = base_command + f" --disc_arch {disc_arch} --lrD 0.0002 --lrG 0.0001" \
                                                   f" --loss_function WGANLoss --gp_weight 10 --train_name {train_name} --r_bs {64} --f_bs {64}"
                    commands[train_name] = train_command

            train_name = f"{data_name}-{z_prior}_DirectW1"
            train_command = base_command + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1 --r_bs {64} --f_bs {64}" \
                                           f" --lrG 0.001 --train_name {train_name}"
            commands[train_name] = train_command
            #
            # train_name = f"{data_name}-{z_prior}_DirectSWD"
            # train_command = base_command + f" --loss_function MiniBatchLoss-dist=swd --D_step_every -1 --r_bs {64} --f_bs {64}" \
            #                                f" --lrG 0.001 --train_name {train_name}"
            # commands[train_name] = train_command

            # train_name = f"{data_name}-{z_prior}_DirectPatchSWD"
            # train_command = base_command + f" --loss_function MiniBatchPatchLoss-dist=swd-p=16-s=8 --D_step_every -1 --r_bs {64} --f_bs {64}" \
            #                                f" --lrG 0.001 --train_name {train_name}"
            # commands[train_name] = train_command

            # train_name = f"{data_name}-{z_prior}_DirectPatchSWD-16-1"
            # train_command = base_command + f" --loss_function MiniBatchPatchLoss-dist=swd-p=16-s=1 --D_step_every -1 --r_bs {64} --f_bs {64}" \
            #                                f" --lrG 0.001 --train_name {train_name}"
            # commands[train_name] = train_command

            train_name = f"{data_name}-{z_prior}_DirectPatchSWD-16-8"
            train_command = base_command + f" --loss_function MiniBatchPatchLoss-dist=swd-p=16-s=8 --D_step_every -1 --r_bs {64} --f_bs {64}" \
                                           f" --lrG 0.001 --train_name {train_name}"
            commands[train_name] = train_command

            # train_name = f"{data_name}-{z_prior}_DirectPatchSWD-16-16"
            # train_command = base_command + f" --loss_function MiniBatchPatchLoss-dist=swd-p=16-s=16 --D_step_every -1 --r_bs {64} --f_bs {64}" \
            #                                f" --lrG 0.001 --train_name {train_name}"
            # commands[train_name] = train_command
            #
            # train_name = f"{data_name}-{z_prior}_DirectLocalPatchSWD-16-8"
            # train_command = base_command + f" --loss_function MiniBatchLocalPatchLoss-dist=swd-p=16-s=8 --D_step_every -1 --r_bs {N} --f_bs {M}" \
            #                                f" --lrG 0.001 --train_name {train_name}"
            # commands[train_name] = train_command

        for train_name, train_command in commands.items():
            print(train_command)
            run_sbatch(train_command, f'{train_name}', hours=2, killable=True, gpu_memory=8, cpu_memory=64, task_name="sbatch")