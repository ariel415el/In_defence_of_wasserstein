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
        "mnist": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 10000', 10000),
    }
    dim = 64
    M = 10000
    for gen_arch in ['FC', 'DCGAN']:
        project_name = f"train_SCNN_const=10k_2"
        commands = dict()
        for data_name, (data_path, args, N) in data_map.items():
            for z_prior in [f'const={M}']:
                base_command = f"python3 train.py --data_path {data_path} {args}" \
                                        f" --project_name {project_name} --log_freq 1000" \
                                        f" --z_dim {dim} --z_prior {z_prior} --im_size {dim}" \
                                        f" --load_data_to_memory --n_workers 0 --n_iterations 300000 " \
                                        f" --gen_arch {gen_arch}"

                # for disc_arch in ['DCGAN', 'CNN_LAP', 'SCNN-p=8-s=4', 'SCNN-p=16-s=8', 'CNN-GAP=True', 'CNN-GAP=True-depth=2']:
                for disc_arch in [ 'CNN_LAP-win=7',  'CNN_LAP-p=5-win=9']:
                    train_name = f"{data_name}-{z_prior}_WGAN_G-{gen_arch}_D-{disc_arch}"
                    train_command = base_command + f" --disc_arch {disc_arch} --lrD 0.0002 --lrG 0.0001" \
                                                   f" --loss_function WGANLoss --gp_weight 10 --train_name {train_name} --r_bs {64} --f_bs {64}"
                    commands[train_name] = train_command


        for train_name, train_command in commands.items():
            print(train_command)
            run_sbatch(train_command, f'{train_name}', hours=2, killable=True, gpu_memory=8, cpu_memory=64, task_name="sbatch")