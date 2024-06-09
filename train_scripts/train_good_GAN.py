import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from train_scripts.send_experiment import run_sbatch


if __name__ == '__main__':
    data_root = '/cs/labs/yweiss/ariel1/data'
    N = 70000
    M = 70000
    data_map = {
        "ffhq_hq": (f'{data_root}/FFHQ/FFHQ_HQ_cropped', f' --limit_data {N}'),
        # "squares": (f'{data_root}/square_data/black_S-10_O-1_S-1', ' --gray_scale'),
        # "mnist": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 10000'),
    }

    project_name = "trainFASTGAN-5-4-2024"
    loss_function = 'WGANLoss'
    for data_name, (data_path, args) in data_map.items():
        for z_prior in [f'const={M}']:
            for reg_name, reg in [('GP', '--gp_weight 10')]:
                for dim in [64, 128]:
                    for gen_arch, disc_arch in [
                            # ('DCGAN-nf=256-normalize=in', 'DCGAN-nf=256-normalize=in'),
                            # ('FastGAN-nf=64-skip_connections=True', 'FastGAN-nf=48-skip_connections=False'),
                            ('FastGAN', 'FastGAN'),
                            ('FastGAN-skip_connections=False', 'FastGAN-skip_connections=False')
                    ]:
                            train_name = f"{data_name}_{z_prior}_I-{dim}_Z-{dim}_Reg-{reg_name}_G-{gen_arch}_D-{disc_arch}"
                            train_command = f"python3 train.py --data_path {data_path} {args}" \
                                            f" --project_name {project_name} --log_freq 5000" \
                                            f" --f_bs 16 --r_bs 16" \
                                            f" --z_dim {dim} --im_size {dim}" \
                                            f" --load_data_to_memory --n_workers 0 --z_prior {z_prior} --n_iterations 5000000" \
                                            f" --gen_arch {gen_arch} --lrD 0.0002 --lrG 0.0001 --loss_function {loss_function} {reg}" \
                                            f" --train_name {train_name} " \
                                            f" --disc_arch {disc_arch} --wandb --full_batch_metrics"
                            print(train_command)
                            run_sbatch(train_command, f'{train_name}',
                                       hours=72, killable=False, gpu_memory=16, cpu_memory=64, task_name="sbatch")

