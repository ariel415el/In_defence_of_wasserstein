import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from train_scripts.send_experiment import run_sbatch

if __name__ == '__main__':
    data_root = '/cs/labs/yweiss/ariel1/data'
    project_name = "3-4-2024-effect-of-batchsize-on-memorization"
    loss_function = 'WGANLoss'
    data_map = {
        "ffhq_10k": (f'{data_root}/FFHQ/FFHQ_HQ_cropped', f' --limit_data 10000', 10000),
        # "ffhq_70k": (f'{data_root}/FFHQ/FFHQ_HQ_cropped', f' --limit_data 70000', 70000),
        # "squares": (f'{data_root}/square_data/black_S-10_O-1_S-1', ' --gray_scale'),
        # "mnist": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 10000', 10000),
    }
    for data_name, (data_path, args, N) in data_map.items():
        for z_prior in [f'const={N}']:
            for reg_name, reg in [('GP', '--gp_weight 10')]:
                for dim in [64]:
                    for bs in [16, 64, 256, 1024]:
                        for gen_arch, disc_arch in [
                                ('DCGAN-nf=16-normalize=in', 'DCGAN-nf=16-normalize=in'),
                                ('DCGAN-nf=64-normalize=in', 'DCGAN-nf=64-normalize=in'),
                        ]:
                                train_name = f"{data_name}_{z_prior}_I-{dim}_Z-const={N}_G-{gen_arch}_D-{disc_arch}_BS-{bs}"
                                train_command = f"python3 train.py --data_path {data_path} {args}" \
                                                f" --project_name {project_name} --log_freq 5000" \
                                                f" --f_bs {bs} --r_bs {bs}" \
                                                f" --z_dim {dim} --im_size {dim}" \
                                                f" --load_data_to_memory --n_workers 0 --z_prior {z_prior} --n_iterations 1000000" \
                                                f" --gen_arch {gen_arch} --lrD 0.0002 --lrG 0.0001 --loss_function {loss_function} {reg}" \
                                                f" --train_name {train_name} " \
                                                f" --disc_arch {disc_arch} --wandb"
                                print(train_command)
                                run_sbatch(train_command, f'{train_name}',
                                           hours=72, killable=False, gpu_memory=21, cpu_memory=64, task_name="sbatch")

