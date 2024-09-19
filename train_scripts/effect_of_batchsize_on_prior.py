import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from train_scripts.send_experiment import run_sbatch


if __name__ == '__main__':
    data_root = '/cs/labs/yweiss/ariel1/data/'
    data_map = {
        "ffhq": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 90 --limit_data 1000'),
        "mnist": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 1000'),
        "squares": (f'{data_root}/square_data/black_S-10_O-1_S-1', ' --gray_scale --limit_data 1000'),
    }
    N = 1000
    project_name = "effect_of_batchsize_on_prior_5"
    commands = dict()
    for data_name, (data_path, args) in data_map.items():
        for gen_arch in ['FC']:
            priors_and_fbs = []
            for m in [16, 1000]:
                priors_and_fbs += [
                ('normal', m),
                (f'const={m}', m),
                # ('const=1000', m)
            ]
            for z_prior, f_bs in priors_and_fbs:
                base_command = f"python3 train.py --data_path {data_path} {args}" \
                               f" --load_data_to_memory --n_workers 0 --n_iterations 100000 " \
                               f" --z_prior {z_prior} --gen_arch {gen_arch}" \
                               f" --project_name {project_name}"

                base_train_name = f"{data_name}_Z-{z_prior}_G-{gen_arch}"
                for disc_arch in ['FC-nf=1024']:#, 'CNN-GAP=True']:
                    train_name = base_train_name + f"_WGAN-{disc_arch}_fbs-{f_bs}"
                    train_command = base_command + f" --disc_arch {disc_arch} --lrD 0.0002 --lrG 0.0001" \
                                                   f" --loss_function WGANLoss --gp_weight 10 --train_name {train_name} --r_bs {64} --f_bs {64}"
                    commands[train_name] = train_command

                train_name = base_train_name + f"_DirectW1_fbs-{f_bs}"
                train_command = base_command + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1" \
                                               f" --r_bs {N} --f_bs {f_bs}" \
                                               f" --lrG 0.0001"
                commands[train_name] = train_command


    for train_name, train_command in commands.items():
        print(train_command)
        run_sbatch(train_command, f'{train_name}', hours=24, killable=True, gpu_memory=8, cpu_memory=64, task_name="sbatch")