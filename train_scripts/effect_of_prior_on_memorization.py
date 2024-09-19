import subprocess
from time import sleep, strftime


def run_sbatch(train_command, train_name, hours, killable, gpu_memory, cpu_memory=64, task_name="sbatch"):
    sbatch_text = f"#!/bin/csh\n#SBATCH --time={hours}:0:0" \
                   f"\n#SBATCH --mem={cpu_memory}g" \
                   f"\n#SBATCH -c 5" \
                   f"\n#SBATCH --mail-type=END" \
                   f"\n#SBATCH --mail-user=ariel1" \
                   f"\n#SBATCH --output=/cs/labs/yweiss/ariel1/cluster_runs/{train_name}.out"
    if gpu_memory > 0:
        sbatch_text += f"\n#SBATCH --gres=gpu:1,vmem:{gpu_memory}g"
    if killable:
        sbatch_text += "\n#SBATCH --killable"
    sbatch_text += f"\nsource /cs/labs/yweiss/ariel1/venv/bin/activate.csh" \
                   f"\ncd /cs/labs/yweiss/ariel1/repos/In_defence_of_wasserstein" \
                   f"\n{train_command} "
    if train_name is not None:
        sbatch_text += f" --train_name {train_name}"  # _{strftime('%m-%d_T-%H:%M:%S')}"

    print("writing")
    f = open(f"{task_name}.csh", "w")
    f.write(sbatch_text)
    f.close()
    subprocess.Popen(["cat", f"{task_name}.csh"])
    subprocess.Popen(["sbatch", f"{task_name}.csh"])
    sleep(2)
    # subprocess.Popen(["rm", "-rf", f"{task_name}.csh"])


if __name__ == '__main__':
    data_root = '/cs/labs/yweiss/ariel1/data/'
    data_map = {
        "ffhq_10k": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 90 --limit_data 10000', 10000),
        # "ffhq_70k": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 90 --limit_data 70000', 70000),
        # "squares": (f'{data_root}/square_data/black_S-10_O-1_S-1', ' --gray_scale', 2704),
        # "mnist": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 10000', 10000),
    }
    dim = 64
    project_name = "5-april-effect_of_prior_on_memorization"
    commands = dict()
    for data_name, (data_path, args, N) in data_map.items():
        for gen_arch in ['FC-depth=4', 'FC-nf=1024', 'DC-normalize=in', 'DC-normalize=in-nf=128']:
            for M, z_prior in [(N, f'const={N}'),
                               # (int(1.5*N), f'const={int(1.5*N)}'),
                               # (2*N, f'const={2*N}'),
                               (N, 'normal')]:
                base_command = f"python3 train.py --data_path {data_path} {args}" \
                                        f" --project_name {project_name} --log_freq 1000" \
                                        f" --z_dim {dim} --z_prior {z_prior} --im_size {dim}" \
                                        f" --load_data_to_memory --n_workers 0 --n_iterations 300000 " \
                                        f" --gen_arch {gen_arch} --wandb --log_freq 100"

                train_name = f"{data_name}-{z_prior}-G-{gen_arch}_DirectW1"
                train_command = base_command + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1 --r_bs {N} --f_bs {M}" \
                                               f" --lrG 0.001"
                commands[train_name] = train_command


    for train_name, train_command in commands.items():
        print(train_command)
        run_sbatch(train_command, f'{train_name}', hours=8, killable=False, gpu_memory=8, cpu_memory=64, task_name="sbatch")