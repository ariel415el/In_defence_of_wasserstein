import os
import sys
import argparse

from plot_train_results import plot, get_dir_paths

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from utils.common import compose_experiment_name
from utils.train_utils import parse_train_args


if __name__ == '__main__':
    data_root = '/cs/labs/yweiss/ariel1/data/'
    out_root = '/cs/labs/yweiss/ariel1/repos/In_defence_of_wasserstein/outputs'
    data_map = {
        "ffhq": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 100 --limit_data 10000'),
        "ffhq1k": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 100 --limit_data 1000'),
        "obama": (f'{data_root}/few-shot-images/100-shot-obama/img', ''),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, default=["ffhq", "obama"])

    parser.add_argument('--gpu_memory', default=8, type=int)
    parser.add_argument('--killable', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hours', default=8, type=int)

    args = parser.parse_args()
    project_name = f"train_FastGANs_no_tanh"

    sbatch_params = args.hours, args.killable, args.gpu_memory

    for dataset_name in args.datasets:
        data_path, data_args = data_map[dataset_name]

        named_commands = dict()
        base = f"python3 train.py --data_path {data_path}  {data_args}" \
               f" --im_size 256 --loss_function SoftHingeLoss --r_bs 8 --f_bs 8 " \
               f" --augmentation 'color,translation,horizontal_flip' --avg_update_factor 0.001" \
               f" --log_freq 5000 --z_dim 256 --lrG 0.0002 --lrD 0.0002 --n_iterations 1000000 " \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name}"

        name = f"FastGAN-{dataset_name}"
        command = base + " --gen_arch FastGAN --disc_arch FastGAN --rec_lambda 0"
        run_sbatch(command, name, task_name=name, *sbatch_params)

        name = f"FastGAN-no-skip-{dataset_name}"
        command = base + " --gen_arch FastGAN-skip_connections=False --disc_arch FastGAN-skip_connections=False --rec_lambda 0"
        run_sbatch(command, name, task_name=name, *sbatch_params)

        name = f"FastGAN-rec-{dataset_name}"
        command = base + " --gen_arch FastGAN --disc_arch FastGAN --rec_lambda 1"
        run_sbatch(command, name, task_name=name, *sbatch_params)

        name = f"MyCNN-{dataset_name}"
        command = base + " --gen_arch MyCNN --disc_arch MyCNN --rec_lambda 0"
        run_sbatch(command, name, task_name=name, *sbatch_params)
