import os
import sys
import argparse

from plot_train_results import plot, get_dir_paths

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from utils.common import compose_experiment_name
from utils.train_utils import parse_train_args


class Figure:
    plot_type=None
    @staticmethod
    def get_run_commands(project_name, dataset, additional_params):
        raise NotImplemented


class Figure_2a(Figure):
    """Figure 1 Show that DiscreteWGAN with FC discriminator approximately minimizes W1 by comparing it to OTmeans
    Use M<N centroids
    """
    plot_type='common'
    @staticmethod
    def get_run_commands(project_name, dataset, additional_params):
        nc = 64
        named_commands = dict()
        for gen_name, gen_arch in [
            # ("Pixels", f"Pixels-n={nc} --lrG 0.001 "),
            ("FC", "FC --lrG 0.0001 ")
        ]:
            base = f"python3  train.py --data_path {dataset}  {additional_params} " \
                   f" --load_data_to_memory --n_workers 0 --project_name {project_name} " \
                   f" --n_iterations 25000 --gen_arch {gen_arch} " \
                   f" --z_prior const={nc} --r_bs {nc} --f_bs {nc}"

            wgan_params = f" --loss_function WGANLoss --gp_weight 10 --lrD 0.001 --G_step_every 5"
            named_commands[f"DiscreteWGAN-{gen_name}-FC"] = base + wgan_params +" --disc_arch FC-nf=1024"

        named_commands["OTmeans"] = f"python3 scripts/experiments/OT/ot_means.py  --k {nc}" \
                                    f" --data_path {dataset} --project_name {project_name} {additional_params}"
        return named_commands


class Figure_2b(Figure):
    """Figure 1 Show that DiscreteWGAN with FC discriminator approximately minimizes W1 by comparing it to OTmeans
    Use M=N centroids
    """
    plot_type='common'
    @staticmethod
    def get_run_commands(project_name, dataset, additional_params):
        nc = 1000
        bs = 64
        named_commands = dict()
        for gen_name, gen_arch in [
            # ("Pixels", f"Pixels-n={nc} --lrG 0.001 "),
            ("FC", "FC-nf=1024 --lrG 0.0001 ")
        ]:
            base = f"python3  train.py --data_path {dataset}  {additional_params} " \
                   f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
                   f" --n_iterations 100000 --gen_arch {gen_arch}" \
                   f" --z_prior const={nc} --r_bs {bs} --f_bs {bs}"

            wgan_params = f" --loss_function WGANLoss --gp_weight 10 --lrD 0.001"
            named_commands[f"DiscreteWGAN-{gen_name}-FC"] = base + wgan_params +" --disc_arch FC-nf=1024"

        named_commands["OTmeans"] = f"python3 scripts/experiments/OT/ot_means.py  --k {nc}" \
                                    f" --data_path {dataset} --project_name {project_name} {additional_params}" \
                                    f" --min_method sgd"
        return named_commands


class Figure_3(Figure):
    """Show that DiscreteWGAN with CNN discriminator minimizes patch-W1 by comparing it to direct patch-SWD miniimzation
    Add FC discriminator DiscreteWGAN as reference
    """
    plot_type='common'
    @staticmethod
    def get_run_commands(project_name, dataset, additional_params):
        nc = 64
        named_commands = dict()
        base = f"python3 train.py --data_path {dataset}  {additional_params}" \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
               f" --n_iterations 50000 --log_freq 2500"\
               f" --r_bs {nc} --f_bs {nc} --z_prior const={nc}"

        wgan_params = ' --loss_function WGANLoss --gp_weight 10 --gen_arch FC --lrG 0.0001  --G_step_every 5 --lrD 0.001'
        named_commands[f"DiscreteWGAN-FC"] = base + wgan_params + f" --disc_arch FC-nf=1024"
        named_commands[f"DiscreteWGAN-CNN-GAP"] = base + wgan_params + f" --disc_arch PatchGAN-depth=3-k=3-normalize=none"


        direct_swd_params = f" --loss_function MiniBatchPatchLoss-dist=swd-p=16-s=1 --D_step_every -1 --gen_arch Pixels-n={nc} --lrG 0.001"
        named_commands[f"DirectPatchSWD"] = base + direct_swd_params

        # direct_loss = """MiniBatchMSPatchLoss-dists='["swd","swd"]'-ps='[8,16]'-ss='[4,8]'-intervals='[10000]'"""
        # direct_swd_params = f" --loss_function {direct_loss} --D_step_every -1 --gen_arch Pixels-n={nc} --lrG 0.01"
        # named_commands[f"DirectPatchSWD-8+16"] = base + direct_swd_params

        return named_commands


class Figure_5(Figure):
    """
    Show that Discrete WGAN with DC discriminator mixes image-level and patch level losses by comparing it to Multi-scale
    patch-SWD loss
    """
    plot_type='separate'
    @staticmethod
    def get_run_commands(project_name, dataset, additional_params):
        nc = 64
        named_commands = dict()

        base = f"python3 train.py --data_path {dataset}  {additional_params}" \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
               f" --n_iterations 50000 --log_freq 5000 --z_prior const={nc}"

        direct_loss = """MiniBatchMSPatchLoss-dists='["w1","swd", "swd"]'-ps='[64,16, 8]'-ss='[1,1,1]'-intervals='[5000, 10000]'"""
        direct_params = f' --loss_function {direct_loss} --D_step_every -1 --gen_arch Pixels-n={nc} --lrG 0.01 '
        # named_commands[f"DirectPatchSWD"] = base + direct_params

        wgan_params = f' --loss_function WGANLoss --gp_weight 10 --G_step_every 5' \
                      f' --lrD 0.001  --gen_arch FC --lrG 0.0001 '

        # named_commands[f"DiscreteWGAN-FC"] = base + wgan_params + f" --disc_arch FC-nf=1024"

        disc_arch = f"PatchGAN-depth=3-k=3-nf=256-normalize=none-GAP=True"
        named_commands[f"DiscreteWGAN-CNN+GAP"] = base + wgan_params + f" --disc_arch {disc_arch}"

        disc_arch = f"PatchGAN-depth=3-k=3-nf=256-normalize=none-GAP=False"
        named_commands[f"DiscreteWGAN-CNN+FC"] = base + wgan_params + f" --disc_arch {disc_arch}"


        return named_commands


if __name__ == '__main__':
    data_root = '/cs/labs/yweiss/ariel1/data/'
    out_root = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs'
    data_map = {
        "ffhq": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 80 --limit_data 10000'),
        "ffhq1k": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 80 --limit_data 1000'),
        "squares": (f'{data_root}/square_data/black_S-10_O-1_S-1', ' --gray_scale'),
        "squares1k": (f'{data_root}/square_data/black_S-10_O-1_S-1', ' --gray_scale --limit_data 1000'),
        "mnist": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 10000'),
        "mnist1k": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 10000 --limit_data 1000'),
        "ffhq_centroids": (f'{data_root}//FFHQ/FFHQ_centroids/shifted_crops/shifted_crops', ''),
        "mnist_centroids": (f'{data_root}//MNIST/MNIST_centroids/floating_images/floating_images', ' --gray_scale'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('figure_idx', type=str)
    parser.add_argument('--datasets', nargs='+', type=str, default=["ffhq", "squares", "mnist"])

    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--gpu_memory', default=8, type=int)
    parser.add_argument('--killable', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hours', default=4, type=int)

    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--n', default=4, type=int)
    parser.add_argument('--s', default=4, type=int)
    parser.add_argument('--tag', default='', type=str)

    args = parser.parse_args()
    project_name = f"Figure_Exp{args.figure_idx}{args.tag}"

    sbatch_params = args.hours, args.killable, args.gpu_memory
    task_name = f"Figure_{args.figure_idx}"
    figure_command_generator = globals()[task_name]

    for dataset_name in args.datasets:
        data_path, data_args = data_map[dataset_name]
        named_commands = figure_command_generator.get_run_commands(project_name, data_path, additional_params=data_args)
        if args.run:
            for name, command in named_commands.items():
                run_sbatch(command, f"{dataset_name}-{name}", task_name=task_name, *sbatch_params)

        elif args.plot:
            for name in named_commands:
                named_commands[name] = os.path.join(out_root, project_name, f"{dataset_name}-{name}")
            plot(named_commands, os.path.join(out_root, project_name, f"Exp-{dataset_name}.png"),
                                plot_type=figure_command_generator.plot_type, n=args.n, s=args.s)

        else:
            raise ValueError("Please supply at least one task (run, plot)")
