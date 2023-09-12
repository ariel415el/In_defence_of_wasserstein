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

    @staticmethod
    def plot_fig(names_and_commands, dataset_name, plot_type):
        titles_and_name_lists = [
                (name, [compose_experiment_name(parse_train_args(command)).replace("_test", "")], [])
              for name,command in names_and_commands]
        named_dirs = get_dir_paths(f'{out_root}/{project_name}', titles_and_name_lists)
        plot(named_dirs, f"{out_root}/Exp-{dataset_name}.png",plot_loss=plot_type, n=args.n, s=args.s)


class Figure_1(Figure):
    """Figure 1 in the papers compares the outputs of the OT-means algorithm to that of CTransformLoss"""
    @staticmethod
    def get_run_commands(project_name, dataset, additional_params):
        gen_arch = "Pixels --lrG 0.001"
        base = f" --data_path {dataset}  {additional_params}" \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
               f" --n_iterations 25000 --gen_arch {gen_arch}"

        names_and_commands = [
            ("Exp1-PixelCTGAN", base + f" --loss_function CtransformLoss "),
            ("Exp1-Pixel-W1", base + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1 --r_bs -1 "),
            ("Exp1-OTmeans", f"python3 scripts/EMD/ot_means.py {additional_params} --data_path {dataset}"
                              f" --project_name {project_name}",  f"Exp1-OTmeans-{os.path.basename(dataset)}")
        ]
        return names_and_commands


    @staticmethod
    def plot_fig(names_and_commands, dataset_name):
        names_and_commands = [(name, compose_experiment_name(parse_train_args(command))) for name, command in names_and_commands[:-1]]
        names_and_commands += ["OTmeans"]
        plot(f'{out_root}/{project_name}', f"Exp-{dataset_name}.png",
             [(name, [compose_experiment_name(parse_train_args(command)).replace("_test", "")], [])
              for name,command in names_and_commands],
             plot_loss=None, n=args.n
             )


class Figure_2(Figure):
    """Figure 2 in the Shows that Discrete GANS behave like CTransformLoss"""

    @staticmethod
    def get_run_commands(project_name, dataset, additional_params):
        gen_arch = "Pixels --lrG 0.001"
        base = f" --data_path {dataset}  {additional_params}" \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name} --z_prior const=64" \
               f"  --n_iterations 25000 " \
               f"--gen_arch {gen_arch} "

        names_and_commands = [
            ("Exp2-PixelWGAN-FC", base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.001 --G_step_every 5 --disc_arch FC-nf=1024"),
            ("Exp2-PixelCTGAN-FC", base + f" --loss_function CtransformLoss --lrD 0.001 --disc_arch FC-nf=1024"),

        ]
        return names_and_commands


class Figure_3(Figure):
    plot_type = "separate"
    """Compare DiscreteWGAN with CNN discriminator to direct patch ot minimization of patches of the same size"""
    @staticmethod
    def get_run_commands(project_name, dataset, additional_params):
        n_iterations = 250000
        gen_arch = "FC"
        base = f" --data_path {dataset}  {additional_params}" \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
               f" --n_iterations {n_iterations} " \
               f"--gen_arch {gen_arch} --lrG 0.0001 --z_prior const=64"

        names_and_commands = [
            ("Exp3-DiscreteWGAN-FC", base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.001 --G_step_every 5"
                                            f" --disc_arch FC-nf=1024"),
            ("Exp3-DiscreteWGAN-DC", base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.001 --G_step_every 5 "
                                            f"--disc_arch DCGAN-normalize=none"),
            ("Exp3-DiscreteWGAN-GAP-16", base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.001 --G_step_every 5"
                                                f" --disc_arch PatchGAN-normalize=in-k=3"),
            ("Exp3-Discrete-W1", base + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1"),
            # ("Exp3-Discrete-W1_p=16-s=8", base + f" --loss_function MiniBatchPatchLoss-dist=w1-p=16-s=8-n_samples=1024"
            #                                      f" --D_step_every -1")
        ]
        return names_and_commands


if __name__ == '__main__':
    data_root = '/cs/labs/yweiss/ariel1/data/'
    out_root = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs'
    data_map = {
        "ffhq": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 80 --limit_data 10000'),
        "squares": (f'{data_root}/square_data/black_S-10_O-1_S-1', ' --gray_scale'),
        "mnist": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 10000' )
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('figure_idx', type=int)
    parser.add_argument('--datasets', nargs='+', type=str, default=["ffhq", "squares", "mnist"])

    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--gpu_memory', default=8, type=int)
    parser.add_argument('--killable', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hours', default=4, type=int)

    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--n', default=8, type=int)
    parser.add_argument('--s', default=3, type=int)

    args = parser.parse_args()
    project_name = f"Figure_Exp{args.figure_idx}"

    sbatch_params = args.hours, args.killable, args.gpu_memory
    figure_command_generator = globals()[f"Figure_{args.figure_idx}"]

    for dataset_name in args.datasets:
        data_path, data_args = data_map[dataset_name]
        names_and_commands = figure_command_generator.get_run_commands(project_name, data_path, additional_params=data_args)
        if args.run:
            for name, command in names_and_commands:
                run_sbatch("python3 train.py " + command, f"{name}-{os.path.basename(data_path)}", *sbatch_params)

        elif args.plot:
            figure_command_generator.plot_fig(names_and_commands, dataset_name, figure_command_generator.plot_type)

        else:
            raise ValueError("Please supply at least one task (run, plot)")
