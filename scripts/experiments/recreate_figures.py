import os
import sys
import argparse
import subprocess
from time import sleep, strftime
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sbatch_python import run_sbatch
from plot_train_results import find_dir, find_last_image, plot


class Figure1:
    """Figure 1 in the papers compares the outputs of the OT-means algorithm to that of CTransformLoss"""
    @staticmethod
    def send_tasks(project_name, dataset, additional_params):
        gen_arch = "Pixels --lrG 0.001"
        base = f"python3 train.py  --data_path {dataset}  {additional_params}" \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
               f" --n_iterations 25000 --gen_arch {gen_arch}"

        run_sbatch(base + f" --loss_function CtransformLoss", f"Exp1-PixelCTGAN",
                   args.hours, args.killable, args.gpu_memory)

        # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1",  f"Exp1-Pixel-W1",
        #            args.hours, args.killable, args.gpu_memory)

        run_sbatch(f"python3 scripts/EMD/ot_means.py {additional_params} --data_path {dataset} "
                   f"--project_name {project_name}",  f"Exp1-OTmeans",
                   args.hours, args.killable, args.gpu_memory)

    @staticmethod
    def plot_fig(project_name, dataset):
        plot(f'{out_root}/{project_name}',
             {
                 f"Exp1-{dataset}.png": [
                     (f"OTmeans", [f"{dataset}_I", "K-64"], []),
                     # (f"BatchW1", [f"{dataset}_I", "L-MiniBatch"], []),
                     (f"CTGAN", [f"{dataset}_I", "L-CtransformLoss", "B-64"], []),
                 ]
            },
             seperate_plots=False, n=args.n
             )


class Figure2:
    """Figure 2 in the Shows that Discrete GANS behave like CTransformLoss"""
    @staticmethod
    def send_tasks(project_name, dataset, additional_params):
        for gen_arch in ["Pixels --lrG 0.001", "FC --lrG 0.0001"]:
            base = f"python3 train.py  --data_path {dataset}  {additional_params}" \
                   f" --load_data_to_memory --n_workers 0 --project_name {project_name} --z_prior const=64" \
                   f"  --n_iterations 25000 " \
                   f"--gen_arch {gen_arch} "

            run_sbatch(base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.001 --G_step_every 5 --disc_arch FC-nf=1024",
                       f"Exp2-PixelWGAN-FC-1024", *sbatch_params)

            run_sbatch(base + f" --loss_function CtransformLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch FC-nf=1024",
                       f"Exp2-PixelCTGAN-FC-1024", *sbatch_params)

            # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1",
            #            f"Exp2-Pixel-W1", hours, killable)

            # run_sbatch(base + f" --loss_function MiniBatchLoss-dist=sinkhorn-epsilon=100 --D_step_every -1",
            #            f"Exp2-Pixel-sinkhorn100", hours, killable)

    @staticmethod
    def plot_fig(project_name, dataset):
        plot(f'{out_root}/{project_name}',
             {
                 f"{dataset}-{gen_arch}.png": [
                     (f"DiscreteWGAN", [dataset, "L-WGANLoss", f"G-{gen_arch}", "D-FC-nf=1024"], []),
                     (f"DiscreteCTGAN", [dataset, "L-CtransformLoss", f"G-{gen_arch}"], []),
                     # (f"W1", [dataset, "L-MiniBatchLoss-dist=w1", f"G-{gen_arch}"], []),
                     #(f"Sinkhorn-100-FC", ["L-MiniBatchLoss-dist=sinkhorn-epsilon=100", f"G-{gen_arch}"], []),
                 ]
                 for gen_arch in ["Pixels", "FC"]
             },
             seperate_plots=False, n=args.n
        )


class Figure3:
    @staticmethod
    def send_tasks(project_name, dataset, additional_params):
        gen_arch = "FC"
        # gen_arch = "FC-nf=1024"
        base = f"python3 train.py  --data_path {dataset}  {additional_params}" \
               f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
               f" --n_iterations 100000 " \
               f"--gen_arch {gen_arch} --lrG 0.001"

        # WGANs
        run_sbatch(
            base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch FC-nf=1024",
            f"Exp3-DiscreteWGAN-FC", *sbatch_params)
        run_sbatch(
            base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch PatchGAN-normalize=none-k=4",
            f"Exp3-DiscreteWGAN-GAP-22", *sbatch_params)
        run_sbatch(
            base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch PatchGAN-depth=4-normalize=none-k=4-nf=128",
            f"Exp3-DiscreteWGAN-GAP-48", *sbatch_params)
        run_sbatch(
            base + f" --loss_function WGANLoss --gp_weight 10 --lrD 0.0001 --G_step_every 5 --disc_arch DCGAN-normalize=none",
            f"Exp3-DiscreteWGAN-DC", *sbatch_params)

        # Direct W1
        run_sbatch(base + f" --loss_function MiniBatchLoss-dist=w1 --D_step_every -1",
                   f"Exp3-Discrete-W1", *sbatch_params)
        run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=w1-p=22-s=8 --D_step_every -1",
                   f"Exp3-Discrete-W1-22", *sbatch_params)
        run_sbatch(base + f" --loss_function MiniBatchPatchLoss-dist=w1-p=48-s=16 --D_step_every -1",
                   f"Exp3-Discrete-W1-48", *sbatch_params)
    
    @staticmethod
    def plot_fig(project_name, dataset):
        plot(f'{out_root}/{project_name}',
             {
                 f"Evidence-.png": [
                     (f"{dataset}-DiscreteWGAN-FC", [dataset, "L-WGANLoss", "D-FC"], []),
                     (f"{dataset}-DiscreteWGAN-GAP-22", [dataset, "L-WGANLoss", "D-PatchGAN-normalize=none-k=4"], ["PatchGAN-depth=4-normalize=none-k=4"]),
                     (f"{dataset}-DiscreteWGAN-GAP-48", [dataset, "L-WGANLoss", "PatchGAN-depth=4-normalize=none-k=4"], []),
                     (f"{dataset}-DiscreteWGAN-DCGAN", [dataset, "L-WGANLoss", "D-DCGAN"], []),

                     (f"{dataset}-Discrete-W1", [dataset, "MiniBatchLoss-dist=w1"], []),
                     (f"{dataset}-Discrete-W1-p=22-s=8", [dataset, "MiniBatchPatchLoss-dist=w1-p=22-s=8"], []),
                     (f"{dataset}-Discrete-W1-p=48-s=16", [dataset, "MiniBatchPatchLoss-dist=w1-p=48-s=16"], []),

                     # (f"Discrete-sinkhorn-epsilon={eps}", [f"MiniBatchLoss-dist=sinkhorn-epsilon={eps}"], []),
                     # (f"Discrete-sinkhorn-epsilon={eps}-p=22-s=8", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=22-s=8"], []),
                     # (f"Discrete-sinkhorn-epsilon={eps}-p=48-s=16", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=48-s=16"], []),
                 ]
             },
             seperate_plots=True
        )


class Figure4:
    @staticmethod
    def send_tasks(project_name, dataset, additional_params):
        for gen_arch in [
            # "FC",
            "FC-nf=1024",
            # "DCGAN-normalize=in-nf=128",
            # "ResNet"
        ]:
            for disc_arch in [
                "DCGANN-normalize=in-nf=128",
                # "ResNet"
            ]:
                for z_prior in [
                    "const=64",
                    "const=512"
                ]:
                    base = f"python3 train.py  --data_path {dataset}  {additional_params}" \
                           f" --load_data_to_memory --n_workers 0 --project_name {project_name}" \
                           f"  --n_iterations 500000 --gen_arch {gen_arch} --lrG 0.0001 " \
                           f" --loss_function WGANLoss --gp_weight 10 --z_prior {z_prior} "

                    run_sbatch(base + f" --disc_arch {disc_arch} --lrD 0.0001 --G_step_every 5 ",
                               f"Exp4-{gen_arch}-Z-{z_prior}-{disc_arch}", hours, killable, gpu_memory)

    @staticmethod
    def plot_fig(project_name, dataset):
        plot(f'{out_root}/{project_name}',
             {
                 f"WGAN.png": [
                     # (f"f"{dataset}-Z-{z_prior}-WGAN-GAP-22", ["L-WGANLoss", "PatchGAN-depth=3-normalize=none-k=4", f"64x{z_prior}", f"G-{gen_arch}"], []),
                     # (f"Z-{z_prior}-WGAN-GAP-48", ["L-WGANLoss", "PatchGAN-depth=4-normalize=none-k=4", f"64x{z_prior}", f"G-{gen_arch}"], []),
                     (f"Z-{z_prior}", ["L-WGANLoss", f"64x{z_prior}"], [])

                     # (f"Z-{z_prior}-sinkhorn-epsilon={eps}", [f"MiniBatchLoss-dist=sinkhorn-epsilon={eps}", f"64x{z_prior}", f"G-{gen_arch}"], []),
                     # (f"Z-{z_prior}-sinkhorn-epsilon={eps}-p=22-s=8", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=22-s=8", f"64x{z_prior}", f"G-{gen_arch}"], []),
                     # (f"Z-{z_prior}-sinkhorn-epsilon={eps}-p=48-s=16", [f"MiniBatchPatchLoss-dist=sinkhorn-epsilon={eps}-p=48-s=16", f"64x{z_prior}", f"G-{gen_arch}"], []),
                     for z_prior in ["const=64", "const=512"]]
             },
             seperate_plots=False
             )


if __name__ == '__main__':
    data_root = '/cs/labs/yweiss/ariel1/data/'
    out_root = '/cs/labs/yweiss/ariel1/repos/DataEfficientGANs/outputs'
    data_map = {
        "ffhq": (f'{data_root}/FFHQ/FFHQ', ' --center_crop 100 --limit_data 10000'),
        "squares": (f'{data_root}/square_data/black_S-10_O-1_S-1', ' --gray_scale'),
        "mnist": (f'{data_root}/MNIST/MNIST/jpgs/training', ' --gray_scale  --limit_data 10000' )
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('figure_idx', type=int)
    parser.add_argument('--datasets', nargs='+', type=str, default=["ffhq", "squares", "mnist"])

    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--gpu_memory', default=8, type=int)
    parser.add_argument('--killable', default=True, type=bool)
    parser.add_argument('--hours', default=4, type=int)

    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--n', default=8, type=int)
    args = parser.parse_args()

    project_name = f"Figure{args.figure_idx}"

    figure_generator = globals()[f"Figure{args.figure_idx}"]
    if args.run:
        sbatch_params = args.hours, args.killable, args.gpu_memory
        for data_path, data_args in [data_map[k] for k in args.datasets]:
            figure_generator.send_tasks(project_name, dataset=data_path, additional_params=data_args)

    elif args.plot:
        for data_path, _ in [data_map[k] for k in args.datasets]:
            figure_generator.plot_fig(project_name=project_name, dataset=os.path.basename(data_path))
    else:
        raise ValueError("Please supply at least one task (run, plot)")
