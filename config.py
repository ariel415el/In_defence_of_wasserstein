import argparse

args = argparse.Namespace()
args.data_path = '/mnt/storage_ssd/datasets/FFHQ_128'
# args.data_path = '/mnt/storage_ssd/datasets/few-shot-images/100-shot-obama/img'
args.architecture = 'FastGAN'
args.batch_size = 16
args.n_iterations = 50000
args.im_size = 128
args.z_dim = 128
args.lr = 0.0002
args.nbeta1 = 0.5
args.name = f"FFHQ_{args.architecture}_Z-{args.z_dim}_B-{args.batch_size}"

args.n_workers = 0
args.save_interval = 1000