import argparse

args = argparse.Namespace()
args.data_path = '/cs/labs/yweiss/ariel1/data/FFHQ_1000_images'
args.architecture = 'FastGAN'
args.batch_size = 64
args.n_iterations = 100000
args.im_size = 128
args.z_dim = 128
args.lr = 0.0002
args.nbeta1 = 0.5
args.name = f"FFHQ_{args.architecture}_Z-{args.z_dim}_B-{args.batch_size}"
args.augmentaion='color,translation'
args.n_workers = 8
args.save_interval = 1000