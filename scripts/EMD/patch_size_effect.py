import matplotlib.pyplot as plt

from utils import *
from scripts.EMD.dists import emd, swd, discrete_dual

if __name__ == '__main__':
    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ_128/FFHQ_128'; c = 3
    outdir = os.path.join("patch_size_effect", os.path.basename(data_path))
    os.makedirs(outdir, exist_ok=True)

    device = torch.device('cuda:0')
    # metric_name, metric, max_num_patches = "EMD", emd, 10000
    # metric_name, metric, max_num_patches = "SWD", swd, None
    metric_name, metric, max_num_patches = "Dual", lambda x, y: discrete_dual(x, y, verbose=True), 10000
    d = 64
    b = 64

    data = get_data(data_path, d, c, limit_data=10000 + 2 * b).to(device)

    r1 = data[:b]
    r2 = data[b:2*b]
    data = data[2*b:]

    batches = {
        'Real': r2,
        'centroids': get_centroids(data, b, use_faiss=True),
        'EMD': read_grid_batch("/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-DCGAN_L-BatchEMD_Z-64_B-64_test/images/8000.png",d, c).to(device),
        'FC': read_grid_batch('/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-FC-normalize=none_L-WGANLoss_Z-64_B-64_pixels-FC_FFHQ_128_06-04_T-18:47:41/images/100000.png', d, c).to(device),
        'DC': read_grid_batch('/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-DCGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-DC_FFHQ_128_06-04_T-18:47:44/images/100000.png', d, c).to(device),
        'GAP': read_grid_batch('/home/ariel/Downloads/outputs/FFHQ_128_64x64_G-pixels_D-PatchGAN-normalize=none_L-WGANLoss_Z-64_B-64_pixels-PatchDisc_FFHQ_128_06-04_T-18:47:46/images/100000.png', d, c).to(device)
    }
    plt.figure(figsize=(10, 10), dpi=80)

    patch_sizes = [3, 9, 11, 22, d]
    color = plt.cm.nipy_spectral(np.linspace(0, 1, len(batches)))
    for i, (name, batch) in enumerate(batches.items()):
        dists = []
        stds = []
        for z in patch_sizes:
            torch.cuda.empty_cache()
            p = s = z
            avg_dists = []
            for _ in range(5):
                x = to_patches(batch, d, c, p, 3, limit_patches=max_num_patches)
                y = to_patches(r1, d, c, p, 3, limit_patches=max_num_patches)
                avg_dists.append(metric(x,y))
            avg = np.mean(avg_dists)
            std = np.std(avg_dists)
            dists.append(avg)
            stds.append(std)
        dists = np.array(dists)
        stds = np.array(stds)
        plt.plot(patch_sizes, dists, "-o", label=name)
        plt.fill_between(patch_sizes, dists - stds / 2, dists + stds / 2, alpha=0.15, color=color[i])

    plt.title(f"{p}x{p} Images")
    plt.xlabel("patch-size")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{metric_name}"))
