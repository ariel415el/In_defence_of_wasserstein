from compare_batches_emd import *
if __name__ == '__main__':
    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ/FFHQ_128'; c = 3
    outdir = os.path.join("path_size effect", os.path.basename(data_path))
    os.makedirs(outdir, exist_ok=True)

    metric = emd()
    d = 64
    b = 64

    data = get_data(data_path, d, c, limit_data=10000 + 2 * b)

    r1 = data[:b]
    r2 = data[b:2*b]
    data = data[2*b:]

    for k in batches:
        batches[k] = read_grid_batch(batches[k], d, c)
    batches = {
        'r2': r2,
        'centroids': get_centroids(data, b, use_faiss=False)
    }
    patch_sizes = [3,5,7,9]
    for i, (name, batch) in enumerate(batches.items()):
        dists = []
        for z in patch_sizes:
            p = s = z
            x = to_patches(batch, d, c, p, s)
            y = to_patches(r1, d, c, p, s)
            dists.append(metric(x,y))
        plt.plot(patch_sizes, dists, label=name)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot.png"))