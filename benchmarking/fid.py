import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

from benchmarking.inception import InceptionV3


def calc_fid(stats1, stats2, eps=1e-6):
    mean1, cov1 = stats1
    mean2, cov2 = stats2
    cov_sqrt, _ = linalg.sqrtm(cov1 @ cov2, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(cov1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((cov1 + offset) @ (cov1 + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = mean1 - mean2
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


inception = None


class FID_score:
    def __init__(self, loaders, num_batches, device):
        self.device = device
        print("Computing reference Inception features", end='...', flush=True)
        global inception
        if inception is None:
            inception = InceptionV3([3], normalize_input=False).to(device)
        self.ref_stats_dict = {k: self.get_multi_batch_statistics([next(loader).to(device) for _ in range(num_batches)]) for k,loader in loaders.items()}
        print("Done")

    def __call__(self, batches):
        stats = self.get_multi_batch_statistics(batches)
        return {k: calc_fid(ref_stats, stats) for k, ref_stats in self.ref_stats_dict.items()}


    def get_multi_batch_statistics(self, batches):
        global inception
        with torch.no_grad():
            features = [inception(batch.to(self.device))[0].view(batch.shape[0], -1).cpu().numpy() for batch in batches]
        features = np.concatenate(features, axis=0)
        return np.mean(features, 0), np.cov(features, rowvar=False)

