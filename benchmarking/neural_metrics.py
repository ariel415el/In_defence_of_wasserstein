import numpy as np
import torch

from benchmarking.fid import frechet_distance
from benchmarking.inception import InceptionV3
from benchmarking.prdc import compute_prdc

inception = None


class InceptionMetrics:
    def __init__(self, batches, device):
        self.device = device
        if batches:
            self.ref_features = self.process_batches(batches)
            self.ref_stats = np.mean(self.ref_features, 0), np.cov(self.ref_features, rowvar=False)
        print("Done")

    def process_batches(self, batches):
        with torch.no_grad():
            global inception
            if inception is None:
                inception = InceptionV3([3], normalize_input=False).to(self.device)
            features = []
            for batch in batches:
                feature = inception(batch.to(self.device))[0].view(batch.shape[0], -1)
                feature = feature.cpu().numpy()
                features.append(feature)
        features = np.concatenate(features, axis=0)
        return features

    def __call__(self, batches):
        if self.ref_features is None:
            return dict()
        fake_features = self.process_batches(batches)

        result_dict = compute_prdc(real_features=self.ref_features, fake_features=fake_features, k=5)

        fake_stats = np.mean(fake_features, 0), np.cov(fake_features, rowvar=False)
        result_dict["FID"] = frechet_distance(self.ref_stats, fake_stats)

        return result_dict