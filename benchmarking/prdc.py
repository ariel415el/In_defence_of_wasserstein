"""
prdc
Copyright (c) 2020-present NAVER Corp.
MIT license
Code based on https://github.com/clovaai/generative-evaluation-prdc
"""
import numpy as np
import sklearn.metrics


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_self_knn_distances(input_features, k):
    """
    Geth the k-nearest neighbor distance for each rown in input_features
    :param input_features: array of shape (n,d)
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, k):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    :param real_features: array of shape (N, d)
    :param fake_features: array of shape (N, d)
    :param k: k-nearest neighbor
    :return: dict
    """

    print("Computing PRDC", end='...')
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)
    real_knn_distances = compute_self_knn_distances(real_features, k)[:, None]
    fake_knn_distances = compute_self_knn_distances(fake_features, k)[None, :]

    precision = (distance_real_fake < real_knn_distances).any(axis=0).mean()

    recall = (distance_real_fake < fake_knn_distances ).any(axis=1).mean()

    density = (1. / float(k)) * (distance_real_fake < real_knn_distances).sum(axis=0).mean()

    coverage = (distance_real_fake.min(axis=1) < real_knn_distances).mean()

    result_dict = dict(precision=precision, recall=recall, density=density, coverage=coverage)
    return result_dict


if __name__ == '__main__':
    num_real_samples = num_fake_samples = 10000
    feature_dim = 1000
    k = 6
    real_features = np.random.normal(loc=0.0, scale=1.0,
                                     size=[num_real_samples, feature_dim])

    fake_features = np.random.normal(loc=0.0, scale=1.0,
                                     size=[num_fake_samples, feature_dim])

    metrics = compute_prdc(real_features=real_features,
                           fake_features=fake_features,
                           k=k)

    print(metrics)