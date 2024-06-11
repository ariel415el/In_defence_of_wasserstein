import math

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch.utils.data as data


def generate_circular_coordinates(num_points, radius):
    coordinates = []
    angle = 2 * math.pi / num_points  # Angle between each point

    for i in range(num_points):
        x = radius * math.cos(i * angle)
        y = radius * math.sin(i * angle)
        coordinates.append((x, y))

    return coordinates

def generate_uniform_coordinates(num_points, radius):
    coordinates = []

    for i in range(num_points):
        x = np.random.uniform(-radius/2, radius/2)
        y =np.random.uniform(-radius/2, radius/2)
        coordinates.append((x, y))

    return coordinates


def generate_data(num_centroids=8, samples_per_centroid=100, radius=10, std=0.25, mode="Uniform"):
    if mode == "Uniform":
        centers = generate_uniform_coordinates(num_centroids, radius)
    else:
        centers = generate_circular_coordinates(num_centroids, radius)

    points = []
    for center in centers:
        points.append(np.random.multivariate_normal(center, np.eye(2) * std, samples_per_centroid))
    return np.concatenate(points, axis=0)

class NPDataset(Dataset):
    def __init__(self, np_data_path):
        super(NPDataset, self).__init__()
        self.np_data = np.load(open(np_data_path, "rb"))

    def __len__(self):
        return len(self.np_data)

    def __getitem__(self, idx):
        return self.np_data[idx]


if __name__ == '__main__':
    mode = "Circular"; num_centroids = 8; samples_per_centroid = 256; radius = 10; std = 0.3
    # mode = "Uniform"; num_centroids = 32; samples_per_centroid = 10; radius = 10; std = 0.01


    points = generate_data(num_centroids, samples_per_centroid, radius, std, mode)
    np.save(open(f"{mode}-points-(NC-{num_centroids}_NS={samples_per_centroid}_R={radius}_STD={std})", "wb"), points)
    # Plotting
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=3)
    plt.axis('equal')  # Set equal aspect ratio
    plt.title('Points on a Circle')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()