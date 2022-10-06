import os

import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, paths, im_size):
        super(ImageFolder, self).__init__()

        transform = transforms.Compose([
            transforms.Resize((int(im_size), int(im_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.images = []
        for path in tqdm(paths, desc="Loading images into memory"):
            img = Image.open(path).convert('RGB')
            if transform is not None:
                img = transform(img)
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def get_datasets(data_root, im_size, val_percentage):
    paths = sorted([os.path.join(data_root, im_name) for im_name in os.listdir(data_root)])

    n_val_images = int(val_percentage * len(paths))
    train_paths, test_paths = paths[n_val_images:], paths[:n_val_images]
    print(f"Train images: {len(train_paths)}, test images: {len(test_paths)}")

    train_dataset = ImageFolder(paths=train_paths, im_size=im_size)
    test_dataset = ImageFolder(paths=test_paths, im_size=im_size)

    return train_dataset, test_dataset


def get_dataloader(data_root, im_size, batch_size, n_workers, val_percentage=0.1):
    train_dataset, test_dataset = get_datasets(data_root, im_size, val_percentage=val_percentage)

    train_loader = iter(DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=False,
                                 sampler=InfiniteSamplerWrapper(train_dataset),
                                 num_workers=n_workers,
                                 pin_memory=True))

    test_loader = iter(DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=n_workers,
                                 pin_memory=True))

    return train_loader, test_loader
