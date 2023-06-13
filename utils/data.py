import os
from random import shuffle

import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm


def get_transforms(im_size, center_crop):
    transforms = [
             T.ToTensor(),
             T.Resize(im_size, antialias=True),
             T.CenterCrop(size=im_size),
             T.Normalize((0.5,), (0.5,))
        ]

    if center_crop:
        transforms = [T.CenterCrop(size=center_crop)] + transforms

    return T.Compose(transforms)


class MemoryDataset(Dataset):
    def __init__(self, paths, im_size, center_crop=None):
        super(MemoryDataset, self).__init__()
        transforms = get_transforms(im_size, center_crop)

        self.images = []
        for path in tqdm(paths, desc="Loading images into memory"):
            img = Image.open(path).convert('RGB')
            if transforms is not None:
                img = transforms(img)
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class DiskDataset(Dataset):
    def __init__(self, paths, im_size, center_crop=None):
        super(DiskDataset, self).__init__()
        self.paths = paths
        self.transforms = get_transforms(im_size, center_crop)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        # return img, idx
        return img


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


def get_dataloader(data_root, im_size, batch_size, n_workers, val_percentage=0.1, load_to_memory=False):
    # paths = sorted([os.path.join(data_root, im_name) for im_name in os.listdir(data_root)])
    paths = [os.path.join(data_root, im_name) for im_name in os.listdir(data_root)]
    shuffle(paths)

    n_val_images = int(val_percentage * len(paths))
    train_paths, test_paths = paths[n_val_images:], paths[:n_val_images]
    print(f"Train images: {len(train_paths)}, test images: {len(test_paths)}")

    dataset_type = MemoryDataset if load_to_memory else DiskDataset

    train_dataset = dataset_type(paths=train_paths, im_size=im_size)
    test_dataset = dataset_type(paths=test_paths, im_size=im_size)

    train_loader = iter(DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=False,
                                 sampler=InfiniteSamplerWrapper(train_dataset),
                                 num_workers=n_workers,
                                 pin_memory=True))

    test_loader = None
    if val_percentage > 0:
        test_loader = iter(DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False,
                                     sampler=InfiniteSamplerWrapper(test_dataset),
                                     num_workers=n_workers,
                                     pin_memory=True))

    return train_loader, test_loader
