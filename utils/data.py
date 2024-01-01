import os
from random import shuffle

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm


def get_transforms(im_size, center_crop,  gray_scale):
    transforms = [T.ToTensor()]

    if center_crop:
        transforms += [T.CenterCrop(size=center_crop)]
    if gray_scale:
        transforms += [T.Grayscale()]
    if im_size is not None:
        transforms += [T.Resize(im_size, antialias=True),]
    transforms+=[T.Normalize((0.5,), (0.5,))]

    return T.Compose(transforms)


class MemoryDataset(Dataset):
    def __init__(self, paths, im_size, center_crop=None, gray_scale=False):
        super(MemoryDataset, self).__init__()
        transforms = get_transforms(im_size, center_crop, gray_scale)

        self.images = []
        for path in tqdm(paths, desc="Loading images into memory"):
            img = Image.open(path).convert('RGB')
            if transforms is not None:
                img = transforms(img)
            self.images.append(img)

        # if on_gpu:
        #     self.images = torch.stack(self.images)
        #     self.images = self.images.cuda()
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class DiskDataset(Dataset):
    def __init__(self, paths, im_size, center_crop=None, gray_scale=False):
        super(DiskDataset, self).__init__()
        self.paths = paths
        self.transforms = get_transforms(im_size, center_crop, gray_scale)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        # return img, idx
        return img


def get_dataloader(data_root, im_size, batch_size, n_workers, val_percentage=0,
                   load_to_memory=False, limit_data=None, gray_scale=False, center_crop=None):
    # paths = [os.path.join(data_root, im_name) for im_name in os.listdir(data_root)]
    # shuffle(paths)
    paths = sorted([os.path.join(data_root, im_name) for im_name in os.listdir(data_root)])
    if limit_data is not None:
        paths = paths[:limit_data]

    n_val_images = int(val_percentage * len(paths))
    train_paths, test_paths = paths[n_val_images:], paths[:n_val_images]
    print(f"Train images: {len(train_paths)}, test images: {len(test_paths)}")

    dataset_type = MemoryDataset if load_to_memory else DiskDataset

    train_dataset = dataset_type(paths=train_paths, im_size=im_size, gray_scale=gray_scale, center_crop=center_crop)
    drop_last = (not limit_data) or (limit_data != batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers,
                              pin_memory=True, drop_last=drop_last)

    test_loader = None
    if val_percentage > 0:
        test_dataset = dataset_type(paths=test_paths, im_size=im_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=n_workers,
                                 pin_memory=True, drop_last=drop_last)

    return train_loader, test_loader
