import numpy as np
import random
import torch
import torchvision
from tqdm.auto import tqdm
import hashlib

from torchvision import transforms
from torch.utils.data import Subset
import torchvision.transforms.functional as F


def load_dataset(hparams):
    name = hparams["dataset"]
    path = hparams["dataset_path"]

    if name == "CIFAR10":
        train_data_raw, train_data, test_data = load_cifar10(path)
    elif name == "CIFAR100":
        train_data_raw, train_data, test_data = load_cifar100(path)
    elif name == "SVHN":
        train_data_raw, train_data, test_data = load_svhn(path)
    else:
        raise Exception(f"Dataset '{name}' is not implemented")

    # Split the training data into num_train_splits partitions
    print("Partitioning the training data")
    train_split_idx = partition_data(train_data_raw,
                                     hparams['k_t'])

    list_train_data_raw = [Subset(train_data_raw, idx)
                           for idx in train_split_idx]

    # sort the data in each partition according to raw files,
    # but return dataset with preprocessing transformations
    print("Sorting the training data")
    list_train_data = sort_data(list_train_data_raw,
                                train_data,
                                train_split_idx)

    # Compute the mean and std of each training partition
    print("Computing the mean and std of each partition")
    means, stds = partition_statistics(list_train_data_raw)

    return list_train_data, test_data, means, stds


def partition_statistics(list_train_data_raw):
    means = []
    stds = []
    for i in tqdm(range(len(list_train_data_raw))):
        length = len(list_train_data_raw[i])
        train_loader = torch.utils.data.DataLoader(list_train_data_raw[i],
                                                   batch_size=length,
                                                   shuffle=False,
                                                   num_workers=0)
        input = next(iter(train_loader))[0]
        means.append(input.mean((0, 2, 3)))
        stds.append(input.std((0, 2, 3)))
    return means, stds


def sort_data(list_train_data_raw, train_data, train_split_idx):
    list_train_data = []
    for partition in tqdm(range(len(list_train_data_raw))):
        imgs = torch.stack([img
                            for img, label in list_train_data_raw[partition]])
        imgs = imgs.reshape(imgs.shape[0], -1)
        labels = torch.stack([torch.tensor(label).unsqueeze(-1)
                              for img, label in list_train_data_raw[partition]]
                             )
        data = torch.cat([labels, imgs], dim=1)
        sorted_idx = np.lexsort(data.numpy().T)

        sorted_data = Subset(train_data, train_split_idx[partition])
        sorted_data = Subset(sorted_data, sorted_idx)
        list_train_data.append(sorted_data)
    return list_train_data


def partition_data(train_data_raw, num_partitions):
    train_split_idx = [[] for _ in range(num_partitions)]
    enumerator = tqdm(enumerate(train_data_raw), total=len(train_data_raw))
    for idx, (image, label) in enumerator:
        index = int(image.sum()) % num_partitions
        train_split_idx[index].append(idx)

    train_split_sizes = [len(split) for split in train_split_idx]
    assert sum(train_split_sizes) == len(train_data_raw)
    return train_split_idx


def load_cifar10(path):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_raw = torchvision.datasets.CIFAR10(path,
                                             train=True,
                                             download=True,
                                             transform=transforms.ToTensor())

    train_data = torchvision.datasets.CIFAR10(path,
                                              train=True,
                                              download=True,
                                              transform=transform_train)

    test_data = torchvision.datasets.CIFAR10(path,
                                             train=False,
                                             download=True,
                                             transform=transforms.ToTensor()
                                             )

    return train_raw, train_data, test_data


def load_cifar100(path):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_raw = torchvision.datasets.CIFAR100(path,
                                              train=True,
                                              download=True,
                                              transform=transforms.ToTensor())
    train_data = torchvision.datasets.CIFAR100(path,
                                               train=True,
                                               download=True,
                                               transform=transform_train)
    test_data = torchvision.datasets.CIFAR100(path,
                                              train=False,
                                              download=True,
                                              transform=transforms.ToTensor()
                                              )

    return train_raw, train_data, test_data


def load_svhn(path):
    train_data = torchvision.datasets.SVHN(path,
                                           split="train",
                                           download=True,
                                           transform=transforms.ToTensor())

    test_data = torchvision.datasets.SVHN(path,
                                          split="test",
                                          download=True,
                                          transform=transforms.ToTensor())
    return train_data, train_data, test_data
