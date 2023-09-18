# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

prepare dataloader

@author: tadahaya
"""
import time
from tqdm import tqdm
import numpy as np
from typing import Tuple

import torch
import torchvision.transforms as transforms

# frozen
class MyDataset(torch.utils.data.Dataset):
    """ to create my dataset """
    def __init__(self, data=None, label=None, transform=None):
        if data is None:
            raise ValueError('!! Give data !!')
        if label is None:
            label = np.full(len(data), np.nan)
        if type(transform) != list:
            self.transform = [transform]
        else:
            self.transform = transform
        self.data = data
        self.label = label
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            for t in self.transform:
                out_data = t(out_data)
        return out_data, out_label


class MyTransforms:
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.from_numpy(x.astype(np.float32))  # example
        return x


def prep_dataset(data, label=None, transform=None) -> torch.utils.data.Dataset:
    """
    prepare dataset from row data
    
    Parameters
    ----------
    data: array
        input data such as np.array

    label: array
        input labels such as np.array
        would be None with unsupervised learning

    transform: a list of transform functions
        each function should return torch.tensor by __call__ method
    
    """
    return MyDataset(data, label, transform)


def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True
    ) -> torch.utils.data.DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
        )    
    return loader


def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prep_data(
    train_x, train_y, test_x, test_y, batch_size,
    transform=(None, None), shuffle=(True, False),
    num_workers=None, pin_memory=None
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    prepare train and test loader from data
    combination of prep_dataset and prep_dataloader for model building
    
    Parameters
    ----------
    train_x, train_y, test_x, test_y: arrays
        arrays for training data, training labels, test data, and test labels
    
    batch_size: int
        the batch size

    transform: a tuple of transform functions
        transform functions for training and test, respectively
        each given as a list
    
    shuffle: (bool, bool)
        indicates shuffling training data and test data, respectively
    
    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing    

    """
    train_dataset = prep_dataset(train_x, train_y, transform[0])
    test_dataset = prep_dataset(test_x, test_y, transform[1])
    train_loader = prep_dataloader(
        train_dataset, batch_size, shuffle[0], num_workers, pin_memory
        )
    test_loader = prep_dataloader(
        test_dataset, batch_size, shuffle[1], num_workers, pin_memory
        )
    return train_loader, test_loader