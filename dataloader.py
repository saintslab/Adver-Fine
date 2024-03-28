# dataloader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

class FASHIONMNISTDataLoader:
    def __init__(self, batch_size=64, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self._prepare_data()

    def _prepare_data(self):
        # Load FASHIONMNIST dataset
        train_data = datasets.FashionMNIST('data', train=True, download=True, transform=self._transform)
        test_data = datasets.FashionMNIST('data', train=False, download=True, transform=self._transform)

        # Prepare data loader for training (no validation set)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_classes(self):
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                'Sneaker', 'Bag', 'Ankle boot']

    

class CIFAR10DataLoader:
    '''
    Example usage:
    data_loader = CIFAR10DataLoader()
    train_loader = data_loader.train_loader
    valid_loader = data_loader.valid_loader
    test_loader = data_loader.test_loader
    classes = data_loader.get_classes()
    '''
    def __init__(self, batch_size=64, valid_size=0.2, num_workers=0):
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self._prepare_data()

    def _prepare_data(self):
        # Load CIFAR-10 dataset
        train_data = datasets.CIFAR10('data', train=True, download=True, transform=self._transform)
        test_data = datasets.CIFAR10('data', train=False, download=True, transform=self._transform)

        # Obtain training indices for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # Define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Prepare data loaders (combine dataset and sampler)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers)
        self.valid_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=valid_sampler, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_classes(self):
        return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


