import h5py
import glob
import os
import bisect
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

class CustomHDF5Dataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.cumulative_lengths = [0]
        for file in self.file_list:
            with h5py.File(file, 'r') as hf:
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(hf['activations']))

    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        with h5py.File(self.file_list[file_idx], 'r') as hf:
            return torch.Tensor(hf['activations'][idx - self.cumulative_lengths[file_idx]])

class HDF5Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        with h5py.File(self.path, 'r') as hf:
            self.length = len(hf['activations'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as hf:
            return torch.Tensor(hf['activations'][idx])

def get_datasets_and_dataloaders_multihdf5(path, batch_size = 4096, split = None):
    """
    Assumes that the data is stored in the following format:
    path
    ├── train
    │   ├── filename0.h5
    │   ├── filename1.h5
    │   ├── ...
    │   └── filenameN.h5
    ├── val (optional)
        ├── filename0.h5
        ├── filename1.h5
        ├── ...
        └── filenameN.h5

    OR 

    path
    ├── filename0.h5
    ├── filename1.h5
    ├── ...
    └── filenameN.h5

    """
    # Get the filenames for each split
    train_file_list = glob.glob(os.path.join(path, 'train', '*.h5'))
    val_file_list = glob.glob(os.path.join(path, 'val', '*.h5'))
    if train_file_list == [] and val_file_list == []:
        # If no train or val folders are provided, assume all files in path are training data
        train_file_list = glob.glob(os.path.join(path, '*.h5'))
    if val_file_list == []:
        # If no validation set is provided, split the training set
        train_dataset = CustomHDF5Dataset(train_file_list)
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        train_dataset = CustomHDF5Dataset(train_file_list)
        val_dataset = CustomHDF5Dataset(val_file_list)

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    return train_dataset, val_dataset, train_dataloader, val_dataloader

def get_datasets_and_dataloaders_hdf5(path, batch_size = 4096, split = None):
    """
    Assumes that the data is stored in the following format:
    path
    ├── train
    │   ├── filename.h5
    ├── val (optional)
        ├── filename.h5

    OR
    path
    ├── filename.h5
    """
    # Get the filenames for each split
    train_file_list = glob.glob(os.path.join(path, 'train', '*.h5'))
    val_file_list = glob.glob(os.path.join(path, 'val', '*.h5'))
    if train_file_list == [] and val_file_list == []:
        # If no train or val folders are provided, assume all files in path are training data
        train_file_list = glob.glob(os.path.join(path, '*.h5'))
    if val_file_list == []:
        # If no validation set is provided, split the training set
        train_dataset = HDF5Dataset(train_file_list[0])
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        train_dataset = HDF5Dataset(train_file_list[0])
        val_dataset = HDF5Dataset(val_file_list[0])

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    return train_dataset, val_dataset, train_dataloader, val_dataloader