import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.one_hot import encoding_func


class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        return features, labels


def make_dataloader(table, batch_size=32, shuffle=True):
    # make torch tensors
    X_encoded = torch.Tensor(encoding_func(table))
    y = torch.Tensor(table['Label'].values)

    # make dataset
    dataset = MyDataset(X_encoded, y)

    # make dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
