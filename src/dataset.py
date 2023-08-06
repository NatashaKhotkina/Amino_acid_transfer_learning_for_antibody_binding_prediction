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


def make_dataloader(table, sequence_column_name, label_column_name, batch_size=32, shuffle=True):
    # make torch tensors
    X_encoded = torch.Tensor(encoding_func(table, sequence_column_name))
    y = torch.Tensor(table[label_column_name].values)

    # make dataset
    dataset = MyDataset(X_encoded, y)

    # make dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
