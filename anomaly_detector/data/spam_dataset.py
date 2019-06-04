#!/usr/bin/python3
import pandas
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    """Abstract class for the smsSpamCollection

    :param csv_path: (string) path to the dataset
    :param label2int:
    :param transform:
    """
    def __init__(self, csv_path, label2int=None, transform=None):
        self._data = pandas.read_table(csv_path, delimiter=',', encoding='latin-1')
        self._label2int = label2int
        self._transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # Input processing
        input = self._data.v2[idx]
        input = input if not self._transform else self._transform(input)

        # Target processing
        target = self._data.v1[idx]
        target = self._label2int[target] if self._label2int else target

        return input, target

    @property
    def input_dim_(self):
        return len(self[0][0])
