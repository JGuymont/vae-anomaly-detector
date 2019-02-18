#!/usr/bin/python3
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


class SpamDataset(Dataset):
    """
    Abstract class for the smsSpamCollection

    Args
        path: (string) path to the dataset
        split: (list) list of float [train_pct, valid_pct, test_pct]
    """
    def __init__(self, csv_path, label2int=None, transform=None):
        self._data = pd.read_table(csv_path, delimiter=',', encoding='latin-1')
        self._label2int = label2int
        self._transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # Input processing
        input = self._data.v2[idx]
        input = input if not self._transform else self._transform(input)
        # input = Variable(torch.Tensor(input))

        # Target processing
        target = self._data.v1[idx]
        target = self._label2int[target] if self._label2int else target
        # target = Variable(torch.LongTensor([target]))

        return input, target

    @property
    def input_dim_(self):
        return len(self[0][0])
