#!/usr/bin/python3
import random
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer
from core.preprocessor import Preprocessor


class SpamDataset(Dataset):
    """
    Abstract class for the smsSpamCollection

    Args
        path: (string) path to the dataset
        split: (list) list of float [train_pct, valid_pct, test_pct]
    """
    def __init__(self, csv_path, preprocess=None):
        self._data = pd.read_table(csv_path, delimiter=',', encoding='latin-1')
        self._preprocess = preprocess

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        label = self._data.v1[idx]
        sms = self._data.v2[idx]
        input = sms if not self._preprocess else self._preprocess(sms)
        return input, label
