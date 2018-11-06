import random
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

from sklearn.feature_extraction.text import CountVectorizer

class Data(CountVectorizer):
    """Abstract class for the smsSpamCollection

    Args
        path: (string) path to the dataset
        split: (list) list of float [train_pct, valid_pct, test_pct]
    """

    PUNCTUATION = [',', '.', ';', ':', '?', '!']
    LABELS = {'spam': 1, 'ham': 0}

    def __init__(self, path, split, labels=LABELS, punctuation=PUNCTUATION, **kargs):
        super().__init__(self, binary=True, **kargs)
        self.labels = labels
        self._punctuation = punctuation
        self._dataframe = pd.read_table(path, delimiter=',')
        self._data = self._read_data()
        self._data_index = self._split_index(split)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def _read_data(self):
        sms = self._dataframe.message
        targets = self._dataframe.target
        return list(zip(sms, targets))

    def _split_index(self, split):
        storage = {'train': [], 'valid': [], 'test': []}
        train_size = round(len(self)*split[0])
        valid_size = round((len(self) - train_size)*split[1])

        examples = range(len(self))
        storage['train'] = random.sample(examples, train_size)
        examples = [ex for ex in examples if ex not in storage['train']] # remove index
        storage['valid'] = random.sample(examples, valid_size)
        storage['test'] = [ex for ex in examples if ex not in storage['valid']]
        return storage

    def _relabel(self, targets):
        for label in self.labels.keys():
            targets[targets==label] = self.labels[label]
        return targets

    def _remove_punctuation(self, sms):
        for punc in self._punctuation:
            sms = sms.replace(punc, ' ')
        return sms

    def _preprocess(self, x):
        x = x.lower()
        x = self._remove_punctuation(x)
        return x

    def preprocess(self):
        sms = self._dataframe.message
        sms = [self._preprocess(s) for s in sms]
        targets = self._relabel(self._dataframe.target)
        self._data = list(zip(sms, targets))

    def tokenize(self):
        for i in range(len(self)):
            input, target = self[i]
            self._data[i] = (input.split(), target)
    
    def _get_raw_document(self):
        raw_document = []
        for i in self._data_index['train']:
            input, _ = self[i]
            raw_document.append(input)
        return raw_document

    def vectorize(self):
        """Count vectorization

        *) Word of one letter are removed*
        *) Apostrophe are removed
        """
        raw_document = self._get_raw_document()
        self.fit(raw_document)
        for i in range(len(self)):
            input, target = self[i]
            self._data[i] = (self.transform([input]).toarray(), target)

    def train(self):
        return [self._data[i] for i in self._data_index['train']]

    def valid(self):
        return [self._data[i] for i in self._data_index['valid']]

    def test(self):
        return [self._data[i] for i in self._data_index['test']]

    def bow_to_string(self, bow):
        """does not work"""
        string = ''
        for i in bow:
            curr_word = self.vocabulary_[i]
            string += '{} '.format(curr_word)
        return string

    def input_dim_(self):
        return len(self.vocabulary_)

class DataLoader:

    def __init__(self, data, batch_size):
        self._data = data
        self._inputs, self._targets = [list(t) for t in zip(*data)] 
        self._batch_size = batch_size
        self._data_size = len(self._data)
        self._n_batch = self._data_size // self._batch_size + 1

        self._dataloader = self._split_in_batch()

    def __len__(self):
        pass

    def __getitem__(self, i):
        return self._dataloader[i]

    def _stack_inputs(self, inputs):
        pass

    def _get_next_input_batch(self, last=False):
        last_idx = self._batch_size if not last else len(self._inputs)
        cur_batch = np.array([self._inputs.pop(0)[0] for _ in range(last_idx)])
        return cur_batch

    def _get_next_target_batch(self, last=False):
        last_idx = self._batch_size if not last else len(self._targets)
        cur_batch = np.array([self._targets.pop(0) for _ in range(last_idx)])
        return cur_batch

    def _split_in_batch(self):
        storage = {'inputs': [], 'targets': []}
        for i in range(self._n_batch-1):
            cur_batch_inputs = Variable(torch.Tensor(self._get_next_input_batch()))
            cur_batch_targets = Variable(torch.LongTensor(self._get_next_target_batch()))
            storage['inputs'].append(cur_batch_inputs)
            storage['targets'].append(cur_batch_targets)
        cur_batch_inputs = Variable(torch.Tensor(self._get_next_input_batch(last=True)))
        cur_batch_targets = Variable(torch.LongTensor(self._get_next_target_batch(last=True)))
        storage['inputs'].append(cur_batch_inputs)
        storage['targets'].append(cur_batch_targets) 
        return list(zip(storage['inputs'], storage['targets']))
