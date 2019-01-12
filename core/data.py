#!/usr/bin/python3
"""
Prepare the data
"""
import random
import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer
from core.preprocessor import Preprocessor

TOKEN_PATTERN = r"(?u)\b\w\w+\b|<\w*>|\?|\"|\'"

class Data(CountVectorizer):
    """
    Abstract class for the smsSpamCollection

    Args
        path: (string) path to the dataset
        split: (list) list of float [train_pct, valid_pct, test_pct]
    """
    def __init__(self, config):
        self.token_type = config['tokenizer']
        self.seed = config['seed']
        tokenizer = list if self.token_type == 'char' else None
        token_pattern = TOKEN_PATTERN if self.token_type == 'word' else None
        preprocessor = Preprocessor()
        preprocess = preprocessor.preprocess if config['tokenizer'] == 'word' else None
        super().__init__(self, min_df=config['min_df'],
                         lowercase=config['lowercase'],
                         preprocessor=preprocess,
                         tokenizer=tokenizer,
                         token_pattern=token_pattern,
                         ngram_range=config['ngram_range'],
                         binary=config['binary'])

        self._labels = config['labels']
        self._spam_pct = config['spam_pct']

        train_path = os.path.join(config['dir'], config['train_file'])
        test_path = os.path.join(config['dir'], config['test_file'])
        
        self.train = self._read_data(train_path)
        self.test = self._read_data(test_path)

    def _read_data(self, path):
        """
        spam_pct = spam / (ham + spam)
        spam = spam_pct * ham / (1 - spam_pct)
        """
        dataframe = pd.read_table(path, delimiter=',')
        if self._spam_pct:
            random.seed(self.seed)
            spam_idx = dataframe.index[dataframe['target'] == 'spam'].tolist()
            len_ham = len(dataframe) - len(spam_idx)
            len_spam = int(self._spam_pct * len_ham / (1 - self._spam_pct))
            idx_to_remove = random.sample(spam_idx, len(spam_idx) - len_spam)
            dataframe = dataframe.drop(idx_to_remove, axis=0)
        sms = dataframe.message
        if self.token_type == 'char': 
            sms = [s.replace(' ', '') for s in sms]
        targets = self._relabel(dataframe.target)
        return list(zip(sms, targets))

    def _relabel(self, targets):
        for label in self._labels.keys():
            targets[targets == label] = self._labels[label]
        return targets

    def _get_raw_document(self):
        raw_document = []
        for text, _ in self.train:
            raw_document.append(text)
        return raw_document

    def vectorize(self):
        """Count vectorization

        *) Word of one letter are removed*
        *) Apostrophe are removed
        """
        raw_document = self._get_raw_document()
        self.fit(raw_document)
        for i, (text, target) in enumerate(self.train):
            self.train[i] = (self.transform([text]).toarray(), target)
        for i, (text, target) in enumerate(self.test):
            self.test[i] = (self.transform([text]).toarray(), target)
        self._log_vocabulary()

    def _log_vocabulary(self):
        vocab_file = open("./data/vocabulary.txt", "w")
        vocabulary = list(self.vocabulary_.keys())
        for w in sorted(vocabulary):
            vocab_file.write(w+'\n')
        vocab_file.close()

    @property
    def data_size_(self):
        """
        Number of examples
        """
        return len(self.train)

    @property
    def targets_(self):
        """
        Targets of the data
        """
        _, targets = zip(*self.train)
        return list(targets)

    @property
    def input_dim_(self):
        """
        Number of features
        """
        return len(self.vocabulary_)

    @property
    def spam_count_(self):
        """
        Number of spam if the data
        """
        return self.targets_.count(self._labels['spam'])

    @property
    def spam_percentage_(self):
        """
        Percentage of spam in the data
        """
        return self.spam_count_ / self.data_size_

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
        cur_batch_inputs = self._get_next_input_batch(last=True)
        cur_batch_targets = self._get_next_target_batch(last=True)
        if not cur_batch_inputs.size == 0:
            storage['inputs'].append(Variable(torch.Tensor(cur_batch_inputs)))
            storage['targets'].append(Variable(torch.LongTensor(cur_batch_targets))) 
        return list(zip(storage['inputs'], storage['targets']))

    def data_size_(self):
        return self._data_size
