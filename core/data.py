import random
import string
import re
import numpy as np
import pandas as pd
import nltk

import torch
from torch.autograd import Variable

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


from sklearn.feature_extraction.text import CountVectorizer

from core.url_regex import url_regex2 as URL_REGEX

class Data(CountVectorizer):
    """Abstract class for the smsSpamCollection

    Args
        path: (string) path to the dataset
        split: (list) list of float [train_pct, valid_pct, test_pct]
    """
    PUNCTUATION = [',', '.', ';', ':', '?', '!']
    STOPWORDS = set(stopwords.words('english'))
    LABELS = {'spam': 1, 'ham': 0}

    def __init__(self, path, split, labels=LABELS, punctuation=PUNCTUATION, stopwords=STOPWORDS, **kargs):
        super().__init__(self, binary=True, min_df=3, token_pattern=r"(?u)\b\w\w+\b|@\w*@|\?|\"|\'", **kargs)
        self.labels = labels
        self._punctuation = punctuation
        self._dataframe = pd.read_table(path, delimiter=',')
        self._data = self._read_data()
        self._data_index = self._split_index(split)
        self.stopwords = stopwords

        # self.stemmer = nltk.stem.SnowballStemmer('english')
        self.stemmer = nltk.stem.PorterStemmer()

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
            targets[targets == label] = self.labels[label]
        return targets

    def _remove_punctuation(self, sms):
        for punc in self._punctuation:
            sms = sms.replace(punc, ' ')
        return sms

    def _replace_dollar_sign(self, sms):
        sms = sms.replace('$', ' @dollar_sign@ ')
        return sms

    def _replace_email(self, sms):
        regex = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
        sms = regex.sub(' @email_address@ ', sms)
        return sms

    def _replace_url(self, sms):
        regex = re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", re.IGNORECASE)
        sms = regex.sub(' @url@ ', sms)
        return sms

    def _replace_one_letter_word(self, sms):
        for letter in list(string.ascii_lowercase):
            regex = re.compile(r"\b[{}]\b".format(letter), re.IGNORECASE)
            sms = regex.sub('@{}@'.format(letter), sms)
        return sms

    def _replace_long_digit_seq(self, sms):
        regex = re.compile(r"\b\d\d\d\d\d\d\d\d+\b", re.IGNORECASE)
        sms = regex.sub('@long_digit@', sms)
        return sms

    def replace_medium_digit_seq(self, string_):
        regex = re.compile(r"\b0\d\d\d+\b", re.IGNORECASE)
        string_ = regex.sub('@medium_digit@', string_)
        return string_

    def replace_small_digit_seq(self, string_):
        regex = re.compile(r"\b0\d+\b", re.IGNORECASE)
        string_ = regex.sub('@small_digit@', string_)
        return string_

    def replace_number(self, string_):
        regex = re.compile(r"\b0|[1-9][0-9]+\b", re.IGNORECASE)
        string_ = regex.sub('@number@', string_)
        return string_

    def _preprocess(self, x):
        """Preprocess a string
        """
        x = x.lower()
        x = self._replace_dollar_sign(x)
        x = self._replace_email(x)
        x = self._replace_url(x)
        x = self._replace_one_letter_word(x)
        x = self._replace_long_digit_seq(x)
        x = self.replace_medium_digit_seq(x)
        x = self.replace_small_digit_seq(x)
        x = self.replace_number(x)
        # x = self._remove_punctuation(x)
        x = x.split()
        x = [self.stemmer.stem(w) for w in x]
        x = [w for w in x if w not in self.stopwords]
        return ' '.join(x)

    def preprocess(self):
        text_messages = self._dataframe.message
        preprocessed_text_messages = []
        for sms in text_messages:
            preprocessed_sms = self._preprocess(sms)
            preprocessed_text_messages.append(preprocessed_sms)
        targets = self._relabel(self._dataframe.target)
        self._data = list(zip(preprocessed_text_messages, targets))

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
        preprocessed_sms_file = open("./data/preprocessed_sms.txt", "w")
        raw_document = self._get_raw_document()
        self.fit(raw_document)
        for i in range(len(self)):
            input, target = self[i]
            preprocessed_sms_file.write(' '.join([w for w in input.split() if w in self.vocabulary_])+'\n')
            self._data[i] = (self.transform([input]).toarray(), target)
        preprocessed_sms_file.close()
        self._log_vocabulary()

    def _log_vocabulary(self):
        vocab_file = open("./data/vocabulary.txt", "w")
        vocabulary = list(self.vocabulary_.keys())
        for w in sorted(vocabulary):
            vocab_file.write(w+'\n')
        vocab_file.close()

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

    def data_size_(self):
        return len(self)

    def input_dim_(self):
        return len(self.vocabulary_)

    def inputs_(self):
        inputs, _ = zip(*self._data)
        return list(inputs)

    def targets_(self):
        _, targets = zip(*self._data)
        return list(targets)

    def spam_count_(self):
        return self.targets_().count(self.labels['spam'])

    def spam_percentage_(self):
        return self.spam_count_() / self.data_size_()

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
