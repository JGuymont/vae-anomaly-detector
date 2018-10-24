import unicodedata
import string
from io import open
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

class Data:
    PONCTUATION = [',', '.', ';', ':', '?', '!']

    def __init__(self, path, ponctuation=PONCTUATION, **kargs):
        self._ponctuation = ponctuation
        self._data = self._read_data(path)
        self.preprocessed_data = self._preprocess_data()

    def __len__(self):
        return len(self._inputs)
    
    def __getitem__(self):

    def _read_data(self, path):
        with open(path) as f:
            lines = [line for line in f]
        data = []
        for line in lines:
            y, x = self._preprocess_line(line)
            targets.append(y)
            inputs.append(x)    
        return inputs, targets
        return data[1:]

    def _extract_target(self, line):
        if line[0:3] == 'ham':
            target = 0
            line = line[4:]
        elif line[0:4] == 'spam':
            target = 1
            line = line[5:]
        else:
            exit('line do not have a valid target')
        return target, line

    def _preprocess_line(self, line):
        target, line = self._extract_target(line)
        #line = self._unicode_to_ascii(line)
        #line = self._remove_ponctuation(line)
        #line = line.lower()
        return target, line

    def _preprocess_data(self): 
        inputs = []
        targets = []
        for line in self._data:
            y, x = self._preprocess_line(line)
            targets.append(y)
            inputs.append(x)    
        return inputs, targets

class DataLoader(CountVectorizer):
    
    PONCTUATION = [',', '.', ';', ':', '?', '!']

    def __init__(self, path, ponctuation=PONCTUATION, **kargs):
        self._ponctuation = ponctuation
        self._data = self._read_data(path)
        self.preprocessed_data = self._preprocess_data()
    
        super().__init__(self, **kargs)

        self.vectorized_data = self.fit_transform(self.preprocessed_data[0])

    def _read_data(self, path):
        with open(path) as f:
            data = [line for line in f]
        return data[1:]
    
    
    
    def _remove_ponctuation(self, line):
        for i in self._ponctuation:
            line = line.replace(i, " ")
        return line

    def _unicode_to_ascii(self, line):
        # Turn a Unicode string to plain ASCII.
        # thanks to http://stackoverflow.com/a/518232/2809427
        
        ALL_LETTERS = string.ascii_letters + " .,;'"
        
        return ''.join(
            c for c in unicodedata.normalize('NFD', line)
            if unicodedata.category(c) != 'Mn'
            and c in ALL_LETTERS
        )

    def _preprocess_line(self, line):
        target, line = self._extract_target(line)
        #line = self._unicode_to_ascii(line)
        #line = self._remove_ponctuation(line)
        #line = line.lower()
        return target, line
    
    def _preprocess_data(self):
        inputs = []
        targets = []
        for line in self._data:
            y, x = self._preprocess_line(line)
            targets.append(y)
            inputs.append(x)    
        return inputs, targets

    def bow_to_string(self, bow):
        string = ''
        for i in bow:
            curr_word = self.vocabulary_[i]
            string += '{} '.format(curr_word)
        return string



    
    