import unicodedata
import string
from io import open
import pandas as pd

class Data:
    
    def __init__(self):
        self.ponctuation = [',', '.', ';', ':', "'", '?', '!']
    
    def extract_target(self, line):
        if line[0:3] == 'ham':
            target = 0
            line = line[4:]
        elif line[0:4] == 'spam':
            target = 1
            line = line[5:]
        else:
            exit('line do not have a valid target')
        return target, line
    
    def remove_comma(self, line):
        return line.replace(',', " ")
    
    def remove_dot(self, line):
        return line.replace('.', " ")
    
    def remove_apostrophe(self, line):
        return line.replace("'", " ")

    def remove_semicolon(self, line):
        return line.replace(";", " ")

    def remove_colon(self, line):
        return line.replace(":", " ")
    
    def remove_ponctuation(self, line):
        for i in self.ponctuation:
            line = line.replace(i, " ")
        return line

    def unicode_to_ascii(self, line):
        # Turn a Unicode string to plain ASCII.
        # thanks to http://stackoverflow.com/a/518232/2809427
        
        ALL_LETTERS = string.ascii_letters + " .,;'"
        
        return ''.join(
            c for c in unicodedata.normalize('NFD', line)
            if unicodedata.category(c) != 'Mn'
            and c in ALL_LETTERS
        )

    def preprocess(self, line):
        target, line = self.extract_target(line)
        line = self.remove_ponctuation(line)
        line = self.unicode_to_ascii(line)
        line = line.lower()
        return target, line.split()

def read_csv(path):

    inputs = []
    targets = []

    data = Data()
    with open(path) as f:
        lines = [line for line in f]

    for line in lines[1:]:
        y, x = data.preprocess(line)
        targets.append(y)
        inputs.append(x)    
    return inputs, targets
    