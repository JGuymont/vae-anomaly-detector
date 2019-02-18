import json
import pandas
import numpy
import torch
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessor import WordLevelPreprocessor, CharLevelPreprocessor
from constants import CHARACTER, WORD, TOKEN_PATTERN


class FeatureExtractor(CountVectorizer):
    def __init__(self, config):
        token_type = config.get("features", "token_type")

        preprocessor = WordLevelPreprocessor() if token_type == WORD else CharLevelPreprocessor()

        param = {}
        param['lowercase'] = config.getboolean("features", "lowercase")
        param['preprocessor'] = preprocessor.preprocess
        param['tokenizer'] = list if token_type == CHARACTER else None
        param['token_pattern'] = TOKEN_PATTERN if token_type == WORD else None
        param['ngram_range'] = json.loads(config.get("features", "ngram_range"))
        param['min_df'] = config.getint("features", "min_df")
        param['binary'] = config.getboolean("features", "binary")

        super().__init__(self, **param)

    def get_raw_documents(self, csv_path):
        dataframe = pandas.read_table(csv_path, delimiter=',', encoding='latin-1')
        messages = dataframe.v2
        corpus = [text for text in messages]
        return corpus

    def log_vocabulary(self, path):
        vocab_file = open(path, "w", encoding='latin-1')
        vocabulary = list(self.vocabulary_.keys())
        for w in sorted(vocabulary):
            vocab_file.write(w + '\n')
        vocab_file.close()

    def vectorize(self, text):
        return self.transform([text]).toarray().reshape(len(self.vocabulary_)).astype(numpy.float32)
