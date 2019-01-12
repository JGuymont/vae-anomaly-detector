#!/usr/bin/python3
"""
Preprocessing module
"""
import re
from string import ascii_lowercase
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
EMAIL_REGEX = r"[\w\.-]+@[\w\.-]+\.\w+"
URL_REGEX = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
LONG_DIGIT_REGEX = r"\b\d\d\d\d\d\d\d\d+\b"
MED_DIGIT_REGEX = r"\b0\d\d\d+\b"
SMALL_DIGIT_REGEX = r"\b0\d+\b"
NUMBER_REGEX = r"\b0|[1-9][0-9]+\b"
LEMMATIZER = WordNetLemmatizer()

class Preprocessor:

    def __init__(self, lemmatize=None):
        self.lemmatize = LEMMATIZER.lemmatize if lemmatize else None

    def _replace_question_mark(self, string):
        return string.replace('?', ' <question_mark> ')

    def _replace_dollar_sign(self, string):
        return string.replace('$', ' <dollar_sign> ')

    def _replace_exclamation_mark(self, string):
        return string.replace('!', ' <exclamation_mark> ')

    def _replace_email(self, string):
        regex = re.compile(EMAIL_REGEX, re.IGNORECASE)
        return regex.sub(' <email_address> ', string)

    def _replace_url(self, string):
        regex = re.compile(URL_REGEX, re.IGNORECASE)
        return regex.sub(' <url> ', string)

    def _replace_one_letter_word(self, string):
        for letter in list(ascii_lowercase):
            regex = re.compile(r"\b[{}]\b".format(letter), re.IGNORECASE)
            string = regex.sub('<{}>'.format(letter), string)
        return string

    def _replace_long_digit_seq(self, string):
        regex = re.compile(LONG_DIGIT_REGEX, re.IGNORECASE)
        return regex.sub('<long_digit>', string)

    def _replace_medium_digit_seq(self, string):
        regex = re.compile(MED_DIGIT_REGEX, re.IGNORECASE)
        return regex.sub('<medium_digit>', string)

    def _replace_small_digit_seq(self, string):
        regex = re.compile(SMALL_DIGIT_REGEX, re.IGNORECASE)
        return regex.sub('<small_digit>', string)

    def _replace_number(self, string):
        regex = re.compile(NUMBER_REGEX, re.IGNORECASE)
        return regex.sub('<number>', string)

    def _remove_punctuation(self, text):
        for punc in punctuation:
            text = text.replace(punc, ' ')
        return text

    def preprocess(self, string):
        """
        Preprocess a string
        """
        string = string.lower()
        string = self._replace_exclamation_mark(string)
        string = self._replace_question_mark(string)
        string = self._replace_dollar_sign(string)
        string = self._replace_email(string)
        string = self._replace_url(string)
        string = self._replace_one_letter_word(string)
        string = self._replace_long_digit_seq(string)
        string = self._replace_medium_digit_seq(string)
        string = self._replace_small_digit_seq(string)
        string = self._replace_number(string)
        string = self._remove_punctuation(string)
        string = string.split()
        if self.lemmatize:
            string = [self.lemmatize(w) for w in string] if self.lemmatize else string
        string = [w for w in string if w not in STOPWORDS]
        return ' '.join(string)
