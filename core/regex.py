import re
import string

SPECIAL_TOKEN = r"@\w*@"

def find_special_token(string_):
    regex = re.compile(SPECIAL_TOKEN, re.IGNORECASE)
    string_ = regex.sub('@special_token@', string_)
    return string_

def replace_one_letter_word(string_):
    for letter in list(string.ascii_lowercase):
        regex = re.compile(r"\b[{}]\b".format(letter), re.IGNORECASE)
        string_ = regex.sub('@{}@'.format(letter), string_)
    return string_

def replace_long_digit_seq(string_):
    regex = re.compile(r"\b\d\d\d\d\d\d\d\d+\b", re.IGNORECASE)
    string_ = regex.sub('@long_digit@', string_)
    return string_

def replace_medium_digit_seq(string_):
    regex = re.compile(r"\b0\d\d\d+\b", re.IGNORECASE)
    string_ = regex.sub('@medium_digit@', string_)
    return string_

def replace_small_digit_seq(string_):
    regex = re.compile(r"\b0\d+\b", re.IGNORECASE)
    string_ = regex.sub('@small_digit@', string_)
    return string_

def replace_number(string_):
    regex = re.compile(r"\b0|[1-9][0-9]+\b", re.IGNORECASE)
    string_ = regex.sub('@number@', string_)
    return string_

if __name__ == '__main__':
    test_string = 'u r c || @test_test@ || 750 || 1234 || 100 || 01234567 || 0123 || 02 || 012 || 0'
    test_string = find_special_token(test_string)
    print(test_string)
    test_string = replace_one_letter_word(test_string)
    print(test_string)
    test_string = replace_long_digit_seq(test_string)
    print(test_string)

    test_string = replace_medium_digit_seq(test_string)
    print(test_string)

    test_string = replace_small_digit_seq(test_string)
    print(test_string)

    test_string = replace_number(test_string)
    print(test_string)