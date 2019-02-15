#!/usr/bin/python3
"""
Split data into a training set and a test set

    data/ train.csv test.csv

where the number after the spam indicate the percentage
of spams in the data

To run this script:
    python split_data.py --train_size 0.5
"""
import argparse
import random
import pandas


def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./data/spam.csv')
    parser.add_argument('--train_size', type=float, default=0.5)
    parser.add_argument('--out_dir', type=str, default='./data')
    return parser.parse_args()


def load_data(path):
    """
    Load data into a pandas dataframe
    """
    dataframe = pandas.read_table(path, delimiter=',', encoding='latin-1')
    return dataframe[['v1', 'v2']]


def _split_indices(data, split):
    storage = {'train': [], 'test': []}
    data_size = len(data)
    train_size = round(data_size * split[0])
    examples = range(len(data))
    storage['train'] = random.sample(examples, train_size)
    storage['test'] = [ex for ex in examples if ex not in storage['train']]
    return storage


def split_data(dataframe, split):
    """
    Split the data into a training set
    and a test set according to 'train_size'

    Args:
        dataframe: (pandas.Dataframe)
        split: (list of float) train/valid/test split
    """
    split_idx = _split_indices(dataframe, split)
    train_data = dataframe.iloc[split_idx['train']]
    test_data = dataframe.iloc[split_idx['test']]
    return train_data, test_data


def save_data(dataframe, name, args):
    """
    Save a dictionary data to a pickle file
    """
    out_dir = '{}/{}.csv'.format(
        args.out_dir, name)
    dataframe.to_csv(out_dir, index=False)


def main(args):
    """
    wrapper - create the data<spam_percentage>.csv file
    """
    split = [args.train_size, 1 - args.train_size]
    data = load_data(args.csv_path)
    train_data, test_data = split_data(data, split)

    save_data(train_data, 'train', args)
    save_data(test_data, 'test', args)


if __name__ == '__main__':
    main(argparser())
