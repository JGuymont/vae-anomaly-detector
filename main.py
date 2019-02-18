"""
Main experiment
"""
import json
import os
import argparse
import torch
from torch.utils.data import DataLoader
from configparser import ConfigParser

from core.vae import VAE
from utils.data import SpamDataset
from utils.feature_extractor import FeatureExtractor
from core import utils
from constants import MODELS


def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='VAE spam detector')
    parser.add_argument('--globals', type=str, default='./configs/globals.ini')
    parser.add_argument('--model_common', type=str)
    parser.add_argument('--model_specific', type=str)
    parser.add_argument('--n_epochs', '-e', type=int, default=None)
    parser.add_argument('--restore', type=int, default=None)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--task', '-t', type=str, default='train', choices=['train', 'eval', 'plot'])
    return parser.parse_args()


def load_config(args):
    """
    Load .INI configuration files
    """
    config = ConfigParser()
    config.read(args.globals)
    config.read(args.model_common)
    config.read(args.model_specific)
    config.set('model', 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if args.n_epochs is not None:
        config.set('training', 'n_epochs', str(args.n_epochs))
    return config


def train(config, trainloader, devloader=None):
    input_dim = trainloader.dataset.input_dim_
    vae = VAE(input_dim, config).to(config['model']['device'])
    vae.fit(trainloader)


if __name__ == '__main__':
    args = argparser()
    config = load_config(args)

    # Get data path
    data_dir = config.get("paths", "data_directory")
    train_data_file_name = config.get("paths", "train_data_file_name")
    train_csv_path = os.path.join(data_dir, train_data_file_name)

    # Set text processing function
    transformer = FeatureExtractor(config)
    raw_documents = transformer.get_raw_documents(train_csv_path)
    transformer.fit(raw_documents)
    transformer.log_vocabulary('data/vocab.txt')

    train_data = SpamDataset(
        train_csv_path,
        label2int=json.loads(config.get("data", "label2int")),
        transform=transformer.vectorize)

    trainloader = DataLoader(
        train_data,
        batch_size=config.getint("training", "batch_size"),
        shuffle=True,
        num_workers=0,
        pin_memory=False)

    checkpoints_directory = os.path.join(
        config['paths']['checkpoints_directory'],
        '{}{}'.format(config['model']['name'], config['model']['config_id'])
    )
    os.makedirs(checkpoints_directory, exist_ok=True)

    train(config, trainloader)
