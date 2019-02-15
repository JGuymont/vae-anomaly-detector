"""
Main experiment
"""
import json
import argparse
from core.data import Data
from core.vae import VAE
from core import utils
from core.data import DataLoader
from utils.data import SpamDataset
from torchtext.data import TabularDataset, Field, Iterator


MODELS = ['bow', 'binary_bow', 'boc']


def argparser():
    """
    Command line argument parser
    """

    parser = argparse.ArgumentParser(description='VAE spam detector')

    parser.add_argument('--model', '-m', type=str, choices=['bow', 'binary_bow', 'boc', 'all'])
    parser.add_argument('--config', '-c', type=str, default='01')
    parser.add_argument('--n_epochs', '-e', type=int, default=None)
    parser.add_argument('--restore', type=int, default=None)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--task', '-t', type=str, default='train', choices=['train', 'eval', 'plot'])
    parser.add_argument('--lr', type=float, default=None)

    return parser.parse_args()


def load_config(args):
    """
    Load json configuration file
    """
    path = './configs/{}/config{}.json'.format(args.model, args.config)
    config = json.load(open(path, 'r'))
    if args.n_epochs is not None:
        config['training']['n_epochs'] = args.n_epochs
    if args.lr is not None:
        config['training']['lr'] = args.lr
    return config


def main(args):
    """
    Lunch the expriment
    """
    if args.task == 'train':
        if args.model == 'all':
            for model in MODELS:
                args.model = model
                config = load_config(args)
                data = Data(config['data'])
                data.vectorize()
                vae = VAE(data.input_dim_, config, args.device).to(args.device)
                if args.restore is not None:
                    vae.restore_model(args.restore)
                vae.fit(data.train, cur_epoch=args.restore)
            return
        config = load_config(args)
        data = Data(config['data'])
        data.vectorize()
        print(data.data_size_)
        exit()
        vae = VAE(data.input_dim_, config, args.device).to(args.device)
        if args.restore is not None:
            vae.restore_model(args.restore)
        vae.fit(data.train, cur_epoch=args.restore)
    if args.task == 'plot':
        models = ['{}{}'.format(model, args.config) for model in ['bow', 'binarybow', 'boc']]
        precision, recall, logp, kldiv, log_densities, params = utils.load_results(args.config)
        utils.plot_precision(precision, models, args.config)
        utils.plot_recall(recall, models, args.config)
        utils.plot_logp(logp, models, args.config)
        utils.plot_kldiv(kldiv, models, args.config)
        for model in models:
            utils.hist_densities(log_densities[model], model)
            utils.hist_param(params[model].reshape(-1), model)
    if args.task == 'eval':
        config = load_config(args)
        data = Data(config['data'])
        data.vectorize()
        vae = VAE(data.input_dim_, config, args.device).to(args.device)
        if args.restore is not None:
            vae.restore_model(args.restore)
        vae.eval()
        testloader = DataLoader(data.test, batch_size=len(data.test))
        precisions, recalls, logliks, kldivs = [], [], [], []
        for _ in range(100):
            _, _, precision, recall = vae.evaluate(testloader)
            loglik, kldiv = vae.evaluate_loss(testloader)
            precisions.append(precision)
            recalls.append(recall)
            logliks.append(loglik)
            kldivs.append(kldiv)
        print(utils.mean_confidence_interval(precisions))
        print(utils.mean_confidence_interval(recalls))
        print(utils.mean_confidence_interval(logliks))
        print(utils.mean_confidence_interval(kldivs))

def tokenizer(string):
    return [char for char in string]

if __name__ == '__main__':
    args = argparser()
    config = load_config(args)

    data = SpamDataset(csv_path=config['data']['csv_path'])

    for x, y in data:
        print(x, y)
        print(tokenizer(x))
        exit()
