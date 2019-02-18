import argparse

from utils import visualization
from constants import MODELS


def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--boc', type=str, default='00')
    parser.add_argument('--bow', type=str, default='00')
    parser.add_argument('--binary_bow', type=str, default='00')
    return parser.parse_args()


def make_plots(args):
    configs = [args.bow, args.binary_bow, args.boc]
    models = ['{}{}'.format(model, config) for (model, config) in zip(MODELS, configs)]
    precision, recall, logp, kldiv, log_densities, params = visualization.load_results(models)
    visualization.plot_precision(precision, models)
    visualization.plot_recall(recall, models)
    visualization.plot_logp(logp, models)
    visualization.plot_kldiv(kldiv, models)
    for model in models:
        visualization.hist_densities(log_densities[model], model)
        visualization.hist_param(params[model].reshape(-1), model)


if __name__ == '__main__':
    make_plots(argparser())
