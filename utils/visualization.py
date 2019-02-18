#!/usr/bin/python3
"""
Utility functions for visualization
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def load_results(models):
    """
    Load results corresponging to config 01
    """
    precision, recall, logp, kldiv, log_densities, params = {}, {}, {}, {}, {}, {}
    for model in models:
        path_results = './results/{}.pkl'.format(model)
        results = load_pickle(path_results)
        precision[model] = results['precision']
        recall[model] = results['recall']
        logp[model] = results['-logp(x|z)']
        kldiv[model] = results['kldiv']
        log_densities[model] = results['log_densities']
        params[model] = results['params']
    return precision, recall, logp, kldiv, log_densities, params


def plot_precision(precisions, models):
    """
    Plot the results of the model, i.e. the precision of each method
    """
    # x = np.arange(len(precisions[models[0]]))
    for model in models:
        plt.plot(precisions[model])
    plt.ylim(0., 1.)
    plt.legend(models)
    plt.xlabel('Iterations')
    plt.ylabel('Precision')
    plt.savefig('./figures/precisions.png')
    plt.close()


def plot_recall(recalls, models):
    """
    Plot the results of the model, i.e. the precision of each method
    """
    for model in models:
        plt.plot(recalls[model])
    plt.ylim(0., 1.)
    plt.legend(models)
    plt.xlabel('Iterations')
    plt.ylabel('Recall')
    plt.savefig('./figures/recall.png')
    plt.close()


def plot_logp(logp, models):
    """
    Plot the results of the model, i.e. the precision of each method
    """
    for model in models:
        plt.plot(logp[model])
    plt.legend(models)
    plt.xlabel('Iterations')
    plt.ylabel('-logp(x|z)')
    plt.savefig('./figures/logp.png')
    plt.close()


def plot_kldiv(kldiv, models):
    """
    Plot the results of the model, i.e. the precision of each method
    """
    for model in models:
        plt.plot(kldiv[model])
    plt.legend(models)
    plt.xlabel('Iterations')
    plt.ylabel('KL-divergence')
    plt.savefig('./figures/kldiv.png')
    plt.close()


def hist_densities(log_densities, model):
    plt.hist(log_densities)
    plt.savefig('./figures/log_densities_{}.png'.format(model))
    plt.close()


def hist_param(params, model):
    plt.hist(params)
    plt.savefig('./figures/hist_param_{}.png'.format(model))
    plt.close()


def load_pickle(path):
    """
    Load a dictionary containing the precision of each models
    """
    with open(path, 'rb') as pkl:
        results = pickle.load(pkl)
    return results


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, '+-', h
