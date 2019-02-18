#!/usr/bin/python3
"""
Pytorch Variational Autoendoder Network Implementation
"""
from itertools import chain
import time
import json
import pickle
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class Encoder(nn.Module):
    """
    Probabilistic Encoder

    Return the mean and the variance of z ~ q(z|x). The prior
    of x is assume to be normal(0, I).

    Arguments:
        input_dim {int} -- number of features

    Returns:
        (tensor, tensor) -- mean and variance of the latent variable
            output from the forward propagation
    """
    def __init__(self, input_dim, config):
        super(Encoder, self).__init__()

        config_encoder = json.loads(config.get("encoder"))
        config_read_mu = json.loads(config.get("read_mu"))
        config_read_logvar = json.loads(config.get("read_sigma"))

        config_encoder[0]['in_features'] = input_dim

        encoder_network = []
        for layer in config_encoder:
            if layer['type'] == 'linear':
                encoder_network.append(nn.Linear(layer['in_features'], layer['out_features']))
            elif layer['type'] == 'relu':
                encoder_network.append(nn.ReLU())
            elif layer['type'] == 'tanh':
                encoder_network.append(nn.Tanh())
            elif layer['type'] == 'dropout':
                encoder_network.append(nn.Dropout(layer['rate']))
            elif layer['type'] == 'batch_norm':
                encoder_network.append(nn.BatchNorm1d(layer['num_features']))

        self.encoder_network = nn.Sequential(*encoder_network)
        self.read_mu = nn.Linear(config_read_mu['in_features'], config.getint('latent_dim'))
        self.read_logvar = nn.Linear(config_read_logvar['in_features'], config.getint('latent_dim'))
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Xavier initialization
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, inputs):
        """
        Forward propagation
        """
        hidden_state = self.encoder_network(inputs)
        mean = self.read_mu(hidden_state)
        logvar = self.read_logvar(hidden_state)
        return mean, logvar


class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, input_dim, config):
        super(Decoder, self).__init__()
        config_decoder = json.loads(config.get("decoder"))
        self._distr = config['distribution']

        decoder_network = []
        for layer in config_decoder:
            if layer['type'] == 'linear':
                decoder_network.append(nn.Linear(layer['in_features'], layer['out_features']))
            elif layer['type'] == 'relu':
                decoder_network.append(nn.ReLU())
            elif layer['type'] == 'relu6':
                decoder_network.append(nn.ReLU6())
            elif layer['type'] == 'tanh':
                decoder_network.append(nn.Tanh())
            elif layer['type'] == 'sigmoid':
                decoder_network.append(nn.Sigmoid())
            elif layer['type'] == 'dropout':
                decoder_network.append(nn.Dropout(layer['rate']))
            elif layer['type'] == 'batch_norm':
                decoder_network.append(nn.BatchNorm1d(layer['num_features']))
            elif layer['type'] == 'read_x':
                decoder_network.append(nn.Linear(layer['in_features'], input_dim))
        self.decoder = nn.Sequential(*decoder_network)
        if self._distr == 'poisson':
            self.read_alpha = nn.Sequential(
                nn.Linear(config.getint('latent_dim'), input_dim),
                nn.ReLU6()
            )
        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, z):
        if self._distr == 'poisson':
            alpha = 0.5 * self.read_alpha(z)
            return alpha * self.decoder(z)
        else:
            return self.decoder(z)


class VAE(nn.Module):
    """
    VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
    """
    def __init__(self, input_dim, config):
        super(VAE, self).__init__()
        self.config = config
        self._model_name = config['model']['name']
        self._distr = config['model']['distribution']
        self._device = config['model']['device']
        self._encoder = Encoder(input_dim, config['model'])
        self._decoder = Decoder(input_dim, config['model'])

        self.num_epochs = config.getint('training', 'n_epochs')

        self._optim = optim.Adam(
            self.parameters(), 
            lr=config.getfloat('training', 'lr'),
            betas=json.loads(config['training']['betas'])
        )

        self.mu = None
        self.logvar = None

        self.precentile_threshold = config.getfloat('model', 'threshold')
        self.threshold = None

        self._save_every = config.getint('model', 'save_every')

    def parameters(self):
        return chain(self._encoder.parameters(), self._decoder.parameters())

    def _sample_z(self, mu, logvar):
        epsilon = torch.randn(mu.size())
        epsilon = Variable(epsilon, requires_grad=False).type(torch.FloatTensor).to(self._device)
        sigma = torch.exp(logvar / 2)
        return mu + sigma * epsilon

    def forward(self, inputs):
        """
        Forward propagation
        """
        self.mu, self.logvar = self._encoder(inputs)
        latent = self._sample_z(self.mu, self.logvar)
        theta = self._decoder(latent)
        return theta

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()

    def poisson_cross_entropy(self, logtheta, inputs):
        return - inputs * logtheta + torch.exp(logtheta)

    def loglikelihood(self, reduction):
        """
        Return the log-likelihood
        """
        if self._distr == 'poisson':
            if reduction == 'none':
                return self.poisson_cross_entropy
            return nn.PoissonNLLLoss(reduction=reduction)
        elif self._distr == 'bernoulli':
            return nn.BCELoss(reduction=reduction)
        else:
            raise ValueError('{} is not a valid distribution'.format(self._distr))

    def fit(self, trainloader, cur_epoch=None, print_every=1):
        """
        Train the neural network
        """
        self.cur_epoch = 0 if cur_epoch is None else cur_epoch

        start_time = time.time()

        storage = {
            'loss': [], 'kldiv': [], '-logp(x|z)': [],
            'precision': [], 'recall': [], 'log_densities': None, 'params': None
        }

        for epoch in range(self.cur_epoch, self.cur_epoch + self.num_epochs):

            self.cur_epoch += 1

            # temporary storage
            losses, kldivs, neglogliks = [], [], []

            for inputs, _ in trainloader:
                inputs = inputs.to(self._device)
                logtheta = self.forward(inputs)
                loglikelihood = -self.loglikelihood(reduction='sum')(logtheta, inputs) / inputs.shape[0]
                kl_div = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) / inputs.shape[0]
                loss = -loglikelihood + kl_div
                loss.backward()
                self._optim.step()
                self._optim.zero_grad()
                losses.append(self._to_numpy(loss))
                kldivs.append(self._to_numpy(kl_div))
                neglogliks.append(self._to_numpy(-loglikelihood))

            storage['loss'].append(np.mean(losses))
            storage['kldiv'].append(np.mean(kldivs))
            storage['-logp(x|z)'].append(np.mean(neglogliks))

            if (epoch + 1) % print_every == 0:
                epoch_time = self._get_time(start_time, time.time())
                self.eval()
                f1, acc, prec, recall = self.evaluate(trainloader)
                storage['precision'].append(prec)
                storage['recall'].append(recall)
                print('epoch: {} | loss: {:.3f} | -logp(x|z): {:.3f} | kldiv: {:.3f} | time: {}'.format(
                    epoch + 1,
                    storage['loss'][-1],
                    storage['-logp(x|z)'][-1],
                    storage['kldiv'][-1],
                    epoch_time))
                print('F1. {:.3f} | acc. {:.3f} | prec.: {:.3f} | rec. {:.3f}'.format(f1, acc, prec, recall))
                self.train()
            
            if (epoch + 1) % self._save_every == 0:
                self.save_checkpoint()

        storage['log_densities'] = self._get_densities(trainloader)
        storage['params'] = self._get_parameters(trainloader)
        with open('./results/{}.pkl'.format(self._model_name), 'wb') as _f:
            pickle.dump(storage, _f, pickle.HIGHEST_PROTOCOL)

    def _remove_spam(self, dataloader, data):
        idx_to_remove = self._find_threshold(dataloader)
        data.pop(idx_to_remove)
        self._encoder.initialize_parameters()
        self._decoder.initialize_parameters()
        self._optim = optim.Adam(self.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return data

    def _get_time(self, starting_time, current_time):
        total_time = current_time - starting_time
        minutes = round(total_time // 60)
        seconds = round(total_time % 60)
        return '{} min., {} sec.'.format(minutes, seconds)

    def _get_parameters(self, dataloader):
        self.eval()
        parameters = []
        for inputs, _ in dataloader:
            inputs = inputs.to(self._device)
            logtheta = self._to_numpy(self.forward(inputs))
            parameters.extend(logtheta)
        self.train()
        if self._distr == 'poisson':
            parameters = np.exp(np.array(parameters))
        else:
            parameters = np.array(parameters)
        return parameters

    def _evaluate_probability(self, inputs):
        with torch.no_grad():
            inputs = inputs.to(self._device)
            logtheta = self.forward(inputs)
            log_likelihood = -self.loglikelihood(reduction='none')(logtheta, inputs)
            log_likelihood = torch.sum(log_likelihood, 1)
            assert inputs.shape[0] == log_likelihood.shape[0]
            return self._to_numpy(log_likelihood)

    def _get_densities(self, dataloader):
        self.eval()
        all_log_densities = []
        for inputs, _ in dataloader:
            mini_batch_log_densities = self._evaluate_probability(inputs)
            all_log_densities.extend(mini_batch_log_densities)
        self.train()
        all_log_densities = np.array(all_log_densities)
        return all_log_densities

    def _find_threshold(self, dataloader):
        log_densities = self._get_densities(dataloader)
        lowest_density = np.argmin(log_densities)
        self.threshold = np.percentile(log_densities, self.precentile_threshold)
        return lowest_density

    def predict(self, inputs):
        """
        Predict the class of the inputs
        """
        log_density = self._evaluate_probability(inputs)
        predictions = np.zeros_like(log_density).astype(int)
        predictions[log_density < self.threshold] = 1
        return list(predictions)

    def evaluate(self, dataloader):
        """
        Evaluate accuracy.
        """
        self.eval()
        self._find_threshold(dataloader)
        predictions = []
        ground_truth = []

        for inputs, targets in dataloader:
            pred = self.predict(inputs)
            predictions.extend(pred)
            ground_truth.extend(list(self._to_numpy(targets)))
        self.train()

        f1 = f1_score(ground_truth, predictions)
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions)
        recall = recall_score(ground_truth, predictions)

        return f1, accuracy, precision, recall

    def evaluate_loss(self, dataloader):
        loglikelihood, kl_div = 0., 0. 
        for inputs, _ in dataloader:
            inputs = inputs.to(self._device)
            logtheta = self.forward(inputs)
            loglikelihood += - self._to_numpy(self.loglikelihood(reduction='sum')(logtheta, inputs)) / inputs.shape[0]
            kl_div += -0.5 * self._to_numpy(torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())) / inputs.shape[0]
        return loglikelihood, kl_div

    def save_checkpoint(self):
        """Save model paramers under config['model_path']"""
        model_path = '{}{}{}_{}.pt'.format(
            self.config['paths']['checkpoints_directory'],
            self.config['model']['name'],
            self.config['model']['config_id'],
            self.cur_epoch)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optim.state_dict()
        }
        torch.save(checkpoint, model_path)

    def restore_model(self, epoch):
        """
        Retore the model parameters
        """
        model_path = './{}/{}{}_{}.pt'.format(
            self.config['paths']['checkpoints_directory'],
            self.config['model']['name'],
            self.config['model']['config_id'],
            epoch)
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])
