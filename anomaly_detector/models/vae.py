#!/usr/bin/python3
"""
Pytorch Variational Autoendoder Network Implementation
"""
from itertools import chain
import torch
from torch.autograd import Variable
from torch import nn

from .encoder import Encoder
from .decoder import Decoder


class VAE(nn.Module):
    """
    VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
    """
    def __init__(self, input_dim, latent_dim, device):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)
        self.device = device

        self.mu = None
        self.logvar = None

    def parameters(self):
        return chain(self.encoder.parameters(), self.decoder.parameters())

    def sample_z(self, mu, logvar):
        epsilon = torch.randn(mu.size())
        epsilon = Variable(epsilon, requires_grad=False).type(torch.FloatTensor).to(self.device)
        sigma = torch.exp(logvar / 2)
        return mu + sigma * epsilon

    def forward(self, inputs):
        """
        Forward propagation
        """
        inputs = inputs.unsqueeze(0) if len(inputs.shape) == 1 else inputs
        inputs = inputs.to(self.device)
        self.mu, self.logvar = self.encoder(inputs)
        latent = self.sample_z(self.mu, self.logvar)
        theta = self.decoder(latent)
        return theta

    def save(self, path):
        """
        Save model paramers under config['model_path']
        """
        checkpoint = {'model_state_dict': self.state_dict()}
        torch.save(checkpoint, path)

    def restore_model(self, path):
        """
        Retore the model parameters
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
