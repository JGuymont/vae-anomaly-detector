"""Tests for `core.vae.py`."""
import unittest

import torch
from torch.autograd import Variable

from context import core
from core.vae import Encoder, Decoder, VAE

class VAETestCase(unittest.TestCase):
    """Tests for `core.vae.py`."""

    def __init__(self, *args, **kwargs):    
        unittest.TestCase.__init__(self, *args, **kwargs)

        self._BATCH_SIZE = 2
        self.X_DIM = 5
        self.H_DIM = 4 
        self.Z_DIM = 2

        self._device = 'cuda'

        self._input = self._get_syntetic_input(shape=(1, self.X_DIM))
        self._batch_x = self._get_syntetic_input(shape=(self._BATCH_SIZE, self.X_DIM))

        self._encoder = Encoder(x_dim=5, h_dim=4, z_dim=2).to(self._device)
        self._decoder = Decoder(x_dim=5, h_dim=4, z_dim=2).to(self._device)

    def _get_syntetic_input(self, shape):
        """Random normal N(0, 1) input"""
        return Variable(torch.randn(shape)).to(self._device)

    def _get_syntetic_target(self, shape):
        """Random normal N(0, 1) input"""
        p = torch.empty(shape).uniform_(0, 1)
        #return Variable(torch.bernoulli(p).type(torch.LongTensor))
        #return Variable(torch.bernoulli(p).type(torch.ShortTensor))
        return Variable(torch.bernoulli(p)).to(self._device)

    def test_mini_batch_encoder(self):
        """Test encoder when input one example at a time"""
        mu, log_var = self._encoder(self._input)
        latent_dim = (self._input.size(0), self.Z_DIM)
        self.assertTrue(mu.size() == latent_dim and log_var.size() == latent_dim)

    def test_batch_encoder(self):
        """Test encoder when input _BATCH_SIZE example at a time"""
        mu, log_var = self._encoder(self._batch_x)
        latent_dim = (self._BATCH_SIZE, self.Z_DIM)
        self.assertTrue(mu.size() == latent_dim and log_var.size() == latent_dim)

    def _sample_z(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False).type(torch.FloatTensor).to(self._device)
        sigma = torch.exp(log_var / 2)
        return mu + sigma * epsilon

    def test_mini_batch_decoder(self):
        """Test encoder when input one example at a time"""
        mu, log_var = self._encoder(self._input)
        z = self._sample_z(mu, log_var)
        x = self._decoder(z)
        self.assertTrue(x.size() == self._input.size())

    def test_batch_decoder(self):
        """Test encoder when input _BATCH_SIZE example at a time"""
        mu, log_var = self._encoder(self._batch_x)
        z = self._sample_z(mu, log_var)
        x = self._decoder(z)
        self.assertTrue(x.size() == self._batch_x.size())

if __name__ == '__main__':
    unittest.main()
