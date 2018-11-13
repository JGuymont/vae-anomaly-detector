from itertools import chain
import numpy as np

import time
import torch
from torch.autograd import Variable
from torch import nn


class Encoder(nn.Module):
    """Probabilistic Encoder

    Return the mean and the variance of z ~ q(z|x). The prior
    of x is assume to be normal(0, I). 
    
    Args
        x_dim:
        h_dim:
        z_dim:

    Return: (mu_z, log_var_z)
    """

    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()
        self._main_network = nn.Sequential(
            nn.Linear(x_dim, h_dim),    
            nn.ReLU()
        )
        self._mu = nn.Linear(h_dim, z_dim)
        self._log_var = nn.Linear(h_dim, z_dim)
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Parameter initialization

            W ~ U[-1 / nc, 1/ nc]
        
        where nc is the number of neuros on the 
        previous layer. The bias are initialize at
        0.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1/np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, inputs):
        """Forward propagation"""
        hidden_state = self._main_network(inputs)
        mu = self._mu(hidden_state)
        log_var = self._log_var(hidden_state) 
        return mu, log_var


class Decoder(nn.Module):
    """Decoder Decoder
    """
    def __init__(self, x_dim, h_dim, z_dim):
        super(Decoder, self).__init__()
        self._decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )
        self._initialize_parameters()

    def _initialize_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nc = layer.in_features
                stdv = 1/np.sqrt(nc)
                layer.weight.data.uniform_(-stdv, stdv)
                layer.bias.data.zero_()

    def forward(self, z):
        x = self._decoder(z)
        return x


class VAE(nn.Module):
    """
    VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
    """

    _EPS = 1e-10

    def __init__(self, x_dim, h_dim, z_dim, device, eps=_EPS):
        super(VAE, self).__init__()
        self._eps = eps
        self._device = device
        self._encoder = Encoder(x_dim, h_dim, z_dim)
        self._decoder = Decoder(x_dim, h_dim, z_dim)
        self._optim = torch.optim.Adam(
            self.parameters(), 
            lr=1.e-3, 
            betas=(0.5, 0.999)
        )
        self._bce_loss = torch.nn.BCELoss(reduction='sum')
        self._num_epochs = 100
        
        self.mu = None 
        self.log_var = None

    def parameters(self):
        return chain(self._encoder.parameters(), self._decoder.parameters())

    def _sample_z(self, mu, log_var):
        epsilon = torch.randn(mu.size())
        epsilon = Variable(epsilon, requires_grad=False).type(torch.FloatTensor).to(self._device)
        sigma = torch.exp(log_var / 2)
        return mu + sigma * epsilon

    def forward(self, inputs):
        self.mu, self.log_var = self._encoder(inputs)
        z = self._sample_z(self.mu, self.log_var)
        x = self._decoder(z)
        return x

    def train(self, dataloader):

        to_np = lambda x: x.data.cpu().numpy()

        for epoch in range(self._num_epochs):
            if epoch >= 1:
                print("\n[%2.2f]" % (time.time() - t0), end="\n")
            
            t0 = time.time()

            for step, (inputs, _) in enumerate(dataloader, 0):
                if step >= 1:
                    print("Epoch [%i] | iter [%i] | time [%2.2f] | loss [%2.2f]" % (
                        epoch, step, time.time() - t1, to_np(loss)), end="\r"
                    )
                t1 = time.time()

                batch_size = inputs.size(0)

                x = inputs.to(self._device)
                gamma = self.forward(x)

                log_likelihood = self._bce_loss(gamma+self._eps, x) / batch_size
                kl_divirgence = torch.mean(.5 * torch.sum(self.mu**2 + torch.exp(self.log_var) - 1 - self.log_var, 1))
                loss = log_likelihood + kl_divirgence
                
                loss.backward()

                self._optim.step()
                self._optim.zero_grad() 

if __name__ == '__main__':
    
    pass