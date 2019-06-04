import numpy as np
from torch import nn


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

    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()

        self.encoder_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(64)
        )
        self.read_mu = nn.Linear(64, latent_dim)
        self.read_logvar = nn.Linear(64, latent_dim)
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
