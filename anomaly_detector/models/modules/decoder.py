import json
import numpy as np
from torch import nn


class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, input_dim, latent_dim):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, input_dim),
            nn.Tanh()
        )

        self.read_alpha = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
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
        alpha = 0.5 * self.read_alpha(z)
        return alpha * self.decoder(z)
