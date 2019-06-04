from torch import nn


class FullyConnected(nn.Module):

    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_units, activation='relu', dropout_rate=None, batchnormalize=False):
        super(FullyConnected, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.layer_sizes = [input_dim] + [int(hidden_units)] * num_hidden_layers + [output_dim]

        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'tanh':
            self.activation = nn.Tanh

        self.dropout_rate = dropout_rate
        self.batchnormalize = batchnormalize
        self.network = self.build_network()

    def build_network(self):
        network = []

        for i in range(self.num_hidden_layers):
            network.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            network.append(self.activation())
            if self.dropout_rate:
                network.append(nn.Dropout(self.dropout_rate))
            if self.batchnormalize:
                network.append(nn.BatchNorm1d(self.layer_sizes[i+1]))
        network.append(nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]))
        return nn.Sequential(*network)

    def forward(self, x):
        output = x
        for layer in self.network:
            output = layer(output)
        return output