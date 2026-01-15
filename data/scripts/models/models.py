import torch
import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        layers = []
        in_features = input_dim

        # num_layers fully connected hidden layers, each with 'hidden_size' neurons
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_size

        # output layer
        layers.append(nn.Linear(hidden_size, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (B, 3, 32, 32) -> flatten to (B, 3072)
        x = x.view(x.size(0), -1)
        return self.net(x)

class FCNetBinary(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super().__init__()
        layers = []
        in_features = input_dim

        # num_layers fully connected hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_size

        # output layer: single neuron for binary classification
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze()

