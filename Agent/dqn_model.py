
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import STATE_DIM, ACTION_DIM, HIDDEN_DIMS

class DQN(nn.Module):
    """
    Deep Q-Network for LBNN.
    Input: State vector (STATE_DIM features)
    Output: Q-values for ACTION_DIM actions (server-1, server-2, server-3)
    Architecture driven by HIDDEN_DIMS in config.py.
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dims=HIDDEN_DIMS):
        super(DQN, self).__init__()

        layer_sizes = [state_dim] + hidden_dims + [action_dim]
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))  # output, no activation

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
