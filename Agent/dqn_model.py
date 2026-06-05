
import torch.nn as nn

from config import STATE_DIM, ACTION_DIM, HIDDEN_DIMS

class DQN(nn.Module):
    """
    Dueling DQN for LBNN.
    Shared trunk -> value head (1) + advantage head (action_dim) -> Q-values.
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dims=HIDDEN_DIMS):
        super(DQN, self).__init__()
        self.action_dim = action_dim

        trunk_layers = []
        in_dim = state_dim
        for h in hidden_dims:
            trunk_layers.append(nn.Linear(in_dim, h))
            trunk_layers.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        self.value_head = nn.Linear(in_dim, 1)
        self.advantage_head = nn.Linear(in_dim, action_dim)

    def forward(self, x):
        features = self.trunk(x)
        V = self.value_head(features)                          # [batch, 1]
        adv = self.advantage_head(features)                    # [batch, action_dim]
        Q = V + (adv - adv.mean(dim=1, keepdim=True))         # [batch, action_dim]
        return Q
