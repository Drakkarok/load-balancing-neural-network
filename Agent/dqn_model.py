
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network for LBNN.
    Input: State vector (12 features)
    Output: Q-values for 3 actions (server-1, server-2, server-3)
    """
    def __init__(self, state_dim=12, action_dim=3):
        super(DQN, self).__init__()
        
        # Architecture: 12 -> 64 -> 64 -> 32 -> 3
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_dim)
        
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): State tensor of shape (batch_size, state_dim)
        Returns:
            torch.Tensor: Q-values of shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x) # Linear output (Q-values)
