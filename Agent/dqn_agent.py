
import torch
import torch.optim as optim
import numpy as np
import random
from dqn_model import DQN
from replay_buffer import ReplayBuffer

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TAU = 0.005
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_dim=12, action_dim=3):
        self.action_dim = action_dim
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Replay Buffer
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Exploration
        self.epsilon = EPSILON_START
        self.steps_done = 0
        
    def select_action(self, state, eval_mode=False):
        """
        Select action using Epsilon-Greedy policy.
        Args:
            state (np.array): Current state vector
            eval_mode (bool): If True, use greedy policy (no exploration)
        """
        if eval_mode:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        
        # Epsilon-Greedy
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randint(0, self.action_dim - 1)
            
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < MIN_REPLAY_SIZE:
            return 0.0 # Not enough data
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(states).to(device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        next_state_batch = torch.FloatTensor(next_states).to(device)
        done_batch = torch.FloatTensor(dones).to(device)
        
        # Compute Q(s, a)
        # policy_net(state_batch) -> [batch, 3]. gather(1, action_batch) -> [batch, 1]
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s') for next states using Target Net
        # We want max_a Q_target(s', a)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            
        # Compute expected Q values: reward + gamma * max_a Q(s', a) * (1 - done)
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
        
        # Compute Loss (MSE or Huber)
        criterion = torch.nn.MSELoss()
        loss = criterion(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Soft Update Target Network
        # theta_target = tau * theta_policy + (1 - tau) * theta_target
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        
        return loss.item()

    def save_checkpoint(self, filepath):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', EPSILON_START)
