
import torch
import torch.optim as optim
import numpy as np
import random
from dqn_model import DQN
from replay_buffer import ReplayBuffer
import config

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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # Replay Buffer
        self.memory = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        
        # Exploration
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY # Allow runtime modification
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
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * self.epsilon_decay)
        
    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < config.MIN_REPLAY_SIZE:
            return 0.0 # Not enough data
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(config.BATCH_SIZE)
        
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
        expected_state_action_values = reward_batch + (config.GAMMA * next_state_values * (1 - done_batch))
        
        # Compute Loss (MSE or Huber)
        criterion = torch.nn.MSELoss()
        loss = criterion(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), config.GRAD_CLIP_VALUE)
        self.optimizer.step()
        
        # Soft Update Target Network
        # theta_target = tau * theta_policy + (1 - tau) * theta_target
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*config.TAU + target_net_state_dict[key]*(1-config.TAU)
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
        self.epsilon = checkpoint.get('epsilon', config.EPSILON_START)

