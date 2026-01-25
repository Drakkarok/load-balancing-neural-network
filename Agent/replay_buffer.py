
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    Stores transitions (state, action, reward, next_state, done).
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition"""
        # Expand state dims if needed or store as is.
        # Storing as numpy arrays is best.
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        
        # Transpose batch: [(s,a,r,ns,d), ...] -> (s_batch, a_batch, ...)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.bool_)
        )
    
    def __len__(self):
        return len(self.buffer)
