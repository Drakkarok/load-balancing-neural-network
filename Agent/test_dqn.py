
import numpy as np
import torch
import pytest
from dqn_model import DQN
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent

def test_dqn_model_shape():
    model = DQN(state_dim=12, action_dim=3)
    # Batch of 4 states
    dummy_input = torch.randn(4, 12)
    output = model(dummy_input)
    assert output.shape == (4, 3)

def test_replay_buffer():
    buffer = ReplayBuffer(capacity=10)
    # Push 5 items
    for i in range(5):
        buffer.push(np.zeros(12), 0, 0.0, np.zeros(12), False)
    
    assert len(buffer) == 5
    
    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(2)
    assert states.shape == (2, 12)
    assert actions.shape == (2,)

def test_agent_optimization():
    # Force CPU for test stability
    agent = DQNAgent(state_dim=12, action_dim=3)
    
    # Needs data to optimize (MIN_REPLAY_SIZE=1000 in agent)
    # Let's override for test
    from dqn_agent import MIN_REPLAY_SIZE
    # We can't change the constant imported by the module easily without reload
    # But checking the code, it uses MIN_REPLAY_SIZE from global.
    # Hack: Populate buffer with >1000 dummy items
    for _ in range(1005):
        agent.memory.push(np.random.rand(12), 0, 1.0, np.random.rand(12), False)
        
    initial_loss = agent.optimize_model()
    # Loss should be a float (and usually non-zero unless initialized perfectly)
    assert isinstance(initial_loss, float)
    
def test_agent_action_selection():
    agent = DQNAgent()
    state = np.random.rand(12)
    action = agent.select_action(state)
    assert action in [0, 1, 2]

if __name__ == "__main__":
    print("Running DQN tests...")
    test_dqn_model_shape()
    test_replay_buffer()
    test_agent_optimization()
    test_agent_action_selection()
    print("All DQN Tests Passed!")
