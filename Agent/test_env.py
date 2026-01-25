
import pytest
import numpy as np
from lbnn_env import LBNNEnv
from unittest.mock import patch, MagicMock

# Mocking requests because we can't run the actual server in this test environment
# In a real scenario, we would run the docker containers and test against them.
# Here we verify the logic of the Env class.

@patch('lbnn_env.requests')
def test_env_initialization(mock_requests):
    env = LBNNEnv()
    assert env.action_space.n == 3
    assert env.observation_space.shape == (12,)

@patch('lbnn_env.requests')
def test_reset(mock_requests):
    # Setup mock responses
    mock_post = MagicMock()
    mock_requests.post.return_value = mock_post
    
    mock_get = MagicMock()
    mock_requests.get.return_value = mock_get
    
    # Mock server states response
    mock_get.json.return_value = {
        "server-1": {"cpu": 10.0, "memory": 20.0, "connections": 1},
        "server-2": {"cpu": 0.0, "memory": 0.0, "connections": 0},
        "server-3": {"cpu": 50.0, "memory": 40.0, "connections": 5}
    }
    
    env = LBNNEnv()
    obs, info = env.reset()
    
    # Check if reset called on agent
    mock_requests.post.assert_called_with("http://localhost:8080/reset_episode")
    
    # Check observation shape and content
    assert obs.shape == (12,)
    # Just check one value (server-1 cpu normalized: 10/100 = 0.1)
    # Actually code divides by 100.0
    assert np.isclose(obs[0], 0.1)
    
@patch('lbnn_env.requests')
def test_step(mock_requests):
    env = LBNNEnv()
    
    # Needs a current request to step
    env.current_request = {"cpu": 50, "memory": 30, "duration": 2}
    
    # Mock Step Response from Agent
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "processed",
        "current_server_states": {
            "server-1": {"cpu": 15.0, "memory": 25.0, "connections": 2}, # Increased load
            "server-2": {"cpu": 0.0, "memory": 0.0, "connections": 0},
            "server-3": {"cpu": 0.0, "memory": 0.0, "connections": 0}
        }
    }
    mock_requests.post.return_value = mock_response
    
    # Also need GET for reset inside, but we skip reset here
    
    obs, reward, done, truncated, info = env.step(0) # Action 0 = server-1
    
    # Verify post called with correct payload
    call_args = mock_requests.post.call_args[1]['json']
    assert call_args['forced_action'] == 0
    assert call_args['request'] == {"cpu": 50, "memory": 30, "duration": 2}
    
    # Verify Reward Calculation
    # Chosen server-1: Load = 15+25 = 40
    # Others: server-2 (0), server-3 (0). Avg = 0.
    # Reward = 0 - 40 = -40. (This is expected as server-1 took load and others are empty)
    assert np.isclose(reward, -40.0)
    
    # Verify Done condition
    env.current_step = env.episode_length
    obs, reward, done, truncated, info = env.step(0)
    assert done is True

if __name__ == "__main__":
    # verification script
    print("Running tests...")
    try:
        test_env_initialization()
        test_reset()
        test_step()
        print("All Tests Passed!")
    except Exception as e:
        print(f"Tests Failed: {e}")
        import traceback
        traceback.print_exc()
