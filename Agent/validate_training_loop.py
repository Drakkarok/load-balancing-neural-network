
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Mock config to speed up things
sys.modules['config'] = MagicMock()
import config
config.STATE_DIM = 12
config.ACTION_DIM = 3
config.BATCH_SIZE = 10
config.PHASE_1_EPISODES = 2
config.PHASE_1_LENGTH = 10
config.PHASE_2_EPISODES = 0 # Skip
config.PHASE_2_LENGTH = 0
config.PHASE_3_EPISODES = 0 # Skip
config.PHASE_3_LENGTH = 0
config.CHECKPOINT_FREQ = 1
config.CHECKPOINT_DIR = "/tmp/lbnn_checkpoints"

# Import after mocking config
from train_dqn import train
from lbnn_env import LBNNEnv

# @patch('lbnn_env.requests')
def test_training_loop():
    print("Testing Training Loop with REAL requests...")
    
    # We don't mock requests anymore.
    # We DO mock config still to keep it short.
    
    
    # Run training
    try:
        train()
        print("Training loop finished successfully!")
    except Exception as e:
        print(f"Training loop failed: {e}")
        raise e

if __name__ == "__main__":
    test_training_loop()
