
# Hyperparameters and Constants

# Network
STATE_DIM = 12
ACTION_DIM = 3
HIDDEN_DIMS = [64, 64, 32]

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.005

# Experience Replay
REPLAY_BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000

# Exploration
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Curriculum (Episodes)
PHASE_1_EPISODES = 1000
PHASE_1_LENGTH = 150
PHASE_2_EPISODES = 2000
PHASE_2_LENGTH = 250
PHASE_3_EPISODES = 3000
PHASE_3_LENGTH = 500

# Checkpointing
CHECKPOINT_FREQ = 100
CHECKPOINT_DIR = "Models/checkpoints"

# System Constants
AGENT_URL = "http://localhost:8080"
SERVER_CAPACITIES = {
    "server-1": {"cpu": 1500, "memory": 2000},
    "server-2": {"cpu": 3500, "memory": 3200},
    "server-3": {"cpu": 5000, "memory": 3000}
}
