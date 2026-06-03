
# Hyperparameters and Constants

# Network
STATE_DIM = 66
ACTION_DIM = 3
HIDDEN_DIMS = [128, 128, 64]

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.005
GRAD_CLIP_VALUE = 1.0

# Experience Replay
REPLAY_BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000

# Exploration
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# EMA warmup: steps run at episode start before transitions enter the replay buffer.
# After 40 steps the initial seed's influence on EMA is < 2% (alpha=0.095, N=20).
EMA_WARMUP_STEPS = 40

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
