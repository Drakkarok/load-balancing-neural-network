
import os
import torch
import numpy as np
import logging
import time
import argparse
import glob
import re
from datetime import datetime
from lbnn_env import LBNNEnv
from dqn_agent import DQNAgent
from metrics import EpisodeMetrics, save_metrics_to_csv
import config

# Setup Logging
log_dir = "Models/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "training_errors.log"),
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_success(episode):
    with open(os.path.join(log_dir, "training_status.txt"), "a") as f:
        f.write(f"{datetime.now()}: Episode {episode} completed successfully.\n")

def find_latest_checkpoint(checkpoint_dir):
    """Find the checkpoint with the highest episode number"""
    files = glob.glob(os.path.join(checkpoint_dir, "dqn_episode_*.pth"))
    if not files:
        return None, 0
    
    # Extract episode numbers using regex
    latest_file = None
    max_episode = 0
    
    for f in files:
        match = re.search(r"dqn_episode_(\d+).pth", f)
        if match:
            ep = int(match.group(1))
            if ep > max_episode:
                max_episode = ep
                latest_file = f
                
    return latest_file, max_episode

def train(resume=False):
    print("Initializing Environment and Agent...")
    try:
        env = LBNNEnv()
        agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
    except Exception as e:
        logging.error(f"Initialization failed: {e}", exc_info=True)
        print(f"CRITICAL: Initialization failed. See {log_dir}/training_errors.log")
        return
    
    start_episode = 0
    
    # Resume Logic
    if resume:
        print("Checking for checkpoints to resume...")
        ckpt_path, ckpt_episode = find_latest_checkpoint(config.CHECKPOINT_DIR)
        if ckpt_path:
            print(f"Resuming form checkpoint: {ckpt_path} (Episode {ckpt_episode})")
            agent.load_checkpoint(ckpt_path)
            start_episode = ckpt_episode
        else:
            print("No checkpoint found. Starting from scratch.")
    
    # Curriculum phases
    phases = [
        ("Phase 1", config.PHASE_1_EPISODES, config.PHASE_1_LENGTH),
        ("Phase 2", config.PHASE_2_EPISODES, config.PHASE_2_LENGTH),
        ("Phase 3", config.PHASE_3_EPISODES, config.PHASE_3_LENGTH)
    ]
    
    current_episode_tracker = 0 # Tracks global episode count to match against phases
    metrics_buffer = []
    csv_path = "Models/metrics/training_metrics.csv"
    
    for phase_name, num_episodes, episode_len in phases:
        print(f"\n=== Entering {phase_name} (Episodes {current_episode_tracker+1} to {current_episode_tracker+num_episodes}) ===")
        env.episode_length = episode_len
        
        for i in range(num_episodes):
            current_episode_tracker += 1
            
            # Skip episodes that are already done if resuming
            if current_episode_tracker <= start_episode:
                continue
                
            try:
                # Reset Environment
                state, _ = env.reset()
                episode_data = [] # Store tick info for metrics
                total_reward = 0
                
                done = False
                truncated = False
                
                while not (done or truncated):
                    # Throttling to prevent CPU exhaustion on Windows Docker
                    time.sleep(0.05) 

                    # Select Action
                    action = agent.select_action(state)
                    
                    # Step
                    next_state, reward, done, truncated, info = env.step(action)
                    
                    # Store transition
                    agent.memory.push(state, action, reward, next_state, done)
                    
                    # Move to next state
                    state = next_state
                    
                    # Optimize
                    loss = agent.optimize_model()
                    
                    # Store data for metrics
                    episode_data.append({
                    "server_states": info.get("server_states"),
                    "prev_server_states": info.get("prev_server_states"), 
                    "chosen_server": info.get("chosen_server"),
                    "reward": reward
                })
                    total_reward += reward
                
                # End of Episode
                agent.update_epsilon()
                
                # Calculate Metrics
                ep_metrics = EpisodeMetrics()
                ep_metrics.compute_from_episode(episode_data, agent.epsilon, current_episode_tracker)
                metrics_buffer.append(ep_metrics)
                
                # Log success for this episode
                # log_success(current_episode_tracker) 
                
                # Log every 10 episodes
                if current_episode_tracker % 10 == 0:
                    ep_metrics.log()
                    save_metrics_to_csv(metrics_buffer, csv_path)
                    metrics_buffer = [] # Clear buffer
                    
                # Save Checkpoint
                if current_episode_tracker % config.CHECKPOINT_FREQ == 0:
                    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"dqn_episode_{current_episode_tracker}.pth")
                    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                    agent.save_checkpoint(ckpt_path)
                    print(f"Saved checkpoint to {ckpt_path}")

            except Exception as e:
                logging.error(f"Error in Episode {current_episode_tracker}: {e}", exc_info=True)
                print(f"Error in Episode {current_episode_tracker}: {e}. Check logs.")
                # We try to continue to next episode, but sleep a bit to let sockets clear
                time.sleep(5) 
                
    print("\nTraining Complete!")
    agent.save_checkpoint(os.path.join(config.CHECKPOINT_DIR, "dqn_final.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    try:
        train(resume=args.resume)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    except Exception as e:
        logging.critical(f"Global Crash: {e}", exc_info=True)
        print(f"Training crashed. See {log_dir}/training_errors.log")
        import traceback
        traceback.print_exc()
