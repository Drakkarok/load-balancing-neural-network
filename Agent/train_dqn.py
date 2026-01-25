
import os
import torch
import numpy as np
from lbnn_env import LBNNEnv
from dqn_agent import DQNAgent
from metrics import EpisodeMetrics, save_metrics_to_csv
import config

def train():
    print("Initializing Environment and Agent...")
    env = LBNNEnv()
    agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
    
    # Curriculum phases
    phases = [
        ("Phase 1", config.PHASE_1_EPISODES, config.PHASE_1_LENGTH),
        ("Phase 2", config.PHASE_2_EPISODES, config.PHASE_2_LENGTH),
        ("Phase 3", config.PHASE_3_EPISODES, config.PHASE_3_LENGTH)
    ]
    
    total_episodes = 0
    metrics_buffer = []
    csv_path = "Models/metrics/training_metrics.csv"
    
    for phase_name, num_episodes, episode_len in phases:
        print(f"\n=== Starting {phase_name} ({num_episodes} episodes, length {episode_len}) ===")
        env.episode_length = episode_len
        
        for i in range(num_episodes):
            total_episodes += 1
            
            # Reset Environment
            state, _ = env.reset()
            episode_data = [] # Store tick info for metrics
            total_reward = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
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
                    "chosen_server": info.get("chosen_server"),
                    "reward": reward
                })
                total_reward += reward
                
            # End of Episode
            agent.update_epsilon()
            
            # Calculate Metrics
            ep_metrics = EpisodeMetrics()
            ep_metrics.compute_from_episode(episode_data, agent.epsilon, total_episodes)
            metrics_buffer.append(ep_metrics)
            
            # Log every 10 episodes
            if total_episodes % 10 == 0:
                ep_metrics.log()
                save_metrics_to_csv(metrics_buffer, csv_path)
                metrics_buffer = [] # Clear buffer
                
            # Save Checkpoint
            if total_episodes % config.CHECKPOINT_FREQ == 0:
                ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"dqn_episode_{total_episodes}.pth")
                os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                agent.save_checkpoint(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
                
    print("\nTraining Complete!")
    agent.save_checkpoint(os.path.join(config.CHECKPOINT_DIR, "dqn_final.pth"))

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
