
import os
import torch
import numpy as np
import time
from dqn_agent import DQNAgent
from lbnn_env import LBNNEnv
from metrics import EpisodeMetrics
import config

def evaluate(model_path="Models/checkpoints/dqn_final.pth", num_episodes=50):
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # Initialize
    env = LBNNEnv()
    agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
    agent.load_checkpoint(model_path)
    
    # Disable exploration
    agent.epsilon = 0.0
    
    print(f"Starting evaluation over {num_episodes} episodes...")
    
    metrics_buffer = []
    env.episode_length = 200 # Standard testing length
    
    for i in range(num_episodes):
        state, _ = env.reset()
        episode_data = []
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select Greedy Action
            action = agent.select_action(state, eval_mode=True)
            
            # Step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store data
            episode_data.append({
                "server_states": info.get("server_states"),
                "chosen_server": info.get("chosen_server"),
                "reward": reward
            })
            total_reward += reward
            state = next_state
            
            # Tiny sleep to be polite to Docker
            time.sleep(0.01)
            
        # Compute Metrics
        metrics = EpisodeMetrics()
        metrics.compute_from_episode(episode_data, 0.0, i+1)
        metrics_buffer.append(metrics)
        
        print(f"Ep {i+1}: Reward={total_reward:.2f}, Gini={metrics.capacity_normalized_gini:.3f}, Optimal={metrics.optimal_decision_rate:.1f}%")

    # Aggregate Results
    avg_reward = np.mean([m.total_reward for m in metrics_buffer])
    avg_gini = np.mean([m.capacity_normalized_gini for m in metrics_buffer])
    avg_optimal = np.mean([m.optimal_decision_rate for m in metrics_buffer])
    
    print("\n=== Evaluation Results ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Gini Coefficient (Fairness): {avg_gini:.3f}")
    print(f"Average Optimal Decision Rate: {avg_optimal:.1f}%")
    print("==========================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Models/checkpoints/dqn_final.pth", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to test")
    args = parser.parse_args()
    
    evaluate(args.model, args.episodes)
