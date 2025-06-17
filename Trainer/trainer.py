import requests
import json
import time
import numpy as np
import threading
from flask import Flask, request, jsonify
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

app = Flask(__name__)

class LoadBalancerTrainer:
    def __init__(self):
        self.servers = [
            {"id": "server-1", "url": "http://lbnn-server-1:8081"},
            {"id": "server-2", "url": "http://lbnn-server-2:8082"},
            {"id": "server-3", "url": "http://lbnn-server-3:8083"}
        ]
        self.agent_url = "http://lbnn-agent:8080"
        self.training_in_progress = False
        self.training_lock = threading.Lock()
        
    def collect_episode_data(self):
        """Collect complete episode history from all servers"""
        print("Collecting episode data from all servers...")
        episode_data = {}
        
        for server in self.servers:
            try:
                response = requests.get(f"{server['url']}/get_episode_history", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    episode_data[server['id']] = data
                    print(f"Collected {len(data['episode_data'])} ticks from {server['id']}")
                else:
                    print(f"Failed to collect data from {server['id']}: {response.status_code}")
            except Exception as e:
                print(f"Error collecting from {server['id']}: {e}")
        
        return episode_data
    
    def calculate_rewards(self, episode_data):
        """Calculate rewards using gradient-based approach"""
        print("Calculating rewards using gradient-based approach...")
        
        # Get all ticks across servers
        all_ticks = {}
        for server_id, data in episode_data.items():
            for tick_entry in data['episode_data']:
                tick_id = tick_entry['tick_id']
                if tick_id not in all_ticks:
                    all_ticks[tick_id] = {}
                all_ticks[tick_id][server_id] = tick_entry
        
        rewards = []
        
        for tick_id in sorted(all_ticks.keys()):
            tick_data = all_ticks[tick_id]
            
            # Find which server got the real request
            chosen_server = None
            request_data = None
            
            for server_id, tick_entry in tick_data.items():
                if tick_entry.get('processed_real_request', False):
                    chosen_server = server_id
                    request_data = tick_entry['request']
                    break
            
            if chosen_server and request_data:
                # Calculate gradient-based reward
                final_reward, cpu_reward, memory_reward = self.calculate_gradient_reward(
                    tick_data, chosen_server, request_data
                )
                
                rewards.append({
                    'tick_id': tick_id,
                    'chosen_server': chosen_server,
                    'request': request_data,
                    'final_reward': final_reward,
                    'cpu_reward': cpu_reward,
                    'memory_reward': memory_reward,
                    'server_states': {sid: entry['state_after_processing'] for sid, entry in tick_data.items()}
                })
        
        print(f"Calculated gradient-based rewards for {len(rewards)} decisions")
        return rewards
    
    def calculate_gradient_reward(self, tick_data, chosen_server, request_data):
        """Calculate reward using gradient-based approach with tie handling"""
        
        # Collect server states
        server_states = {}
        for server_id, tick_entry in tick_data.items():
            state = tick_entry['state_after_processing']
            server_states[server_id] = {
                'cpu': state['cpu'],
                'memory': state['memory'],
                'total_load': state['cpu'] + state['memory']
            }
        
        # Find minimum load and all servers with that load
        total_loads = {sid: state['total_load'] for sid, state in server_states.items()}
        min_load = min(total_loads.values())
        optimal_servers = [sid for sid, load in total_loads.items() if load == min_load]
        
        # Case 1: All servers are equal (complete tie)
        if len(optimal_servers) == len(total_loads):            
            print(f"All servers tied at {min_load}% load - giving neutral reward")
            return 0.0, 0.0, 0.0
        
        # Case 2: Agent chose optimally (server is in optimal group)
        if chosen_server in optimal_servers:
            # Calculate positive reward vs average of suboptimal servers
            suboptimal_servers = [sid for sid in total_loads.keys() if sid not in optimal_servers]
            
            avg_suboptimal_cpu = sum(server_states[sid]['cpu'] for sid in suboptimal_servers) / len(suboptimal_servers)
            avg_suboptimal_memory = sum(server_states[sid]['memory'] for sid in suboptimal_servers) / len(suboptimal_servers)
            
            chosen_cpu = server_states[chosen_server]['cpu']
            chosen_memory = server_states[chosen_server]['memory']
            
            cpu_reward = avg_suboptimal_cpu - chosen_cpu
            memory_reward = avg_suboptimal_memory - chosen_memory
            
            final_reward = cpu_reward + memory_reward
            print(f"Optimal choice: {chosen_server} vs avg suboptimal. Reward: {final_reward:.3f}")
            
            return final_reward, cpu_reward, memory_reward
        
        # Case 3: Agent chose suboptimally
        else:
            # Calculate penalty vs average of optimal servers
            avg_optimal_cpu = sum(server_states[sid]['cpu'] for sid in optimal_servers) / len(optimal_servers)
            avg_optimal_memory = sum(server_states[sid]['memory'] for sid in optimal_servers) / len(optimal_servers)
            
            chosen_cpu = server_states[chosen_server]['cpu']
            chosen_memory = server_states[chosen_server]['memory']
            
            cpu_reward = avg_optimal_cpu - chosen_cpu  # Negative penalty
            memory_reward = avg_optimal_memory - chosen_memory  # Negative penalty
            
            final_reward = cpu_reward + memory_reward
            print(f"Suboptimal choice: {chosen_server}. Penalty: {final_reward:.3f}")
            
            return final_reward, cpu_reward, memory_reward
    
    def analyze_rewards(self, rewards):
        """Analyze the calculated rewards"""
        if not rewards:
            print("No rewards to analyze")
            return {}
        
        print(f"\n=== REWARD ANALYSIS ===")
        print(f"Total decisions analyzed: {len(rewards)}")
        
        # Overall statistics
        final_rewards = [r['final_reward'] for r in rewards]
        cpu_rewards = [r['cpu_reward'] for r in rewards]
        memory_rewards = [r['memory_reward'] for r in rewards]
        
        print(f"\nFinal Rewards:")
        print(f"  Average: {np.mean(final_rewards):.3f}")
        print(f"  Min: {np.min(final_rewards):.3f}")
        print(f"  Max: {np.max(final_rewards):.3f}")
        print(f"  Std: {np.std(final_rewards):.3f}")
        
        # Decision quality analysis
        optimal_decisions = sum(1 for r in rewards if r['final_reward'] > 0)
        suboptimal_decisions = sum(1 for r in rewards if r['final_reward'] < 0)
        neutral_decisions = sum(1 for r in rewards if r['final_reward'] == 0)
        
        print(f"\nDecision Quality:")
        print(f"  Optimal decisions: {optimal_decisions} ({optimal_decisions/len(rewards)*100:.1f}%)")
        print(f"  Suboptimal decisions: {suboptimal_decisions} ({suboptimal_decisions/len(rewards)*100:.1f}%)")
        print(f"  Neutral decisions: {neutral_decisions} ({neutral_decisions/len(rewards)*100:.1f}%)")
        
        # Server distribution analysis
        server_choices = {}
        for r in rewards:
            server = r['chosen_server']
            if server not in server_choices:
                server_choices[server] = []
            server_choices[server].append(r['final_reward'])
        
        print(f"\nServer Choice Analysis:")
        for server, server_rewards in server_choices.items():
            avg_reward = np.mean(server_rewards)
            count = len(server_rewards)
            print(f"  {server}: {count} choices, avg reward: {avg_reward:.3f}")
        
        # Return summary for API response
        return {
            'total_decisions': len(rewards),
            'average_reward': float(np.mean(final_rewards)),
            'reward_std': float(np.std(final_rewards)),
            'optimal_decisions': optimal_decisions,
            'suboptimal_decisions': suboptimal_decisions,
            'neutral_decisions': neutral_decisions,
            'server_distribution': {server: len(server_rewards) for server, server_rewards in server_choices.items()}
        }
    
    def reset_servers(self):
        """Reset all servers for new episode"""
        print("Resetting servers for new episode...")
        reset_results = {}
        
        for server in self.servers:
            try:
                response = requests.get(f"{server['url']}/reset_episode", timeout=5)
                if response.status_code == 200:
                    print(f"Reset {server['id']}")
                    reset_results[server['id']] = "success"
                else:
                    print(f"Failed to reset {server['id']}: {response.status_code}")
                    reset_results[server['id']] = "failed"
            except Exception as e:
                print(f"Error resetting {server['id']}: {e}")
                reset_results[server['id']] = "error"
        
        return reset_results

# Global trainer instance
trainer = LoadBalancerTrainer()

@app.route('/health')
def health():
    return {
        "status": "Trainer is running", 
        "training_in_progress": trainer.training_in_progress
    }

@app.route('/start_training', methods=['POST'])
def start_training():
    """Triggered by k6 when episode completes"""
    data = request.get_json()
    
    with trainer.training_lock:
        if trainer.training_in_progress:
            return {"error": "Training already in progress"}, 400
        
        trainer.training_in_progress = True
    
    try:
        print(f"=== EPISODE TRAINING STARTED ===")
        if data:
            print(f"k6 reported: {data}")
        
        # Collect episode data
        episode_data = trainer.collect_episode_data()
        
        if not episode_data:
            return {"error": "Failed to collect episode data"}, 500
        
        # Calculate rewards
        rewards = trainer.calculate_rewards(episode_data)
        
        if not rewards:
            return {"error": "No rewards calculated"}, 500
        
        # Analyze rewards
        analysis = trainer.analyze_rewards(rewards)
        
        # Save results
        timestamp = int(time.time())
        
        # Save detailed rewards
        rewards_file = f'/app/models/episode_{timestamp}_rewards.json'
        with open(rewards_file, 'w') as f:
            json.dump(rewards, f, indent=2)
        
        # Save analysis summary
        analysis['timestamp'] = timestamp
        analysis['episode_file'] = rewards_file
        
        summary_file = f'/app/models/episode_{timestamp}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"=== TRAINING COMPLETE ===")
        print(f"Saved to: {rewards_file}")
        print(f"Summary: {summary_file}")
        
        return {
            "status": "training_complete",
            "analysis": analysis,
            "files": {
                "rewards": rewards_file,
                "summary": summary_file
            }
        }
        
    except Exception as e:
        print(f"Training error: {e}")
        return {"error": str(e)}, 500
    
    finally:
        trainer.training_in_progress = False

@app.route('/reset_episode', methods=['POST'])
def reset_episode():
    """Reset all servers for new episode"""
    try:
        reset_results = trainer.reset_servers()
        return {
            "status": "episode_reset",
            "server_resets": reset_results
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/training_status')
def training_status():
    """Check if training is in progress"""
    return {
        "training_in_progress": trainer.training_in_progress,
        "timestamp": time.time()
    }

if __name__ == '__main__':
    print("Load Balancer Trainer Service Starting...")
    print("=== GRADIENT-BASED REWARD SYSTEM ===")
    print("Available endpoints:")
    print("  POST /start_training - Process completed episode")
    print("  POST /reset_episode - Reset servers for new episode")
    print("  GET /training_status - Check training status")
    print("  GET /health - Service health check")
    
    app.run(host='0.0.0.0', port=8084, debug=True)