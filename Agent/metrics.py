
import numpy as np
import csv
import os
from config import SERVER_CAPACITIES

class EpisodeMetrics:
    """Container for all metrics for one episode"""
    
    def __init__(self):
        # Primary metrics
        self.optimal_decision_rate = 0.0  # percentage
        self.average_reward = 0.0
        self.total_reward = 0.0
        
        # Fairness metrics
        self.capacity_normalized_gini = 0.0
        self.utilization_std = 0.0
        
        # Bookkeeping
        self.episode_number = 0
        self.episode_length = 0
        self.epsilon = 0.0
        self.num_optimal_decisions = 0
        self.num_suboptimal_decisions = 0
        
    def compute_from_episode(self, episode_data, epsilon, episode_num):
        """
        Compute all metrics from episode data.
        episode_data: List of dicts with 'server_states', 'chosen_server', 'final_reward'
        """
        self.episode_length = len(episode_data)
        self.epsilon = epsilon
        self.episode_number = episode_num
        
        if self.episode_length == 0:
            return

        optimal_count = 0
        total_reward = 0.0
        
        for tick_data in episode_data:
            if self._is_optimal_decision(tick_data):
                optimal_count += 1
            # Assuming 'final_reward' or using calculated reward
            total_reward += tick_data.get("reward", 0.0)
        
        self.num_optimal_decisions = optimal_count
        self.num_suboptimal_decisions = self.episode_length - optimal_count
        self.optimal_decision_rate = (optimal_count / self.episode_length) * 100
        self.total_reward = total_reward
        self.average_reward = total_reward / self.episode_length
        
        # Fairness metrics
        self.capacity_normalized_gini = self._calculate_gini(episode_data)
        self.utilization_std = self._calculate_utilization_std(episode_data)
    
    def _is_optimal_decision(self, tick_data):
        """Check if chosen server had min total load"""
        servers = tick_data.get("server_states", {})
        chosen_id = tick_data.get("chosen_server")
        
        if not servers or not chosen_id:
            return False
            
        loads = {}
        for sid, state in servers.items():
            loads[sid] = state.get("cpu", 0) + state.get("memory", 0)
        
        # Optimal is minimum load
        optimal_server = min(loads, key=loads.get)
        return chosen_id == optimal_server

    def _calculate_gini(self, episode_data):
        """Calculate Gini of capacity-normalized CPU utilization"""
        total_cpu_time = {"server-1": 0, "server-2": 0, "server-3": 0}
        
        for tick in episode_data:
            states = tick.get("server_states")
            if not states:
                continue
            for sid, state in states.items():
                total_cpu_time[sid] += state.get("cpu", 0)
        
        utilizations = []
        for sid, total in total_cpu_time.items():
            avg_load = total / len(episode_data)
            utilizations.append(avg_load)
            
        utilizations = sorted(utilizations)
        n = len(utilizations)
        cumsum = np.cumsum(utilizations)
        sum_utils = cumsum[-1]
        
        if sum_utils == 0:
            return 0.0
            
        gini = (2 * np.sum((np.arange(1, n+1)) * utilizations)) / (n * sum_utils) - (n + 1) / n
        return gini

    def _calculate_utilization_std(self, episode_data):
        """Std dev of average utils"""
        server_utils = {"server-1": [], "server-2": [], "server-3": []}
        
        for tick in episode_data:
            states = tick.get("server_states")
            if not states:
                continue
            for sid, state in states.items():
                util = state.get("cpu", 0) + state.get("memory", 0)
                server_utils[sid].append(util)
        
        avg_utils = [np.mean(utils) if utils else 0 for utils in server_utils.values()]
        return np.std(avg_utils)

    def to_dict(self):
        return {
            "episode": self.episode_number,
            "optimal_decision_rate": self.optimal_decision_rate,
            "average_reward": self.average_reward,
            "total_reward": self.total_reward,
            "gini_coefficient": self.capacity_normalized_gini,
            "utilization_std": self.utilization_std,
            "epsilon": self.epsilon,
            "episode_length": self.episode_length
        }
    
    def log(self):
        print(f"""
        Episode {self.episode_number}:
        ├─ Optimal Decisions: {self.optimal_decision_rate:.1f}% ({self.num_optimal_decisions}/{self.episode_length})
        ├─ Avg Reward: {self.average_reward:.2f}
        ├─ Total Reward: {self.total_reward:.2f}
        ├─ Gini Coefficient: {self.capacity_normalized_gini:.3f}
        ├─ Utilization Std: {self.utilization_std:.1f}%
        └─ Epsilon: {self.epsilon:.3f}
        """)

def save_metrics_to_csv(metrics_list, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as f:
        fieldnames = ["episode", "optimal_decision_rate", "average_reward", 
                      "total_reward", "gini_coefficient", "utilization_std", 
                      "epsilon", "episode_length"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        for m in metrics_list:
            writer.writerow(m.to_dict())
