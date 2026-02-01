
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import json
import random

# Use localhost for training since the training script runs on the host
AGENT_URL = "http://localhost:8080" # Agent is exposed on 8080

class LBNNEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment acts as the 'Traffic Generator' during training,
    sending requests to the Agent (which then routes them to servers).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LBNNEnv, self).__init__()
        
        # Action space: 0, 1, 2 (corresponding to server-1, server-2, server-3)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 12 numerical features (normalized [0, 1])
        # [s1_cpu, s1_mem, s1_conn, s2_cpu, s2_mem, s2_conn, s3_cpu, s3_mem, s3_conn, req_cpu, req_mem, req_dur]
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        
        # State tracking
        self.current_request = None
        self.episode_length = 150 # Default, will be overridden by curriculum
        self.current_step = 0
        
        # FIX: Cache the server states BEFORE the action is taken usually
        self.last_server_states = {} 
        
        # Request generation configuration (matching K6)
        self.request_types = [
            {"cpu": 50, "memory": 30, "duration": 2},    # Light
            {"cpu": 150, "memory": 100, "duration": 4},  # Medium
            {"cpu": 300, "memory": 200, "duration": 6}   # Heavy
        ]
        
        # Normalization constants
        self.MAX_CPU = 5000.0  # server-3
        self.MAX_MEM = 3200.0  # server-2
        self.MAX_DURATION = 6.0
        self.MAX_CONNECTIONS = 10.0 # Estimated max

        # Use a session for persistent connections to Agent
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        1. Call Agent to reset episode state.
        2. Generate the first request.
        3. Get initial server states (PRE-ACTION state for the first step).
        4. Return initial observation.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Reset Agent
        try:
            # This resets the episode counters on the Agent
            self.session.post(f"{AGENT_URL}/reset_episode")
        except Exception as e:
            print(f"Error resetting environment: {e}")
        
        # Generate first request
        self.current_request = self._generate_request()
        
        # FIX: Get initial server states to serve as the "Pre-Action" state for step 1
        self.last_server_states = self._get_server_states()
        
        # Construct observation using these initial states
        observation = self._construct_state(self.last_server_states, self.current_request)
        
        return observation, {}

    def step(self, action):
        """
        Execute one step in the environment.
        1. Calculate Reward using PRE-ACTION state (self.last_server_states).
        2. Execute action (send to Agent).
        3. Receive NEW state (Post-Action) and cache it for NEXT step's reward.
        4. Return (obs, reward, done, truncated, info).
        """
        self.current_step += 1
        
        # Capture Pre-Action state for Metrics (Deep Copy to avoid reference issues)
        import copy
        pre_action_state = copy.deepcopy(self.last_server_states)
        
        chosen_server_id = f"server-{action+1}"
        
        # FIX 1: Calculate Reward based on PRE-ACTION state (Timing Bug Fix)
        # We use self.last_server_states which was captured at the end of the previous step (or reset)
        reward = self._calculate_reward(self.last_server_states, chosen_server_id)
        
        # Prepare payload for Agent to execute specific action
        payload = {
            "request": self.current_request,
            "forced_action": int(action) 
        }
        
        done = False
        truncated = False
        info = {}
        
        try:
            # Send to Agent to execute action
            # This will apply the load and return the NEW state
            response = self.session.post(f"{AGENT_URL}/step_training", json=payload, timeout=60)
            data = response.json()
            
            # Get the NEW states (Post-Action)
            current_server_states = data.get("current_server_states", {})
            
            # FIX 1 (cont): Update cached state for the NEXT step's reward calculation
            self.last_server_states = current_server_states
            
            # Generate NEXT request (for next observation)
            if self.current_step >= self.episode_length:
                done = True
                self.current_request = None # No next request
                next_request = self._generate_request() # Dummy for shape
            else:
                next_request = self._generate_request()
                self.current_request = next_request
            
            # Construct observation for the NEXT step
            # Note: The observation sees the POST-ACTION state (what the system looks like now)
            # plus the NEXT request to be scheduled.
            observation = self._construct_state(current_server_states, next_request)
            
            info = {
                "server_states": current_server_states,
                "chosen_server": chosen_server_id,
                "prev_server_states": pre_action_state # Correct Pre-Action state for Metrics
            }
            
        except Exception as e:
            print(f"Error in step: {e}")
            raise e 
            
        return observation, reward, done, truncated, info

    def _generate_request(self):
        """Generate a random request based on types"""
        req = random.choice(self.request_types).copy()
        return req

    def _get_server_states(self):
        """Fetch current states from Agent"""
        try:
            resp = self.session.get(f"{AGENT_URL}/server_states")
            return resp.json()
        except:
            # Return empty structure if failed
            return {
                "server-1": {"cpu": 0, "memory": 0, "connections": 0},
                "server-2": {"cpu": 0, "memory": 0, "connections": 0},
                "server-3": {"cpu": 0, "memory": 0, "connections": 0}
            }

    def _construct_state(self, server_states, request):
        """Normalize and construct 12-dim vector"""
        if not server_states:
            server_states = self._get_server_states() # Fallback

        # Normalize Server 1
        s1 = server_states.get("server-1", {"cpu":0, "memory":0, "connections":0})
        s1_vec = [
            s1.get("cpu", 0) / 100.0,
            s1.get("memory", 0) / 100.0,
            min(s1.get("connections", 0) / self.MAX_CONNECTIONS, 1.0)
        ]

        # Normalize Server 2
        s2 = server_states.get("server-2", {"cpu":0, "memory":0, "connections":0})
        s2_vec = [
            s2.get("cpu", 0) / 100.0,
            s2.get("memory", 0) / 100.0,
            min(s2.get("connections", 0) / self.MAX_CONNECTIONS, 1.0)
        ]

        # Normalize Server 3
        s3 = server_states.get("server-3", {"cpu":0, "memory":0, "connections":0})
        s3_vec = [
            s3.get("cpu", 0) / 100.0,
            s3.get("memory", 0) / 100.0,
            min(s3.get("connections", 0) / self.MAX_CONNECTIONS, 1.0)
        ]

        # Normalize Request
        if request:
            req_vec = [
                request["cpu"] / self.MAX_CPU,
                request["memory"] / self.MAX_MEM,
                request["duration"] / self.MAX_DURATION
            ]
        else:
            req_vec = [0.0, 0.0, 0.0]

        return np.array(s1_vec + s2_vec + s3_vec + req_vec, dtype=np.float32)

    def _calculate_reward(self, server_states, chosen_server_id):
        """
        FIX 2: Implement Gradient-Based Reward Logic (Ported from Trainer/trainer.py)
        Logic:
        1. Identify Min Load.
        2. Identify Optimal Group (servers with Min Load).
        3. If Chosen in Optimal Group: Reward = Avg(Suboptimal) - Chosen.
        4. If Chosen in Suboptimal Group: Penalty = Avg(Optimal) - Chosen.
        5. If Tie (All Equal): Reward = 0.
        
        Using Scaling Factor: 1/100 to normalize rewards for NN stability.
        """
        # 1. Parse states into simplified Load dict
        loads = {}
        parsed_states = {}
        
        if not server_states:
             return 0.0
             
        for sid, state in server_states.items():
            cpu = state.get("cpu", 0)
            mem = state.get("memory", 0)
            total_load = cpu + mem
            loads[sid] = total_load
            parsed_states[sid] = {'cpu': cpu, 'memory': mem, 'total_load': total_load}
            
        # 2. Find optimal servers (Min Load)
        min_load = min(loads.values())
        optimal_servers = [sid for sid, load in loads.items() if load == min_load]
        
        # Case 1: All servers are equal (Tie)
        if len(optimal_servers) == len(loads):
            return 0.0
            
        chosen_cpu = parsed_states[chosen_server_id]['cpu']
        chosen_mem = parsed_states[chosen_server_id]['memory']
        
        reward = 0.0
            
        # Case 2: Optimal Choice
        if chosen_server_id in optimal_servers:
            # Positive reward vs Suboptimal
            suboptimal_servers = [sid for sid in loads.keys() if sid not in optimal_servers]
            
            avg_suboptimal_cpu = sum(parsed_states[sid]['cpu'] for sid in suboptimal_servers) / len(suboptimal_servers)
            avg_suboptimal_mem = sum(parsed_states[sid]['memory'] for sid in suboptimal_servers) / len(suboptimal_servers)
            
            cpu_reward = avg_suboptimal_cpu - chosen_cpu
            mem_reward = avg_suboptimal_mem - chosen_mem
            
            reward = cpu_reward + mem_reward
            
        # Case 3: Suboptimal Choice
        else:
            # Negative penalty vs Optimal
            avg_optimal_cpu = sum(parsed_states[sid]['cpu'] for sid in optimal_servers) / len(optimal_servers)
            avg_optimal_mem = sum(parsed_states[sid]['memory'] for sid in optimal_servers) / len(optimal_servers)
            
            cpu_reward = avg_optimal_cpu - chosen_cpu
            mem_reward = avg_optimal_mem - chosen_mem
            
            reward = cpu_reward + mem_reward
            
        # Scale Reward
        return reward / 100.0

