
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

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        1. Call Agent to reset episode state.
        2. Generate the first request.
        3. Get initial server states.
        4. Return initial observation.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Reset Agent
        try:
            requests.post(f"{AGENT_URL}/reset_episode")
            # Also need to reset servers (Agent handles this internally via /reset_episode usually, 
            # or we might need to trigger it. Let's assume Agent.reset_episode does it.
            # *Correction*: Agent.reset_episode in app.py only resets counters. 
            # We need to explicitly reset servers based on current app.py, or add that logic. 
            # Note: app.py reset_episode logic:
            # def reset_episode(self): ... self.current_episode_requests = 0 ...
            # It DOES NOT reset servers.
            # We should likely add a /hard_reset endpoint to Agent or call servers directly.
            # Detailed Plan: I will MODIFY app.py to have a /reset_system endpoint that resets servers too.
            pass 
        except Exception as e:
            print(f"Error resetting environment: {e}")
        
        # Generate first request
        self.current_request = self._generate_request()
        
        # Get current state from Agent
        server_states = self._get_server_states()
        
        # Construct observation
        observation = self._construct_state(server_states, self.current_request)
        
        return observation, {}

    def step(self, action):
        """
        Execute one step in the environment.
        1. Send the routing decision + current request to Agent.
        2. Agent executes the tick.
        3. Receive new state and reward.
        4. Return (obs, reward, done, truncated, info).
        """
        self.current_step += 1
        
        # Prepare payload for Agent to execute specific action
        # Note: We need a new endpoint in app.py that accepts a FORCED action/decision
        # OR we modify /route_request to accept an optional 'forced_server_index'
        payload = {
            "request": self.current_request,
            "forced_action": int(action) 
        }
        
        reward = 0
        done = False
        truncated = False
        info = {}
        
        try:
            # Send to Agent to execute
            response = requests.post(f"{AGENT_URL}/step_training", json=payload, timeout=5)
            data = response.json()
            
            # Extract reward (calculated by Trainer/Agent)
            # The agent will need to return the reward calculated for this specific step.
            # Currently app.py doesn't calculate immediate rewards per tick in the response, 
            # it waits for the Trainer at the end. 
            # *CRITICAL CHANGE*: We need the reward IMMEDIATELY for DQN.
            # We can calculate it here or have Agent do it. 
            # Plan: Have Agent return 'server_states' (before and after) and we calculate reward locally 
            # using the same formula to avoid latency/complexity of calling Trainer every tick?
            # OR we implement the Load-Invariant Reward formula here in Python.
            # Implementation Plan says "DQN will use final_reward directly".
            # "Already calculated by your Trainer...".
            # But Trainer calculates it *post-episode*. 
            # For DQN training, we need it *per step*.
            # I will implement the reward calculation logic HERE in step() for immediate feedback, 
            # ensuring it matches the Trainer's logic.
            
            # data should contain: current_server_states (after tick), etc.
            server_states = data.get("current_server_states", {})
            
            # Calculate Reward (Load Invariant)
            chosen_server_id = f"server-{action+1}"
            reward = self._calculate_reward(server_states, chosen_server_id)
            
            # Generate NEXT request (for next observation)
            if self.current_step >= self.episode_length:
                done = True
                self.current_request = None # No next request
                # Observation will partial/dummy or we just return last state with zero request
                # But standard is to return a valid state.
                next_request = self._generate_request() # Just to keep shape
            else:
                next_request = self._generate_request()
                self.current_request = next_request
            
            observation = self._construct_state(server_states, next_request)
            
            info = {
                "server_states": server_states,
                "chosen_server": chosen_server_id
            }
            
        except Exception as e:
            print(f"Error in step: {e}")
            done = True # End episode on error
            observation = np.zeros(12, dtype=np.float32)
            
        return observation, reward, done, truncated, info

    def _generate_request(self):
        """Generate a random request based on types"""
        req = random.choice(self.request_types).copy()
        # Add random noise if we want, but keeping it discrete for now as per K6
        return req

    def _get_server_states(self):
        """Fetch current states from Agent"""
        try:
            resp = requests.get(f"{AGENT_URL}/server_states")
            return resp.json()
        except:
            return {}

    def _construct_state(self, server_states, request):
        """Normalize and construct 12-dim vector"""
        # Default empty state if fetch failed
        if not server_states:
            return np.zeros(12, dtype=np.float32)

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

    def _calculate_reward(self, server_states, chosen_id):
        """
        Calculate Load-Invariant Reward.
        Reward = (Average Load of Others) - (Load of Chosen)
        Load = CPU% + Memory%
        """
        loads = {}
        for sid, state in server_states.items():
            loads[sid] = state.get("cpu", 0) + state.get("memory", 0)
            
        chosen_load = loads.get(chosen_id, 0)
        
        other_loads = [l for s, l in loads.items() if s != chosen_id]
        if not other_loads:
            return 0.0
            
        avg_other_load = sum(other_loads) / len(other_loads)
        
        return avg_other_load - chosen_load

