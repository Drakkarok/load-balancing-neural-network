
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import random
import copy

from config import AGENT_URL, SERVER_CAPACITIES, EMA_WARMUP_STEPS, MAX_DURATION, MAX_CONNECTIONS, EMA_N

_EMPTY_BRACKETS = {
    "cpu":   {"low": 0.0, "mid": 0.0, "high": 0.0},
    "mem":   {"low": 0.0, "mid": 0.0, "high": 0.0},
    "count": {"low": 0,   "mid": 0,   "high": 0}
}

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

        # Observation space: 66 features per the expanded state spec.
        # Delta features can be negative, so bounds are [-1, 1].
        # Layout: [21 per server × 3 servers] + [3 request features]
        # Per server: cpu_util, mem_util, conn (3)
        #             cpu_brk×3, 
        #             mem_brk×3
        #             cpu_delta×3, 
        #             mem_delta×3
        #             count_brk×3
        #             count_delta×3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(66,), dtype=np.float32)

        # State tracking
        self.current_request = None
        self.episode_length = 150  # Default, overridden by curriculum
        self.current_step = 0

        # Cache server states BEFORE the action is taken
        self.last_server_states = {}

        # Normalization constants for request features — derived from config so they
        # stay in sync if server capacities ever change.
        self.MAX_CPU      = max(cap["cpu"]    for cap in SERVER_CAPACITIES.values())
        self.MAX_MEM      = max(cap["memory"] for cap in SERVER_CAPACITIES.values())
        self.MAX_DURATION = MAX_DURATION
        self.MAX_CONNECTIONS = MAX_CONNECTIONS

        self.EMA_ALPHA = 2.0 / (EMA_N + 1)
        self._ema = None  # initialized on first _construct_state call per episode

        # Use a session for persistent connections to Agent
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self._ema = None  # cold-start: first observation initializes EMA to current value

        try:
            self.session.post(f"{AGENT_URL}/reset_episode")
        except Exception as e:
            print(f"Error resetting environment: {e}")

        self.current_request = self._generate_request()
        self.last_server_states = self._get_server_states()

        # Warm up EMA over EMA_WARMUP_STEPS ticks before the episode proper begins.
        # These transitions are discarded — they only serve to drive the EMA past its
        # cold-start period so delta features carry real signal from step 1.
        self._run_warmup()

        observation = self._construct_state(self.last_server_states, self.current_request)
        return observation, {}

    def step(self, action):
        self.current_step += 1

        pre_action_state = copy.deepcopy(self.last_server_states)
        chosen_server_id = f"server-{action+1}"

        payload = {
            "request": self.current_request,
            "forced_action": int(action)
        }

        reward = 0.0
        done = False
        truncated = False
        info = {}

        try:
            response = self.session.post(f"{AGENT_URL}/step_training", json=payload, timeout=60)
            data = response.json()

            current_server_states = data.get("current_server_states", {})
            self.last_server_states = current_server_states

            reward = self._calculate_reward(current_server_states)

            if self.current_step >= self.episode_length:
                truncated = True
                next_request = self._generate_request()
                self.current_request = None
            else:
                next_request = self._generate_request()
                self.current_request = next_request

            observation = self._construct_state(current_server_states, next_request)

            info = {
                "server_states": current_server_states,
                "chosen_server": chosen_server_id,
                "prev_server_states": pre_action_state,
                "request": payload["request"]
            }

        except Exception as e:
            print(f"Error in step: {e}")
            raise e

        return observation, reward, done, truncated, info

    # ------------------------------------------------------------------
    # Request generation
    # ------------------------------------------------------------------

    def _generate_request(self):
        req_type = random.choice(["light", "medium", "heavy"])
        if req_type == "light":
            return {
                "cpu_cost": random.randint(150, 200),
                "memory_cost": random.randint(100, 125),
                "duration": int(np.clip(round(np.random.normal(5.0, 2.0)), 1, 9)),
                "type": "light"
            }
        elif req_type == "medium":
            return {
                "cpu_cost": random.randint(350, 400),
                "memory_cost": random.randint(300, 350),
                "duration": int(np.clip(round(np.random.normal(10.5, 2.5)), 5, 16)),
                "type": "medium"
            }
        else:
            return {
                "cpu_cost": random.randint(400, 500),
                "memory_cost": random.randint(400, 450),
                "duration": int(np.clip(round(np.random.normal(16.5, 2.0)), 12, 21)),
                "type": "heavy"
            }

    # ------------------------------------------------------------------
    # Server state fetching
    # ------------------------------------------------------------------

    def _get_server_states(self):
        try:
            resp = self.session.get(f"{AGENT_URL}/server_states")
            return resp.json()
        except Exception:
            return {
                sid: {"cpu": 0, "memory": 0, "connections": 0, "bracket_counts": copy.deepcopy(_EMPTY_BRACKETS)}
                for sid in ["server-1", "server-2", "server-3"]
            }

    # ------------------------------------------------------------------
    # EMA warmup
    # ------------------------------------------------------------------

    def _run_warmup(self):
        """Run EMA_WARMUP_STEPS ticks with random actions before the episode starts.
        Transitions are discarded — only the EMA and server state are updated.
        """
        for _ in range(EMA_WARMUP_STEPS):
            action = self.action_space.sample()
            payload = {
                "request": self.current_request,
                "forced_action": int(action)
            }
            try:
                response = self.session.post(
                    f"{AGENT_URL}/step_training", json=payload, timeout=60
                )
                data = response.json()
                server_states = data.get("current_server_states", {})
                self.last_server_states = server_states
                self.current_request = self._generate_request()
                # Update EMA state — observation is thrown away
                self._construct_state(server_states, self.current_request)
            except Exception as e:
                print(f"Warmup step failed: {e}")
                break

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    def _init_ema(self):
        self._ema = {
            sid: {
                "cpu":   {"low": None, "mid": None, "high": None},
                "mem":   {"low": None, "mid": None, "high": None},
                "count": {"low": None, "mid": None, "high": None}
            }
            for sid in ["server-1", "server-2", "server-3"]
        }

    def _update_ema(self, sid, resource, key, value):
        """Update EMA and return delta (current − EMA). First call seeds EMA = value → delta = 0."""
        if self._ema[sid][resource][key] is None:
            self._ema[sid][resource][key] = value
            return 0.0
        self._ema[sid][resource][key] = (
            self.EMA_ALPHA * value + (1 - self.EMA_ALPHA) * self._ema[sid][resource][key]
        )
        return value - self._ema[sid][resource][key]

    # ------------------------------------------------------------------
    # State construction (48-dim)
    # ------------------------------------------------------------------

    def _construct_state(self, server_states, request):
        """Build 66-dim observation vector.

        Per server (21 values × 3 = 63):
          [0-2]   cpu_util, mem_util, connections
          [3-8]   cpu_brk_low/mid/high, mem_brk_low/mid/high  (cost-weighted)
          [9-11]  count_brk_low/mid/high                       (raw request counts)
          [12-14] count_delta_low/mid/high                     (EMA deltas of counts)
          [15-20] cpu_delta_low/mid/high, mem_delta_low/mid/high

        Request (3 values):
          [cpu_cost, memory_cost, duration]
        """
        if not server_states:
            server_states = self._get_server_states()

        if self._ema is None:
            self._init_ema()

        vec = []

        for sid in ["server-1", "server-2", "server-3"]:
            s = server_states.get(sid, {})
            cap = SERVER_CAPACITIES[sid]
            brackets = s.get("bracket_counts", copy.deepcopy(_EMPTY_BRACKETS))

            # --- utilization (3) ---
            vec.append(s.get("cpu", 0) / 100.0)
            vec.append(s.get("memory", 0) / 100.0)
            vec.append(min(s.get("connections", 0) / self.MAX_CONNECTIONS, 1.0))

            # --- cost-weighted bracket sums (6) ---
            cpu_b = {
                k: min(brackets["cpu"].get(k, 0.0) / cap["cpu"], 1.0)
                for k in ["low", "mid", "high"]
            }
            mem_b = {
                k: min(brackets["mem"].get(k, 0.0) / cap["memory"], 1.0)
                for k in ["low", "mid", "high"]
            }
            vec.extend([cpu_b["low"], cpu_b["mid"], cpu_b["high"]])
            vec.extend([mem_b["low"], mem_b["mid"], mem_b["high"]])

            # --- raw count brackets (3) ---
            cnt_b = {
                k: min(brackets["count"].get(k, 0) / self.MAX_CONNECTIONS, 1.0)
                for k in ["low", "mid", "high"]
            }
            vec.extend([cnt_b["low"], cnt_b["mid"], cnt_b["high"]])

            # --- count EMA deltas (3) ---
            for key in ["low", "mid", "high"]:
                vec.append(self._update_ema(sid, "count", key, cnt_b[key]))

            # --- cost EMA deltas (6) ---
            for key in ["low", "mid", "high"]:
                vec.append(self._update_ema(sid, "cpu", key, cpu_b[key]))
            for key in ["low", "mid", "high"]:
                vec.append(self._update_ema(sid, "mem", key, mem_b[key]))

        # --- request features (3) ---
        if request:
            vec.extend([
                request["cpu_cost"] / self.MAX_CPU,
                request["memory_cost"] / self.MAX_MEM,
                request["duration"] / self.MAX_DURATION
            ])
        else:
            vec.extend([0.0, 0.0, 0.0])

        return np.array(vec, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _calculate_reward(self, server_states_after):
        """Negative peak server load across all servers after placement. Range [-1, 0]."""
        if not server_states_after:
            return 0.0
        peak = max(
            max(s.get("cpu", 0) / 100.0, s.get("memory", 0) / 100.0)
            for s in server_states_after.values()
        )
        return -(peak ** 2)
