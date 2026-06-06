"""
Analysis test: 5 episodes × 250 steps with a lightweight server simulator.
No Docker required — the simulator replicates real server tick logic.
Run from the Agent/ directory: python3 analysis_test.py
"""

import sys
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Lightweight server simulator (mirrors server.py logic)
# ---------------------------------------------------------------------------

class SimServer:
    def __init__(self, server_id, max_cpu, max_memory):
        self.server_id = server_id
        self.max_cpu = max_cpu
        self.max_memory = max_memory
        self.active_requests = []
        self.tick = 0

    def reset(self):
        self.active_requests = []
        self.tick = 0

    def advance(self, tick_id):
        self.tick = tick_id
        self.active_requests = [r for r in self.active_requests if r["expires_at"] > tick_id]

    def add_request(self, req, tick_id):
        self.active_requests.append({
            "cpu_cost":    req["cpu_cost"],
            "memory_cost": req["memory_cost"],
            "expires_at":  tick_id + req["duration"]
        })

    def get_state(self):
        total_cpu = sum(r["cpu_cost"]    for r in self.active_requests)
        total_mem = sum(r["memory_cost"] for r in self.active_requests)
        brackets = {
            "cpu":   {"low": 0.0, "mid": 0.0, "high": 0.0},
            "mem":   {"low": 0.0, "mid": 0.0, "high": 0.0},
            "count": {"low": 0,   "mid": 0,   "high": 0}
        }
        for r in self.active_requests:
            remaining = r["expires_at"] - self.tick
            if   1 <= remaining <= 7:  key = "low"
            elif 8 <= remaining <= 14: key = "mid"
            elif 15 <= remaining <= 21: key = "high"
            else: continue
            brackets["cpu"][key]   += r["cpu_cost"]
            brackets["mem"][key]   += r["memory_cost"]
            brackets["count"][key] += 1
        return {
            "cpu":        round(min(100.0, total_cpu / self.max_cpu * 100), 2),
            "memory":     round(min(100.0, total_mem / self.max_memory * 100), 2),
            "connections": len(self.active_requests),
            "bracket_counts": brackets
        }


class SimCluster:
    """Three simulated servers + tick counter — replaces Docker."""
    def __init__(self):
        from config import SERVER_CAPACITIES
        self.servers = {
            sid: SimServer(sid, cap["cpu"], cap["memory"])
            for sid, cap in SERVER_CAPACITIES.items()
        }
        self.tick = 0

    def reset(self):
        self.tick = 0
        for s in self.servers.values():
            s.reset()

    def step(self, chosen_sid, req):
        self.tick += 1
        for sid, s in self.servers.items():
            s.advance(self.tick)
            if sid == chosen_sid:
                s.add_request(req, self.tick)
        return {sid: s.get_state() for sid, s in self.servers.items()}

    def current_states(self):
        return {sid: s.get_state() for sid, s in self.servers.items()}


# ---------------------------------------------------------------------------
# Build mock session backed by the simulator
# ---------------------------------------------------------------------------

def make_session(cluster):
    session = MagicMock()
    session.headers = MagicMock()
    session.headers.update = MagicMock()

    def get_side_effect(url, **kw):
        resp = MagicMock()
        resp.json.return_value = cluster.current_states()
        return resp

    def post_side_effect(url, json=None, **kw):
        resp = MagicMock()
        if "reset_episode" in url:
            cluster.reset()
            resp.json.return_value = {"status": "reset"}
        elif "step_training" in url:
            req    = json["request"]
            action = json["forced_action"]
            sid    = f"server-{action + 1}"
            new_states = cluster.step(sid, req)
            resp.json.return_value = {
                "status": "processed",
                "current_server_states": new_states
            }
        return resp

    session.get.side_effect  = get_side_effect
    session.post.side_effect = post_side_effect
    return session


# ---------------------------------------------------------------------------
# Feature labels for the 66-dim vector
# ---------------------------------------------------------------------------

FEATURE_LABELS = []
for srv in ["S1", "S2", "S3"]:
    FEATURE_LABELS += [
        f"{srv}.cpu_util", f"{srv}.mem_util", f"{srv}.conn",
        f"{srv}.cpu_brk_low", f"{srv}.cpu_brk_mid", f"{srv}.cpu_brk_high",
        f"{srv}.mem_brk_low", f"{srv}.mem_brk_mid", f"{srv}.mem_brk_high",
        f"{srv}.cnt_brk_low", f"{srv}.cnt_brk_mid", f"{srv}.cnt_brk_high",
        f"{srv}.cnt_dlt_low", f"{srv}.cnt_dlt_mid", f"{srv}.cnt_dlt_high",
        f"{srv}.cpu_dlt_low", f"{srv}.cpu_dlt_mid", f"{srv}.cpu_dlt_high",
        f"{srv}.mem_dlt_low", f"{srv}.mem_dlt_mid", f"{srv}.mem_dlt_high",
    ]
FEATURE_LABELS += ["req.cpu_cost", "req.mem_cost", "req.duration"]

DELTA_INDICES   = [12,13,14,15,16,17,18,19,20,
                   33,34,35,36,37,38,39,40,41,
                   54,55,56,57,58,59,60,61,62]
BRACKET_INDICES = [3,4,5,6,7,8,9,10,11,
                   24,25,26,27,28,29,30,31,32,
                   45,46,47,48,49,50,51,52,53]
UTIL_INDICES    = [0,1,2, 21,22,23, 42,43,44]
REQUEST_INDICES = [63,64,65]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_analysis(n_episodes=5, episode_length=250):
    cluster = SimCluster()

    if "lbnn_env" in sys.modules:
        del sys.modules["lbnn_env"]

    with patch("lbnn_env.requests") as mock_req:
        mock_req.Session.return_value = make_session(cluster)
        from lbnn_env import LBNNEnv
        env = LBNNEnv()
        env.episode_length = episode_length

        all_obs      = []
        all_rewards  = []
        all_actions  = []
        ep_rewards   = []
        ep_truncated = []
        ep_done      = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_r = []
            done = False
            truncated = False
            for step in range(episode_length):
                action = env.action_space.sample()
                obs, reward, done, truncated, _ = env.step(action)
                all_obs.append(obs.copy())
                all_rewards.append(reward)
                all_actions.append(action)
                ep_r.append(reward)
                if done or truncated:
                    break
            ep_rewards.append(ep_r)
            ep_truncated.append(truncated)
            ep_done.append(done)

    obs_matrix = np.array(all_obs)
    rewards    = np.array(all_rewards)
    actions    = np.array(all_actions)
    N          = len(obs_matrix)

    sep  = "=" * 72
    sep2 = "-" * 72

    print(f"\n{sep}")
    print(f"  ANALYSIS: {n_episodes} episodes × {episode_length} steps  ({N} total transitions)")
    print(sep)

    # --- 1. Feature coverage ---
    print("\n[1] FEATURE COVERAGE  (mean ± std  |  min … max)")
    print(sep2)
    dead = []
    for i, label in enumerate(FEATURE_LABELS):
        col = obs_matrix[:, i]
        m, s, lo, hi = col.mean(), col.std(), col.min(), col.max()
        flag = "  *** DEAD ***" if s < 1e-4 else ""
        print(f"  {i:2d}  {label:<20s}  {m:+.4f} ± {s:.4f}   [{lo:+.4f}, {hi:+.4f}]{flag}")
        if s < 1e-4:
            dead.append(label)
    print(sep2)
    if dead:
        print(f"  DEAD FEATURES ({len(dead)}): {dead}")
    else:
        print(f"  All 66 features are active (std > 1e-4).")

    # --- 2. Delta analysis ---
    print(f"\n[2] DELTA FEATURES  (should be centred near 0, non-zero variance)")
    print(sep2)
    delta_cols = obs_matrix[:, DELTA_INDICES]
    print(f"  Global delta  mean={delta_cols.mean():+.4f}  std={delta_cols.std():.4f}"
          f"  min={delta_cols.min():+.4f}  max={delta_cols.max():+.4f}")
    nonzero_pct = (np.abs(delta_cols) > 1e-6).mean() * 100
    print(f"  Non-zero delta entries: {nonzero_pct:.1f}%  (expected >>0% after warmup)")

    # --- 3. Bracket analysis ---
    print(f"\n[3] BRACKET FEATURES  (fraction of time each bracket is non-zero)")
    print(sep2)
    for i in BRACKET_INDICES:
        col = obs_matrix[:, i]
        pct = (col > 1e-6).mean() * 100
        print(f"  [{i:2d}] {FEATURE_LABELS[i]:<22s}  non-zero {pct:5.1f}%  "
              f"max={col.max():.4f}")

    # --- 4. Utilisation range ---
    print(f"\n[4] SERVER UTILISATION RANGE")
    print(sep2)
    for i in UTIL_INDICES:
        col = obs_matrix[:, i]
        print(f"  [{i:2d}] {FEATURE_LABELS[i]:<20s}  "
              f"mean={col.mean():.3f}  max={col.max():.3f}")

    # --- 5. Reward analysis (expected range: [-1, 0]) ---
    print(f"\n[5] REWARD ANALYSIS  (expected range [-1, 0] — negative peak load)")
    print(sep2)
    print(f"  Overall   mean={rewards.mean():+.4f}  std={rewards.std():.4f}"
          f"  min={rewards.min():+.4f}  max={rewards.max():+.4f}")
    out_of_range = np.sum((rewards < -1.0 - 1e-6) | (rewards > 1e-6))
    print(f"  Out-of-range rewards (not in [-1,0]): {out_of_range}", end="")
    print("  PASS" if out_of_range == 0 else "  FAIL")
    for ep_i, ep_r in enumerate(ep_rewards):
        a = np.array(ep_r)
        print(f"  Episode {ep_i+1}: sum={a.sum():+.2f}  mean={a.mean():+.4f}  "
              f"std={a.std():.4f}  "
              f"{'truncated' if ep_truncated[ep_i] else 'done' if ep_done[ep_i] else 'incomplete'}")

    # --- 6. Action distribution ---
    print(f"\n[6] ACTION DISTRIBUTION  (random policy — should be ~33% each)")
    print(sep2)
    for a in range(3):
        pct = (actions == a).mean() * 100
        print(f"  Server-{a+1}: {pct:.1f}%")

    # --- 7. Episode termination flags ---
    print(f"\n[7] EPISODE TERMINATION  (all should be truncated=True, done=False)")
    print(sep2)
    wrong_done      = sum(ep_done)
    wrong_truncated = sum(not t for t in ep_truncated)
    print(f"  Episodes ending with done=True    : {wrong_done}  (expected 0)")
    print(f"  Episodes ending with truncated=False: {wrong_truncated}  (expected 0)")
    if wrong_done == 0 and wrong_truncated == 0:
        print("  PASS — all episodes ended by time-limit truncation.")
    else:
        print("  FAIL — unexpected terminal flags.")

    # --- 8. Bounds check ---
    print(f"\n[8] BOUNDS CHECK  (all values must be in [-1, 1])")
    print(sep2)
    out_of_bounds = np.sum((obs_matrix < -1.0 - 1e-6) | (obs_matrix > 1.0 + 1e-6))
    nan_count     = np.sum(np.isnan(obs_matrix))
    print(f"  Out-of-bounds entries : {out_of_bounds}")
    print(f"  NaN entries           : {nan_count}")
    if out_of_bounds == 0 and nan_count == 0:
        print("  PASS — all observations within [-1, 1], no NaN.")
    else:
        print("  FAIL — see above.")

    print(f"\n{sep}\n")
    return obs_matrix, rewards, actions


if __name__ == "__main__":
    run_analysis(n_episodes=5, episode_length=250)
