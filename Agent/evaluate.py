"""
Evaluation harness: comparative evaluation of DQN vs baseline routing policies.

Commands (run from Agent/ directory):

  Generate shared request traces first:
    python evaluate.py generate-traces [--traces 5] [--requests 5150] [--output-dir eval]

  Run evaluation (errors if traces are missing):
    python evaluate.py run [--model path/to/ckpt.pth] [--traces 5] [--skip-dqn] [--output-dir eval]
"""

import os
import sys
import json
import csv
import random
import argparse
import math
import subprocess

import numpy as np
import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_IDS     = ["server-1", "server-2", "server-3"]
WARMUP_TICKS   = 150
MEASURED_TICKS = 5000
TOTAL_TICKS    = WARMUP_TICKS + MEASURED_TICKS   # 5150

BASELINE_METHODS = ["random", "round_robin", "least_connections", "least_load", "greedy_min_peak"]
ALL_METHODS      = BASELINE_METHODS + ["dqn"]

METRICS_KEYS = [
    "mean_peak", "max_peak", "p95_peak", "frac_over_90",
    "gini", "util_std",
    "jitter", "temporal_std", "rolling_std",   # temporal volatility metrics
]
ROLLING_STD_W = 30  # window for rolling_std (ticks)

# Per-server fields recorded in the result CSV.
# Raw state (12) + EMA delta features (9) = 21 per server — same layout as the DQN obs.
_SERVER_FIELDS = [
    # --- raw utilisation ---
    "cpu", "mem", "conn",
    # --- bracket cost / count sums (raw, same as SimServer) ---
    "cpu_brk_low", "cpu_brk_mid", "cpu_brk_high",
    "mem_brk_low", "mem_brk_mid", "mem_brk_high",
    "cnt_brk_low", "cnt_brk_mid", "cnt_brk_high",
    # --- EMA delta features (normalised, same computation as lbnn_env) ---
    "cnt_dlt_low", "cnt_dlt_mid", "cnt_dlt_high",
    "cpu_dlt_low", "cpu_dlt_mid", "cpu_dlt_high",
    "mem_dlt_low", "mem_dlt_mid", "mem_dlt_high",
]

CSV_COLUMNS = (
    ["trace", "method", "tick", "phase",
     "req_cpu", "req_mem", "req_duration", "req_type", "action"]
    + [f"s{i+1}_{f}" for i in range(3) for f in _SERVER_FIELDS]
)


# ---------------------------------------------------------------------------
# Server simulator (mirrors server.py / analysis_test.py)
# ---------------------------------------------------------------------------

class SimServer:
    def __init__(self, server_id, max_cpu, max_memory):
        self.server_id  = server_id
        self.max_cpu    = max_cpu
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
            "expires_at":  tick_id + req["duration"],
        })

    def get_state(self):
        total_cpu = sum(r["cpu_cost"]    for r in self.active_requests)
        total_mem = sum(r["memory_cost"] for r in self.active_requests)
        brackets = {
            "cpu":   {"low": 0.0, "mid": 0.0, "high": 0.0},
            "mem":   {"low": 0.0, "mid": 0.0, "high": 0.0},
            "count": {"low": 0,   "mid": 0,   "high": 0},
        }
        for r in self.active_requests:
            remaining = r["expires_at"] - self.tick
            if   1 <= remaining <= 7:   key = "low"
            elif 8 <= remaining <= 14:  key = "mid"
            elif 15 <= remaining <= 21: key = "high"
            else: continue
            brackets["cpu"][key]   += r["cpu_cost"]
            brackets["mem"][key]   += r["memory_cost"]
            brackets["count"][key] += 1
        return {
            "cpu":        round(min(100.0, total_cpu / self.max_cpu    * 100), 2),
            "memory":     round(min(100.0, total_mem / self.max_memory * 100), 2),
            "connections": len(self.active_requests),
            "bracket_counts": brackets,
        }


class SimCluster:
    def __init__(self):
        self.servers = {
            sid: SimServer(sid, cap["cpu"], cap["memory"])
            for sid, cap in config.SERVER_CAPACITIES.items()
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
# EMA tracker  (exact same computation as lbnn_env._update_ema)
# ---------------------------------------------------------------------------

class EMATracker:
    """
    Tracks EMA-delta features for bracket values across ticks.
    Mirrors lbnn_env._construct_state so the recorded deltas are identical
    to what the DQN observes — making all methods directly comparable.
    """

    def __init__(self):
        self._alpha = 2.0 / (config.EMA_N + 1)
        self._ema = self._empty_ema()

    def _empty_ema(self):
        return {
            sid: {
                "cpu":   {"low": None, "mid": None, "high": None},
                "mem":   {"low": None, "mid": None, "high": None},
                "count": {"low": None, "mid": None, "high": None},
            }
            for sid in SERVER_IDS
        }

    def _update(self, sid, resource, key, value):
        if self._ema[sid][resource][key] is None:
            self._ema[sid][resource][key] = value
            return 0.0
        self._ema[sid][resource][key] = (
            self._alpha * value + (1 - self._alpha) * self._ema[sid][resource][key]
        )
        return value - self._ema[sid][resource][key]

    def seed(self, states):
        """Seed EMA from initial server states (call once after cluster.reset()).
        Mirrors lbnn_env: env.reset() builds the first obs from empty states,
        which seeds all bracket EMAs to 0 and returns delta = 0."""
        self._ema = self._empty_ema()
        self.compute_deltas(states)   # discards returned deltas (all zero)

    def compute_deltas(self, states):
        """
        Normalise bracket values and compute EMA deltas for all servers.
        Returns dict: {sid: {field: delta_value}}.
        """
        deltas = {}
        for sid in SERVER_IDS:
            s   = states[sid]
            brk = s.get("bracket_counts", {})
            cap = config.SERVER_CAPACITIES[sid]
            d   = {}
            for key in ["low", "mid", "high"]:
                cpu_norm = min(brk.get("cpu",   {}).get(key, 0.0) / cap["cpu"],    1.0)
                mem_norm = min(brk.get("mem",   {}).get(key, 0.0) / cap["memory"], 1.0)
                cnt_norm = min(brk.get("count", {}).get(key, 0)   / config.MAX_CONNECTIONS, 1.0)
                d[f"cpu_dlt_{key}"] = self._update(sid, "cpu",   key, cpu_norm)
                d[f"mem_dlt_{key}"] = self._update(sid, "mem",   key, mem_norm)
                d[f"cnt_dlt_{key}"] = self._update(sid, "count", key, cnt_norm)
            deltas[sid] = d
        return deltas


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------

def _generate_one_request(py_rng, np_rng):
    """Mirrors LBNNEnv._generate_request with explicit RNGs for reproducibility."""
    req_type = py_rng.choice(["light", "medium", "heavy"])
    if req_type == "light":
        return {
            "cpu_cost":    py_rng.randint(150, 200),
            "memory_cost": py_rng.randint(100, 125),
            "duration":    int(np.clip(round(np_rng.normal(5.0,  2.0)),  1,  9)),
            "type": "light",
        }
    elif req_type == "medium":
        return {
            "cpu_cost":    py_rng.randint(350, 400),
            "memory_cost": py_rng.randint(300, 350),
            "duration":    int(np.clip(round(np_rng.normal(10.5, 2.5)),  5, 16)),
            "type": "medium",
        }
    else:
        return {
            "cpu_cost":    py_rng.randint(400, 500),
            "memory_cost": py_rng.randint(400, 450),
            "duration":    int(np.clip(round(np_rng.normal(16.5, 2.0)), 12, 21)),
            "type": "heavy",
        }


def generate_trace(seed, n_requests=TOTAL_TICKS):
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    return [_generate_one_request(py_rng, np_rng) for _ in range(n_requests)]


def save_trace(trace, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(trace, f)


def load_trace(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Policies  (signature: states, req, tick, rng -> int action 0/1/2)
# ---------------------------------------------------------------------------

def _util(state):
    return max(state["cpu"], state["memory"]) / 100.0


def policy_random(states, req, tick, rng):
    return rng.randint(0, 2)


def policy_round_robin(states, req, tick, rng):
    return (tick - 1) % 3


def policy_least_connections(states, req, tick, rng):
    return min(range(3), key=lambda i: states[SERVER_IDS[i]]["connections"])


def policy_least_load(states, req, tick, rng):
    return min(range(3), key=lambda i: _util(states[SERVER_IDS[i]]))


def policy_greedy_min_peak(states, req, tick, rng):
    best_action = 0
    best_peak   = float("inf")
    best_util   = float("inf")   # tie-breaker: lowest current utilisation = most headroom
    for i, sid in enumerate(SERVER_IDS):
        cap = config.SERVER_CAPACITIES[sid]
        cpu_after = min(100.0, states[sid]["cpu"]    + req["cpu_cost"]    / cap["cpu"]    * 100)
        mem_after = min(100.0, states[sid]["memory"] + req["memory_cost"] / cap["memory"] * 100)
        peak = 0.0
        for j, sid2 in enumerate(SERVER_IDS):
            if j == i:
                peak = max(peak, max(cpu_after, mem_after) / 100.0)
            else:
                peak = max(peak, _util(states[sid2]))
        cur_util = _util(states[sid])
        if peak < best_peak or (peak == best_peak and cur_util < best_util):
            best_peak, best_util, best_action = peak, cur_util, i
    return best_action


POLICY_FNS = {
    "random":            policy_random,
    "round_robin":       policy_round_robin,
    "least_connections": policy_least_connections,
    "least_load":        policy_least_load,
    "greedy_min_peak":   policy_greedy_min_peak,
}


# ---------------------------------------------------------------------------
# Per-tick row building
# ---------------------------------------------------------------------------

def _make_row(trace_id, method, tick, req, action, states, ema_deltas):
    phase = "warmup" if tick <= WARMUP_TICKS else "measured"
    row = {
        "trace":        trace_id,
        "method":       method,
        "tick":         tick,
        "phase":        phase,
        "req_cpu":      req["cpu_cost"],
        "req_mem":      req["memory_cost"],
        "req_duration": req["duration"],
        "req_type":     req["type"],
        "action":       action,
    }
    for i, sid in enumerate(SERVER_IDS):
        s   = states[sid]
        brk = s.get("bracket_counts", {})
        p   = f"s{i+1}"
        row[f"{p}_cpu"]          = s["cpu"]
        row[f"{p}_mem"]          = s["memory"]
        row[f"{p}_conn"]         = s["connections"]
        row[f"{p}_cpu_brk_low"]  = brk.get("cpu",   {}).get("low",  0.0)
        row[f"{p}_cpu_brk_mid"]  = brk.get("cpu",   {}).get("mid",  0.0)
        row[f"{p}_cpu_brk_high"] = brk.get("cpu",   {}).get("high", 0.0)
        row[f"{p}_mem_brk_low"]  = brk.get("mem",   {}).get("low",  0.0)
        row[f"{p}_mem_brk_mid"]  = brk.get("mem",   {}).get("mid",  0.0)
        row[f"{p}_mem_brk_high"] = brk.get("mem",   {}).get("high", 0.0)
        row[f"{p}_cnt_brk_low"]  = brk.get("count", {}).get("low",  0)
        row[f"{p}_cnt_brk_mid"]  = brk.get("count", {}).get("mid",  0)
        row[f"{p}_cnt_brk_high"] = brk.get("count", {}).get("high", 0)
        d = ema_deltas.get(sid, {})
        row[f"{p}_cnt_dlt_low"]  = d.get("cnt_dlt_low",  0.0)
        row[f"{p}_cnt_dlt_mid"]  = d.get("cnt_dlt_mid",  0.0)
        row[f"{p}_cnt_dlt_high"] = d.get("cnt_dlt_high", 0.0)
        row[f"{p}_cpu_dlt_low"]  = d.get("cpu_dlt_low",  0.0)
        row[f"{p}_cpu_dlt_mid"]  = d.get("cpu_dlt_mid",  0.0)
        row[f"{p}_cpu_dlt_high"] = d.get("cpu_dlt_high", 0.0)
        row[f"{p}_mem_dlt_low"]  = d.get("mem_dlt_low",  0.0)
        row[f"{p}_mem_dlt_mid"]  = d.get("mem_dlt_mid",  0.0)
        row[f"{p}_mem_dlt_high"] = d.get("mem_dlt_high", 0.0)
    return row


def _save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def run_baseline(method_name, policy_fn, trace_id, trace, output_dir):
    cluster = SimCluster()
    cluster.reset()
    ema = EMATracker()
    ema.seed(cluster.current_states())   # seed from empty states, mirrors env.reset()
    rng = random.Random(trace_id * 1000 + hash(method_name) % 1000)

    rows = []
    for tick_0, req in enumerate(trace):
        tick   = tick_0 + 1
        states = cluster.current_states()
        action = policy_fn(states, req, tick, rng)
        new_states  = cluster.step(SERVER_IDS[action], req)
        ema_deltas  = ema.compute_deltas(new_states)
        rows.append(_make_row(trace_id, method_name, tick, req, action, new_states, ema_deltas))

    path = os.path.join(output_dir, "results", method_name, f"trace_{trace_id}.csv")
    _save_csv(rows, path)


# ---------------------------------------------------------------------------
# DQN runner
# ---------------------------------------------------------------------------

def run_dqn(model_path, trace_id, trace, output_dir):
    from unittest.mock import MagicMock
    from dqn_agent import DQNAgent

    agent = DQNAgent()
    agent.load_checkpoint(model_path)
    agent.epsilon = 0.0

    cluster = SimCluster()

    def _make_mock_session():
        session = MagicMock()
        session.headers = MagicMock()
        session.headers.update = MagicMock()

        def _get(url, **kw):
            resp = MagicMock()
            resp.json.return_value = cluster.current_states()
            return resp

        def _post(url, json=None, **kw):
            resp = MagicMock()
            if "reset_episode" in url:
                cluster.reset()
                resp.json.return_value = {"status": "reset"}
            elif "step_training" in url:
                req    = json["request"]
                action = json["forced_action"]
                new_states = cluster.step(SERVER_IDS[action], req)
                resp.json.return_value = {
                    "status": "processed",
                    "current_server_states": new_states,
                }
            return resp

        session.get.side_effect  = _get
        session.post.side_effect = _post
        return session

    if "lbnn_env" in sys.modules:
        del sys.modules["lbnn_env"]
    from lbnn_env import LBNNEnv

    class TraceEnv(LBNNEnv):
        def __init__(self):
            super().__init__()
            self._tr       = []
            self._tr_idx   = 0
            self._injecting = False

        def load_trace_for_eval(self, tr):
            self._tr       = tr
            self._tr_idx   = 0
            self._injecting = True
            if self._tr:
                self.current_request = self._tr[self._tr_idx]
                self._tr_idx += 1

        def _run_warmup(self):
            pass   # skip; eval's 150-tick warmup covers this

        def _generate_request(self):
            if self._injecting and self._tr_idx < len(self._tr):
                req = self._tr[self._tr_idx]
                self._tr_idx += 1
                return req
            return super()._generate_request()

    env = TraceEnv()
    env.session        = _make_mock_session()
    env.episode_length = TOTAL_TICKS

    obs, _ = env.reset()            # seeds env's internal EMA from empty states
    env.load_trace_for_eval(trace)

    # External EMA tracker seeded identically to env's internal EMA
    ema = EMATracker()
    ema.seed(cluster.current_states())

    rows = []
    for tick_0 in range(TOTAL_TICKS):
        tick   = tick_0 + 1
        action = agent.select_action(obs, eval_mode=True)
        obs, _reward, done, truncated, info = env.step(action)

        req        = info["request"]
        states     = info["server_states"]
        ema_deltas = ema.compute_deltas(states)
        rows.append(_make_row(trace_id, "dqn", tick, req, action, states, ema_deltas))

        if done or truncated:
            break

    path = os.path.join(output_dir, "results", "dqn", f"trace_{trace_id}.csv")
    _save_csv(rows, path)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _gini3(a, b, c):
    vals = sorted([a, b, c])
    s = sum(vals)
    if s < 1e-10:
        return 0.0
    return (2 * (vals[0] + 2*vals[1] + 3*vals[2]) - 4*s) / (3 * s)


def compute_metrics(csv_path):
    peaks, ginis, util_stds = [], [], []
    server_utils = [[], [], []]   # per-server utilisation time series

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["phase"] != "measured":
                continue
            utils = [
                max(float(row[f"s{i+1}_cpu"]), float(row[f"s{i+1}_mem"])) / 100.0
                for i in range(3)
            ]
            peaks.append(max(utils))
            ginis.append(_gini3(*utils))
            util_stds.append(float(np.std(utils)))
            for i in range(3):
                server_utils[i].append(utils[i])

    if not peaks:
        return {k: float("nan") for k in METRICS_KEYS}

    peaks     = np.array(peaks)
    ginis     = np.array(ginis)
    util_stds = np.array(util_stds)
    su        = [np.array(u) for u in server_utils]

    # Jitter: mean |u[t] - u[t-1]| per server, averaged across servers.
    jitter = float(np.mean([np.mean(np.abs(np.diff(u))) for u in su]))

    # Temporal std: std of each server's full time series, averaged across servers.
    temporal_std = float(np.mean([np.std(u) for u in su]))

    # Rolling std: mean of per-window std (window = ROLLING_STD_W ticks),
    # averaged across servers.
    rolling_parts = []
    for u in su:
        if len(u) >= ROLLING_STD_W:
            windows = np.lib.stride_tricks.sliding_window_view(u, ROLLING_STD_W)
            rolling_parts.append(float(np.mean(np.std(windows, axis=1))))
    rolling_std = float(np.mean(rolling_parts)) if rolling_parts else float("nan")

    return {
        "mean_peak":    float(peaks.mean()),
        "max_peak":     float(peaks.max()),
        "p95_peak":     float(np.percentile(peaks, 95)),
        "frac_over_90": float((peaks > 0.90).mean()),
        "gini":         float(ginis.mean()),
        "util_std":     float(util_stds.mean()),
        "jitter":       jitter,
        "temporal_std": temporal_std,
        "rolling_std":  rolling_std,
    }


# ---------------------------------------------------------------------------
# Summary statistics + paired diffs
# ---------------------------------------------------------------------------

def compute_summary(output_dir, n_traces=None, trace_ids=None):
    # Accept either an explicit list of IDs or auto-detect from results/dqn/.
    if trace_ids is None:
        if n_traces is not None:
            trace_ids = list(range(1, n_traces + 1))
        else:
            results_dqn = os.path.join(output_dir, "results", "dqn")
            trace_ids = _detect_trace_ids(results_dqn, ext=".csv")
    all_metrics = {m: [] for m in ALL_METHODS}
    for method in ALL_METHODS:
        for t in trace_ids:
            path = os.path.join(output_dir, "results", method, f"trace_{t}.csv")
            if os.path.exists(path):
                all_metrics[method].append(compute_metrics(path))

    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # metrics_summary.csv
    fieldnames   = ["method"] + [f"{k}_{s}" for k in METRICS_KEYS for s in ["mean", "std"]]
    summary_rows = []
    for method in ALL_METHODS:
        ms = all_metrics[method]
        if not ms:
            continue
        row = {"method": method}
        for key in METRICS_KEYS:
            vals = [m[key] for m in ms if not math.isnan(m.get(key, float("nan")))]
            row[f"{key}_mean"] = round(float(np.mean(vals)), 6) if vals else float("nan")
            row[f"{key}_std"]  = round(float(np.std(vals)),  6) if vals else float("nan")
        summary_rows.append(row)

    summary_path = os.path.join(summary_dir, "metrics_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # paired_diffs.csv
    diff_rows = []
    for baseline in ["least_load", "greedy_min_peak"]:
        dqn_ms  = all_metrics["dqn"]
        base_ms = all_metrics[baseline]
        n = min(len(dqn_ms), len(base_ms))
        for key in METRICS_KEYS:
            diffs = [dqn_ms[i][key] - base_ms[i][key] for i in range(n)]
            diff_rows.append({
                "comparison": f"dqn_vs_{baseline}",
                "metric":     key,
                "mean_diff":  round(float(np.mean(diffs)), 6) if diffs else float("nan"),
                "std_diff":   round(float(np.std(diffs)),  6) if diffs else float("nan"),
                "n_traces":   n,
                "dqn_wins":   sum(1 for d in diffs if d < 0),
            })

    diff_path = os.path.join(summary_dir, "paired_diffs.csv")
    with open(diff_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["comparison", "metric", "mean_diff", "std_diff", "n_traces", "dqn_wins"]
        )
        writer.writeheader()
        writer.writerows(diff_rows)

    _print_summary(summary_rows, diff_rows)
    print(f"\nSaved {summary_path}")
    print(f"Saved {diff_path}")
    return all_metrics


def _print_summary(summary_rows, diff_rows):
    sep = "=" * 90
    print(f"\n{sep}")
    print("  RESULTS SUMMARY")
    print(sep)
    print(f"  {'Method':<22} {'mean_peak':>10} {'p95_peak':>10} {'frac_>90':>10}"
          f" {'gini':>8} {'jitter':>8} {'tmp_std':>8} {'roll_std':>9}")
    print("-" * 90)
    for row in summary_rows:
        print(f"  {row['method']:<22} "
              f"{row['mean_peak_mean']:>10.4f} "
              f"{row['p95_peak_mean']:>10.4f} "
              f"{row['frac_over_90_mean']:>10.4f} "
              f"{row['gini_mean']:>8.4f} "
              f"{row['jitter_mean']:>8.4f} "
              f"{row['temporal_std_mean']:>8.4f} "
              f"{row['rolling_std_mean']:>9.4f}")
    print(sep)
    print("\n  PAIRED DIFFERENCES  (dqn − baseline; negative = DQN wins on peak/volatility metrics)")
    print("-" * 90)
    highlight = {"mean_peak", "jitter", "temporal_std", "rolling_std"}
    for row in diff_rows:
        if row["metric"] in highlight:
            sign = "✓" if row["mean_diff"] < 0 else "✗"
            print(f"  {row['comparison']:<30}  {row['metric']:<14}"
                  f" diff = {row['mean_diff']:+.4f} ± {row['std_diff']:.4f}"
                  f"  DQN wins {row['dqn_wins']}/{row['n_traces']}  {sign}")
    print(sep)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_trace(output_dir, trace_id=1):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    colors = {
        "random":            "lightgray",
        "round_robin":       "steelblue",
        "least_connections": "green",
        "least_load":        "orange",
        "greedy_min_peak":   "red",
        "dqn":               "purple",
    }
    fig, ax = plt.subplots(figsize=(14, 5))
    for method in ALL_METHODS:
        path = os.path.join(output_dir, "results", method, f"trace_{trace_id}.csv")
        if not os.path.exists(path):
            continue
        ticks, peaks = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["phase"] == "measured":
                    utils = [
                        max(float(row[f"s{i+1}_cpu"]), float(row[f"s{i+1}_mem"])) / 100.0
                        for i in range(3)
                    ]
                    ticks.append(int(row["tick"]))
                    peaks.append(max(utils))
        ax.plot(ticks, peaks, label=method, color=colors.get(method), alpha=0.85, linewidth=0.9)

    ax.set_xlabel("Tick")
    ax.set_ylabel("Peak Load  max(util₁, util₂, util₃)")
    ax.set_title(f"Peak Load per Tick — Trace {trace_id}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "summary", f"trace_{trace_id}_peak.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _detect_trace_ids(traces_dir, ext=".json"):
    """Return sorted list of trace IDs found as trace_N<ext> in traces_dir."""
    if not os.path.isdir(traces_dir):
        return []
    ids = []
    for name in os.listdir(traces_dir):
        if name.startswith("trace_") and name.endswith(ext):
            stem = name[len("trace_"):-len(ext)]
            if stem.isdigit():
                ids.append(int(stem))
    return sorted(ids)


def cmd_generate_traces(args):
    traces_dir = os.path.join(args.output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    for i in range(1, args.traces + 1):
        path = os.path.join(traces_dir, f"trace_{i}.json")
        seed = random.randint(1, 10000)
        print(f"Generating trace {i} (seed={seed}, {args.requests} requests)...", end=" ", flush=True)
        trace = generate_trace(seed, n_requests=args.requests)
        save_trace(trace, path)
        print(f"done  →  {path}")
    print(f"\n{args.traces} trace(s) saved to {traces_dir}")
    print("Run 'python evaluate.py run' to start evaluation.")


def cmd_run(args):
    traces_dir = os.path.join(args.output_dir, "traces")

    # Auto-detect traces when --traces is not explicitly set.
    if args.traces is None:
        trace_ids = _detect_trace_ids(traces_dir)
        if not trace_ids:
            print("ERROR: No trace files found in", traces_dir)
            print("  Generate them first with:")
            print(f"  python evaluate.py generate-traces --output-dir {args.output_dir}")
            sys.exit(1)
        print(f"Auto-detected {len(trace_ids)} trace(s): {trace_ids}")
    else:
        trace_ids = list(range(1, args.traces + 1))
        # Guard: all requested trace files must exist
        missing = [
            os.path.join(traces_dir, f"trace_{i}.json")
            for i in trace_ids
            if not os.path.exists(os.path.join(traces_dir, f"trace_{i}.json"))
        ]
        if missing:
            print("ERROR: Trace files not found. Generate them first with:")
            print(f"  python evaluate.py generate-traces --traces {args.traces} --output-dir {args.output_dir}")
            print("\nMissing files:")
            for p in missing:
                print(f"  {p}")
            sys.exit(1)

    n_traces = len(trace_ids)

    # Load traces
    traces = []
    for i in trace_ids:
        path = os.path.join(traces_dir, f"trace_{i}.json")
        print(f"Loading trace {i} from {path}")
        traces.append(load_trace(path))

    # Run baselines
    for method_name, policy_fn in POLICY_FNS.items():
        print(f"\nRunning {method_name}...")
        for idx, (tid, trace) in enumerate(zip(trace_ids, traces), 1):
            print(f"  trace {tid} ({idx}/{n_traces})...", end=" ", flush=True)
            run_baseline(method_name, policy_fn, tid, trace, args.output_dir)
            print("done")

    # Run DQN
    if not args.skip_dqn:
        if os.path.exists(args.model):
            print(f"\nRunning dqn ({args.model})...")
            for idx, (tid, trace) in enumerate(zip(trace_ids, traces), 1):
                print(f"  trace {tid} ({idx}/{n_traces})...", end=" ", flush=True)
                run_dqn(args.model, tid, trace, args.output_dir)
                print("done")
        else:
            print(f"\nSkipping DQN — checkpoint not found: {args.model}")
            print("  Pass --skip-dqn to suppress this warning, or --model to specify a path.")

    # Summary + built-in quick plot
    print("\nComputing summary statistics...")
    compute_summary(args.output_dir, n_traces=n_traces, trace_ids=trace_ids)
    plot_trace(args.output_dir, trace_id=args.plot_trace)

    # Full figure generation via plot_results.py
    plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_results.py")
    if os.path.exists(plot_script):
        print("\nGenerating dissertation figures...")
        result = subprocess.run(
            [sys.executable, plot_script,
             "--results-dir", os.path.join(args.output_dir, "results"),
             "--out-dir",     os.path.join(args.output_dir, "figures"),
             "--trace",       str(args.plot_trace)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode != 0:
            print("WARNING: plot_results.py exited with errors — figures may be incomplete.")
    else:
        print(f"\nSkipping figures — plot_results.py not found at {plot_script}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Evaluation harness for DQN load balancer vs baselines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate-traces ---
    gen = subparsers.add_parser(
        "generate-traces",
        help="Generate shared request trace files (do this once before running evaluation).",
    )
    gen.add_argument("--traces",     type=int, default=5,          help="Number of traces to generate")
    gen.add_argument("--requests",   type=int, default=TOTAL_TICKS, help="Requests per trace (default: 5150)")
    gen.add_argument("--output-dir", default="eval",               help="Root output directory")

    # --- run ---
    run = subparsers.add_parser(
        "run",
        help="Run evaluation. Errors if trace files are missing.",
    )
    run.add_argument("--model",      default="Models/checkpoints/dqn_final.pth")
    run.add_argument("--traces",     type=int, default=None, help="Number of traces to use (default: auto-detect from traces/ dir)")
    run.add_argument("--output-dir", default="eval",        help="Root output directory")
    run.add_argument("--skip-dqn",   action="store_true",   help="Run baselines only")
    run.add_argument("--plot-trace", type=int, default=1,   help="Which trace to plot (1-indexed)")

    # --- summarize ---
    summ = subparsers.add_parser(
        "summarize",
        help="Recompute summary statistics from existing result CSVs (no re-simulation).",
    )
    summ.add_argument("--traces",     type=int, default=None, help="Number of traces (default: auto-detect from results/)")
    summ.add_argument("--output-dir", default="eval",       help="Root output directory")

    args = parser.parse_args()

    if args.command == "generate-traces":
        cmd_generate_traces(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "summarize":
        print("Recomputing summary from existing result CSVs...")
        compute_summary(args.output_dir, n_traces=args.traces if args.traces else None)
        plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_results.py")
        if os.path.exists(plot_script):
            print("\nGenerating dissertation figures...")
            subprocess.run(
                [sys.executable, plot_script,
                 "--results-dir", os.path.join(args.output_dir, "results"),
                 "--out-dir",     os.path.join(args.output_dir, "figures")],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        print("Done.")
