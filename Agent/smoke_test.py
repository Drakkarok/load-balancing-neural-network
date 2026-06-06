"""
Smoke test for the 66-dim Dueling Double DQN load balancer.
Run from the Agent/ directory: python smoke_test.py
Requires no Docker — all HTTP calls are mocked.
"""

import sys
import numpy as np
import torch
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Realistic mock server states
# ---------------------------------------------------------------------------

def _make_server_states(variant="mixed"):
    if variant == "mixed":
        return {
            "server-1": {
                "cpu": 30.0, "memory": 25.0, "connections": 3,
                "bracket_counts": {
                    "cpu":   {"low": 300.0, "mid": 150.0, "high": 0.0},
                    "mem":   {"low": 200.0, "mid": 100.0, "high": 0.0},
                    "count": {"low": 2,     "mid": 1,     "high": 0}
                }
            },
            "server-2": {
                "cpu": 10.0, "memory": 8.0, "connections": 1,
                "bracket_counts": {
                    "cpu":   {"low": 150.0, "mid": 0.0, "high": 0.0},
                    "mem":   {"low": 100.0, "mid": 0.0, "high": 0.0},
                    "count": {"low": 1,     "mid": 0,   "high": 0}
                }
            },
            "server-3": {
                "cpu": 60.0, "memory": 50.0, "connections": 7,
                "bracket_counts": {
                    "cpu":   {"low": 500.0, "mid": 400.0, "high": 200.0},
                    "mem":   {"low": 400.0, "mid": 300.0, "high": 150.0},
                    "count": {"low": 3,     "mid": 2,     "high": 1}
                }
            }
        }
    return {
        sid: {
            "cpu": 0.0, "memory": 0.0, "connections": 0,
            "bracket_counts": {
                "cpu":   {"low": 0.0, "mid": 0.0, "high": 0.0},
                "mem":   {"low": 0.0, "mid": 0.0, "high": 0.0},
                "count": {"low": 0,   "mid": 0,   "high": 0}
            }
        }
        for sid in ["server-1", "server-2", "server-3"]
    }


def _make_mock_session(server_states):
    session = MagicMock()

    get_resp = MagicMock()
    get_resp.json.return_value = server_states
    session.get.return_value = get_resp

    reset_resp = MagicMock()
    reset_resp.json.return_value = {"status": "reset"}

    step_resp = MagicMock()
    step_resp.json.return_value = {
        "status": "processed",
        "current_server_states": server_states
    }

    def post_side_effect(url, **kwargs):
        return reset_resp if "reset_episode" in url else step_resp

    session.post.side_effect = post_side_effect
    session.headers = MagicMock()
    session.headers.update = MagicMock()
    return session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_obs(obs, label=""):
    tag = f"[{label}] " if label else ""
    assert obs.shape == (66,),         f"{tag}shape={obs.shape}, expected (66,)"
    assert obs.dtype == np.float32,    f"{tag}dtype={obs.dtype}"
    assert not np.any(np.isnan(obs)),  f"{tag}NaN detected"
    assert not np.any(np.isinf(obs)),  f"{tag}Inf detected"
    assert obs.min() >= -1.0 - 1e-6,  f"{tag}value below -1: min={obs.min()}"
    assert obs.max() <=  1.0 + 1e-6,  f"{tag}value above +1: max={obs.max()}"


# ---------------------------------------------------------------------------
# Test 1 — observation space declaration
# ---------------------------------------------------------------------------

def test_obs_space():
    with patch("lbnn_env.requests") as mock_req:
        mock_req.Session.return_value = _make_mock_session(_make_server_states())
        from lbnn_env import LBNNEnv
        env = LBNNEnv()
        assert env.observation_space.shape == (66,)
        assert env.observation_space.low[0]  == -1.0
        assert env.observation_space.high[0] ==  1.0
    print("  PASS  test_obs_space")


# ---------------------------------------------------------------------------
# Test 2 — reset returns valid 66-dim obs with zero deltas
# ---------------------------------------------------------------------------

def test_reset():
    with patch("lbnn_env.requests") as mock_req:
        mock_req.Session.return_value = _make_mock_session(_make_server_states())
        from lbnn_env import LBNNEnv
        env = LBNNEnv()
        obs, info = env.reset()
        _check_obs(obs, "reset")
        assert env._ema is not None, "_ema not initialised after reset"
        # Warmup feeds the same constant states, so EMA = those values → deltas = 0
        delta_indices = [12,13,14,15,16,17,18,19,20,
                         33,34,35,36,37,38,39,40,41,
                         54,55,56,57,58,59,60,61,62]
        for i in delta_indices:
            assert abs(obs[i]) < 1e-6, f"Expected delta=0 on first obs, got obs[{i}]={obs[i]}"
    print("  PASS  test_reset")


# ---------------------------------------------------------------------------
# Test 3 — reward function: range, value, empty input
# ---------------------------------------------------------------------------

def test_reward_range():
    with patch("lbnn_env.requests") as mock_req:
        mock_req.Session.return_value = _make_mock_session(_make_server_states())
        from lbnn_env import LBNNEnv
        env = LBNNEnv()

        # Mixed states: server-3 has cpu=60 → peak = 60/100 = 0.60 → reward = -0.60
        r = env._calculate_reward(_make_server_states("mixed"))
        assert -1.0 <= r <= 0.0, f"reward {r} outside [-1, 0]"
        assert abs(r - (-0.60)) < 1e-4, f"expected -0.60, got {r}"

        # Zero load → peak = 0 → reward = 0.0
        r_zero = env._calculate_reward(_make_server_states("zero"))
        assert r_zero == 0.0, f"zero-load reward should be 0.0, got {r_zero}"

        # Empty dict → 0.0 (guard clause)
        assert env._calculate_reward({}) == 0.0

    print("  PASS  test_reward_range")


# ---------------------------------------------------------------------------
# Test 4 — 10 episodes: obs valid, reward in [-1,0], truncated not done at end
# ---------------------------------------------------------------------------

def test_10_episodes():
    server_states = _make_server_states("mixed")

    with patch("lbnn_env.requests") as mock_req:
        mock_req.Session.return_value = _make_mock_session(server_states)
        from lbnn_env import LBNNEnv
        env = LBNNEnv()
        env.episode_length = 20

        for ep in range(10):
            obs, _ = env.reset()
            _check_obs(obs, f"ep{ep} reset")
            done = False
            truncated = False
            step = 0
            while not (done or truncated):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                _check_obs(obs, f"ep{ep} step{step}")
                assert isinstance(reward, float), f"reward not float: {type(reward)}"
                assert not np.isnan(reward), f"reward NaN at ep{ep} step{step}"
                assert -1.0 <= reward <= 0.0, f"reward {reward} outside [-1, 0] at ep{ep} step{step}"
                step += 1
            # Time-limit episode ends as truncation, not terminal
            assert truncated, f"ep{ep}: expected truncated=True at episode end"
            assert not done,  f"ep{ep}: expected done=False at episode end"

    print(f"  PASS  test_10_episodes  (10 episodes × {env.episode_length} steps)")


# ---------------------------------------------------------------------------
# Test 5 — EMA deltas are non-zero after load change
# ---------------------------------------------------------------------------

def test_ema_updates():
    from config import EMA_WARMUP_STEPS
    states_a = _make_server_states("mixed")
    states_b = _make_server_states("zero")

    session = _make_mock_session(states_a)

    step_call_count = [0]
    warmup_resp = MagicMock()
    warmup_resp.json.return_value = {"status": "processed", "current_server_states": states_a}
    real_resp = MagicMock()
    real_resp.json.return_value = {"status": "processed", "current_server_states": states_b}

    def post_side_effect(url, **kw):
        if "reset_episode" in url:
            return MagicMock()
        step_call_count[0] += 1
        return warmup_resp if step_call_count[0] <= EMA_WARMUP_STEPS else real_resp

    session.post.side_effect = post_side_effect

    with patch("lbnn_env.requests") as mock_req:
        mock_req.Session.return_value = session
        from lbnn_env import LBNNEnv
        env = LBNNEnv()
        env.episode_length = 5
        env.reset()  # warmup converges EMA to states_a

        obs, *_ = env.step(0)  # returns states_b (zeros) → deltas go negative
        delta_indices = [12,13,14,15,16,17,18,19,20,
                         33,34,35,36,37,38,39,40,41,
                         54,55,56,57,58,59,60,61,62]
        deltas = [obs[i] for i in delta_indices]
        assert any(abs(d) > 1e-6 for d in deltas), \
            f"All deltas zero after load change — EMA not updating. deltas={deltas}"
    print("  PASS  test_ema_updates")


# ---------------------------------------------------------------------------
# Test 6 — Dueling DQN: structure, output shape, Q = V + (adv - mean(adv))
# ---------------------------------------------------------------------------

def test_dqn_model():
    from dqn_model import DQN
    import config

    model = DQN()

    assert hasattr(model, "trunk"),          "missing trunk"
    assert hasattr(model, "value_head"),     "missing value_head"
    assert hasattr(model, "advantage_head"), "missing advantage_head"

    batch = torch.randn(8, config.STATE_DIM)
    out = model(batch)
    assert out.shape == (8, config.ACTION_DIM), f"output shape wrong: {out.shape}"
    assert not torch.any(torch.isnan(out)),     "output contains NaN"

    # Verify the dueling combination: Q = V + (adv - mean(adv))
    with torch.no_grad():
        features = model.trunk(batch)
        V   = model.value_head(features)
        adv = model.advantage_head(features)
        Q_expected = V + (adv - adv.mean(dim=1, keepdim=True))
    assert torch.allclose(out, Q_expected, atol=1e-5), "Q != V + (adv - mean(adv))"

    print(f"  PASS  test_dqn_model  (in={config.STATE_DIM}, out={config.ACTION_DIM}, dueling verified)")


# ---------------------------------------------------------------------------
# Test 7 — DQNAgent selects valid actions
# ---------------------------------------------------------------------------

def test_agent_action_selection():
    from dqn_agent import DQNAgent
    import config
    agent = DQNAgent()
    state = np.random.uniform(-1.0, 1.0, config.STATE_DIM).astype(np.float32)
    action = agent.select_action(state)
    assert action in [0, 1, 2], f"Unexpected action: {action}"
    action_eval = agent.select_action(state, eval_mode=True)
    assert action_eval in [0, 1, 2], f"Unexpected eval action: {action_eval}"
    print("  PASS  test_agent_action_selection")


# ---------------------------------------------------------------------------
# Test 8 — optimize_model runs, returns finite loss; uses Double DQN target
# ---------------------------------------------------------------------------

def test_agent_optimization():
    from dqn_agent import DQNAgent
    import config
    agent = DQNAgent()

    # Rewards in [-1, 0] to match the new reward function
    for _ in range(config.MIN_REPLAY_SIZE + 5):
        s  = np.random.uniform(-1.0, 1.0, config.STATE_DIM).astype(np.float32)
        ns = np.random.uniform(-1.0, 1.0, config.STATE_DIM).astype(np.float32)
        r  = float(np.random.uniform(-1.0, 0.0))
        agent.memory.push(s, np.random.randint(3), r, ns, False)

    loss = agent.optimize_model()
    assert isinstance(loss, float), f"optimize_model returned {type(loss)}"
    assert not np.isnan(loss),      "loss is NaN"
    assert loss >= 0.0,             f"Huber loss must be non-negative, got {loss}"
    print(f"  PASS  test_agent_optimization  (loss={loss:.6f})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_obs_space,
        test_reset,
        test_reward_range,
        test_10_episodes,
        test_ema_updates,
        test_dqn_model,
        test_agent_action_selection,
        test_agent_optimization,
    ]

    failed = []
    for t in tests:
        if "lbnn_env" in sys.modules:
            del sys.modules["lbnn_env"]
        try:
            t()
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed.append(t.__name__)

    print()
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All smoke tests passed.")
