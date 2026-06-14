"""
Generate dissertation figures from training metrics and evaluation results.

Requirements:
    pip install matplotlib     (numpy is already in the venv)

Usage (from Agent/ directory):
    python plot_results.py
    python plot_results.py --trace 2           # different trace for time-series plots
    python plot_results.py --out-dir figures   # custom output directory

Chapter 4 figures require Models/metrics/training_metrics.csv.
Chapter 6 figures require eval/results/ to be populated (run: python evaluate.py run).

Outputs:
    eval/figures/ch4/  — training curves
    eval/figures/ch6/  — evaluation figures
"""

import csv
import glob
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

# DQN is last so it is plotted on top of all other lines in every figure.
METHODS = ["random", "round_robin", "least_connections", "least_load", "greedy_min_peak", "dqn"]

METHOD_LABELS = {
    "random":            "Random",
    "round_robin":       "Round Robin",
    "least_connections": "Least Connections",
    "least_load":        "Least Load",
    "greedy_min_peak":   "Greedy Min-Peak",
    "dqn":               "DQN",
}

COLORS = {
    "random":            "#aaaaaa",
    "round_robin":       "#8ecae6",
    "least_connections": "#219ebc",
    "least_load":        "#fb8500",
    "greedy_min_peak":   "#e63946",
    "dqn":               "#2d6a4f",
}

# Line widths — DQN gets a thicker line to stand out on top.
LINEWIDTHS = {m: 1.5 for m in METHODS}
LINEWIDTHS["dqn"] = 2.8

SERVERS = ["s1", "s2", "s3"]
PEAK_KEYS = ["s1_cpu", "s1_mem", "s2_cpu", "s2_mem", "s3_cpu", "s3_mem"]

SERVER_LABELS = {
    "s1": "Server 1  (small — 1 500 CPU / 2 000 RAM)",
    "s2": "Server 2  (medium — 3 500 CPU / 3 200 RAM)",
    "s3": "Server 3  (large — 5 000 CPU / 3 000 RAM)",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _peak(row):
    return max(float(row[k]) for k in PEAK_KEYS) / 100.0


def _trace_files(method, results_dir):
    return sorted(glob.glob(os.path.join(results_dir, method, "trace_*.csv")))


def load_all_peaks(method, results_dir):
    out = []
    for f in _trace_files(method, results_dir):
        with open(f) as fp:
            for row in csv.DictReader(fp):
                if row["phase"] == "measured":
                    out.append(_peak(row))
    return np.array(out)


def load_all_actions(method, results_dir):
    out = []
    for f in _trace_files(method, results_dir):
        with open(f) as fp:
            for row in csv.DictReader(fp):
                if row["phase"] == "measured":
                    out.append(int(row["action"]))
    return np.array(out)


def load_one_trace_peaks(method, results_dir, trace_id, phase="measured"):
    """phase="measured" for analysis-only; phase=None for all ticks including warmup."""
    f = os.path.join(results_dir, method, f"trace_{trace_id}.csv")
    if not os.path.exists(f):
        return np.array([]), np.array([])
    ticks, peaks = [], []
    with open(f) as fp:
        for row in csv.DictReader(fp):
            if phase is not None and row["phase"] != phase:
                continue
            ticks.append(int(row["tick"]))
            peaks.append(_peak(row))
    return np.array(ticks), np.array(peaks)


def load_one_trace_server_util(method, results_dir, trace_id, phase="measured"):
    """Returns (ticks, {s1_cpu, s1_mem, ...}) normalised to [0,1].
    phase=None loads warmup + measured."""
    f = os.path.join(results_dir, method, f"trace_{trace_id}.csv")
    if not os.path.exists(f):
        return np.array([]), {}
    ticks = []
    cols  = {k: [] for k in PEAK_KEYS}
    with open(f) as fp:
        for row in csv.DictReader(fp):
            if phase is not None and row["phase"] != phase:
                continue
            ticks.append(int(row["tick"]))
            for k in PEAK_KEYS:
                cols[k].append(float(row[k]) / 100.0)
    return np.array(ticks), {k: np.array(v) for k, v in cols.items()}


def load_paired_peaks(method_a, method_b, results_dir):
    files_a = _trace_files(method_a, results_dir)
    files_b = _trace_files(method_b, results_dir)
    pa, pb  = [], []

    def _read_peaks(f):
        out = []
        with open(f) as fp:
            for row in csv.DictReader(fp):
                if row["phase"] == "measured":
                    out.append(_peak(row))
        return out

    for fa, fb in zip(files_a, files_b):
        a, b = _read_peaks(fa), _read_peaks(fb)
        n    = min(len(a), len(b))
        pa.extend(a[:n])
        pb.extend(b[:n])
    return np.array(pa), np.array(pb)


def _gini3(u):
    v = sorted(u)
    s = sum(v)
    if s < 1e-10:
        return 0.0
    return (2 * (v[0] + 2 * v[1] + 3 * v[2]) - 4 * s) / (3 * s)


_ROLLING_W = 30   # must match evaluate.py ROLLING_STD_W


def compute_all_trace_metrics(method, results_dir, trace_id):
    """Compute all 9 evaluation metrics for one trace (measured phase only)."""
    f = os.path.join(results_dir, method, f"trace_{trace_id}.csv")
    if not os.path.exists(f):
        return None
    peaks, utils_list = [], []
    server_utils = [[], [], []]
    with open(f) as fp:
        for row in csv.DictReader(fp):
            if row["phase"] != "measured":
                continue
            u = [max(float(row[f"s{i+1}_cpu"]), float(row[f"s{i+1}_mem"])) / 100.0
                 for i in range(3)]
            peaks.append(max(u))
            utils_list.append(u)
            for i in range(3):
                server_utils[i].append(u[i])
    if not peaks:
        return None
    peaks = np.array(peaks)
    su    = [np.array(v) for v in server_utils]

    jitter       = float(np.mean([np.mean(np.abs(np.diff(v))) for v in su]))
    temporal_std = float(np.mean([np.std(v) for v in su]))
    rolling_parts = []
    for v in su:
        if len(v) >= _ROLLING_W:
            windows = np.lib.stride_tricks.sliding_window_view(v, _ROLLING_W)
            rolling_parts.append(float(np.mean(np.std(windows, axis=1))))
    rolling_std = float(np.mean(rolling_parts)) if rolling_parts else float("nan")

    return {
        "mean_peak":    float(peaks.mean()),
        "max_peak":     float(peaks.max()),
        "p95_peak":     float(np.percentile(peaks, 95)),
        "frac_over_90": float((peaks > 0.9).mean()),
        "gini":         float(np.mean([_gini3(u) for u in utils_list])),
        "util_std":     float(np.mean([np.std(u) for u in utils_list])),
        "jitter":       jitter,
        "temporal_std": temporal_std,
        "rolling_std":  rolling_std,
    }


def load_training_csv(path):
    if not os.path.exists(path):
        return None
    cols = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def smooth(values, window=50):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Chapter 4 — Training curves
# ---------------------------------------------------------------------------

def plot_training_reward(train, out_dir):
    episodes = train["episode"]
    rewards  = train["average_reward"]
    fig, ax  = plt.subplots(figsize=(11, 4))
    ax.plot(episodes, rewards, color="#cccccc", alpha=0.5, linewidth=0.7, label="raw")
    if len(rewards) >= 50:
        sm = smooth(rewards, 50)
        ax.plot(episodes[49:], sm, color="#2d6a4f", linewidth=2, label="smoothed (50-ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, "ch4", "training_reward.png"))


def plot_epsilon_decay(train, out_dir):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(train["episode"], train["epsilon"], color="#e63946", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε (Epsilon)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Epsilon Decay over Training")
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, "ch4", "epsilon_decay.png"))


def plot_training_fairness(train, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    episodes  = train["episode"]
    specs = [
        ("gini_coefficient", "Gini Coefficient", "#219ebc"),
        ("utilization_std",  "Utilization Std",  "#fb8500"),
    ]
    for ax, (col, label, color) in zip(axes, specs):
        if col not in train:
            continue
        vals = train[col]
        ax.plot(episodes, vals, color="#cccccc", alpha=0.5, linewidth=0.7)
        if len(vals) >= 50:
            ax.plot(episodes[49:], smooth(vals, 50), color=color,
                    linewidth=2, label="smoothed")
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Episode")
    fig.suptitle("Training Fairness Metrics")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "ch4", "training_fairness.png"))


# ---------------------------------------------------------------------------
# Chapter 6 — Evaluation figures
# ---------------------------------------------------------------------------

WARMUP_END = 150   # last warmup tick; measured starts at tick 151


def plot_peak_timeseries(results_dir, out_dir, trace_id=1, window=30):
    fig, ax = plt.subplots(figsize=(14, 4))
    # DQN first (behind), Greedy second (on top). Greedy linewidth = 1/3 of DQN.
    greedy_lw = LINEWIDTHS["dqn"] / 3
    for method in ("dqn", "greedy_min_peak"):
        lw = LINEWIDTHS["dqn"] if method == "dqn" else greedy_lw
        # Load all ticks (warmup + measured).
        ticks, peaks = load_one_trace_peaks(method, results_dir, trace_id, phase=None)
        if len(peaks) == 0:
            continue
        ax.plot(ticks, peaks, color=COLORS[method], alpha=0.15, linewidth=0.7)
        if len(peaks) >= window:
            ax.plot(ticks[window - 1:], smooth(peaks, window),
                    color=COLORS[method], linewidth=lw,
                    label=f"{METHOD_LABELS[method]}  (smoothed {window}-tick)")
    ax.axvline(WARMUP_END, color="black", linestyle=":", linewidth=1.5, alpha=0.6,
               label="Warmup end")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Peak Server Load")
    ax.set_title(f"Peak Load Time Series — Trace {trace_id}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, "ch6", "peak", "peak_timeseries.png"))


def plot_peak_cdf(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Baselines first, DQN last (plotted on top).
    for method in METHODS:
        peaks = load_all_peaks(method, results_dir)
        if len(peaks) == 0:
            continue
        peaks_sorted = np.sort(peaks)
        cdf = np.arange(1, len(peaks_sorted) + 1) / len(peaks_sorted)
        ax.plot(peaks_sorted, cdf, color=COLORS[method],
                linewidth=LINEWIDTHS[method], label=METHOD_LABELS[method])

    ax.axvline(0.9, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label="90% threshold")
    ax.set_xlabel("Peak Server Load  (fraction of capacity)")
    ax.set_ylabel("Fraction of ticks at or below this load")
    ax.set_title(
        "CDF of Per-Tick Peak Load\n"
        "A curve shifted left means the method keeps peak load lower more often"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, "ch6", "peak", "peak_cdf.png"))


def plot_action_distribution(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    n = len(METHODS)
    x = np.arange(3)
    w = 0.13
    for i, method in enumerate(METHODS):
        actions = load_all_actions(method, results_dir)
        if len(actions) == 0:
            continue
        fracs = [(actions == a).mean() for a in range(3)]
        ax.bar(x + i * w, fracs, w, label=METHOD_LABELS[method],
               color=COLORS[method], alpha=0.85,
               zorder=3 if method == "dqn" else 2)
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(["Server 1\n(small)", "Server 2\n(medium)", "Server 3\n(large)"])
    ax.axhline(1 / 3, color="black", linestyle="--", linewidth=1, alpha=0.4, label="uniform 33%")
    ax.set_ylabel("Fraction of Requests Routed")
    ax.set_title("Action Distribution per Method")
    ax.grid(True, alpha=0.3, axis="y")
    # Legend below the plot so it never clips the bars.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=4, fontsize=9, framealpha=0.9)
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, os.path.join(out_dir, "ch6", "peak", "action_distribution.png"))


def plot_peak_boxplots(results_dir, out_dir):
    data, labels, colors = [], [], []
    for method in METHODS:
        peaks = load_all_peaks(method, results_dir)
        if len(peaks) == 0:
            continue
        data.append(peaks)
        labels.append(METHOD_LABELS[method])
        colors.append(COLORS[method])
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    flierprops={"marker": ".", "markersize": 1, "alpha": 0.3})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Per-Tick Peak Load")
    ax.set_title("Distribution of Peak Load per Method  (all traces)")
    ax.grid(True, alpha=0.3, axis="y")
    savefig(fig, os.path.join(out_dir, "ch6", "peak", "peak_boxplots.png"))


SMOOTH_W = 25  # smoothing window for server utilization plots


def _plot_server_metric(results_dir, out_dir, metric, trace_id,
                        tick_min=None, tick_max=None, part_label=""):
    """
    Plot CPU or MEM utilization across 3 servers for one trace.
    Faint raw signal + bold smoothed line, DQN on top of Greedy.
    tick_min / tick_max filter the tick range (inclusive).
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    greedy_lw = LINEWIDTHS["dqn"] / 3
    for ax, srv in zip(axes, SERVERS):
        col = f"{srv}_{metric}"
        # DQN first (behind), Greedy second (on top). Greedy linewidth = 1/3 of DQN.
        for method in ("dqn", "greedy_min_peak"):
            lw = LINEWIDTHS["dqn"] if method == "dqn" else greedy_lw
            ticks, cols = load_one_trace_server_util(method, results_dir, trace_id, phase=None)
            if len(ticks) == 0:
                continue
            vals = cols[col]
            # Apply tick range filter.
            mask = np.ones(len(ticks), dtype=bool)
            if tick_min is not None:
                mask &= ticks >= tick_min
            if tick_max is not None:
                mask &= ticks <= tick_max
            t = ticks[mask]
            v = vals[mask]
            if len(t) == 0:
                continue
            # Faint raw underlay.
            raw_alpha = 0.15 if method == "dqn" else 0.25
            ax.plot(t, v, color=COLORS[method], alpha=raw_alpha, linewidth=0.8)
            # Bold smoothed line (labeled for legend).
            if len(v) >= SMOOTH_W:
                sm = smooth(v, SMOOTH_W)
                ax.plot(t[SMOOTH_W - 1:], sm,
                        color=COLORS[method],
                        linewidth=lw,
                        alpha=1.0 if method == "dqn" else 0.75,
                        label=METHOD_LABELS[method])

        # Draw warmup boundary only when tick 150 falls inside the visible range.
        lo = tick_min if tick_min is not None else 1
        hi = tick_max if tick_max is not None else 99999
        if lo <= WARMUP_END <= hi:
            ax.axvline(WARMUP_END, color="black", linestyle=":", linewidth=1.5,
                       alpha=0.6, label="Warmup end" if srv == "s1" else None)
        ax.set_ylabel("Utilisation")
        ax.set_title(SERVER_LABELS[srv])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Tick")
    metric_name = "CPU" if metric == "cpu" else "Memory"
    if tick_min is not None and tick_max is not None:
        tick_range = f"ticks {tick_min}–{tick_max}"
    elif tick_min is not None:
        tick_range = f"ticks {tick_min}+"
    elif tick_max is not None:
        tick_range = f"ticks –{tick_max}"
    else:
        tick_range = "all ticks"
    fig.suptitle(
        f"Per-Server {metric_name} Utilisation — Trace {trace_id}  ({tick_range})\n"
        f"(faint = raw,  bold = {SMOOTH_W}-tick smoothed)",
        fontsize=12,
    )
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    suffix = f"_{part_label}" if part_label else ""
    savefig(fig, os.path.join(out_dir, "ch6", "server", f"server_{metric}_utilization{suffix}.png"))


def plot_server_utilization(results_dir, out_dir, trace_id=1):
    # Remove the old combined file that showed CPU+MEM on the same axes.
    old = os.path.join(out_dir, "ch6", "server", "server_utilization.png")
    if os.path.exists(old):
        os.remove(old)
        print(f"  Removed stale: {old}")

    # Midpoint tick: warmup ends at 150, measured runs 5000 ticks → split at 2650.
    mid = 2650
    for metric in ("cpu", "mem"):
        _plot_server_metric(results_dir, out_dir, metric, trace_id,
                            tick_max=mid,        part_label="p1")
        _plot_server_metric(results_dir, out_dir, metric, trace_id,
                            tick_min=mid + 1,    part_label="p2")


def plot_dqn_vs_greedy_scatter(results_dir, out_dir):
    dqn_peaks, greedy_peaks = load_paired_peaks("dqn", "greedy_min_peak", results_dir)
    if len(dqn_peaks) == 0:
        print("  Skipping scatter — missing data.")
        return
    dqn_better     = dqn_peaks < greedy_peaks
    dqn_win_pct    = dqn_better.mean() * 100
    greedy_win_pct = 100 - dqn_win_pct

    fig, ax = plt.subplots(figsize=(7, 7))
    # Greedy-wins points first, DQN-wins points on top.
    ax.scatter(greedy_peaks[~dqn_better], dqn_peaks[~dqn_better],
               s=1, alpha=0.2, color=COLORS["greedy_min_peak"],
               label=f"Greedy better  ({greedy_win_pct:.1f}%)")
    ax.scatter(greedy_peaks[dqn_better], dqn_peaks[dqn_better],
               s=1, alpha=0.2, color=COLORS["dqn"],
               label=f"DQN better  ({dqn_win_pct:.1f}%)")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Equal")
    ax.set_xlabel("Greedy Min-Peak  (peak load)")
    ax.set_ylabel("DQN  (peak load)")
    ax.set_title("Per-Tick: DQN vs Greedy Min-Peak\n(below diagonal = DQN wins that tick)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(markerscale=6, fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, "ch6", "peak", "dqn_vs_greedy_scatter.png"))


def plot_per_metric_comparison(results_dir, out_dir):
    """One scatter per metric: each dot = one trace, x = Greedy, y = DQN, 45° diagonal."""
    trace_ids = sorted(
        int(os.path.basename(f).replace("trace_", "").replace(".csv", ""))
        for f in glob.glob(os.path.join(results_dir, "dqn", "trace_*.csv"))
    )
    if not trace_ids:
        print("  Skipping per-metric comparison — no DQN trace files found.")
        return

    dqn_vals    = {k: [] for k in _ALL_METRICS}
    greedy_vals = {k: [] for k in _ALL_METRICS}
    for tid in trace_ids:
        dm = compute_all_trace_metrics("dqn",             results_dir, tid)
        gm = compute_all_trace_metrics("greedy_min_peak", results_dir, tid)
        for k in _ALL_METRICS:
            dqn_vals[k].append(dm[k] if dm else float("nan"))
            greedy_vals[k].append(gm[k] if gm else float("nan"))

    for key in _ALL_METRICS:
        dv = np.array(dqn_vals[key])
        gv = np.array(greedy_vals[key])

        dqn_better  = dv < gv
        n_dqn_wins  = int(np.sum(dqn_better))
        dqn_win_pct    = n_dqn_wins / len(trace_ids) * 100
        greedy_win_pct = 100 - dqn_win_pct

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(gv[~dqn_better], dv[~dqn_better],
                   s=18, alpha=0.6, color=COLORS["greedy_min_peak"],
                   label=f"Greedy better  ({greedy_win_pct:.1f}%)", zorder=3)
        ax.scatter(gv[dqn_better], dv[dqn_better],
                   s=18, alpha=0.6, color=COLORS["dqn"],
                   label=f"DQN better  ({dqn_win_pct:.1f}%)", zorder=3)

        # 45° diagonal.
        lo = min(np.nanmin(dv), np.nanmin(gv))
        hi = max(np.nanmax(dv), np.nanmax(gv))
        pad = (hi - lo) * 0.05 if hi > lo else 0.01
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, "k--", linewidth=1, label="Equal", zorder=2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(f"Greedy Min-Peak  —  {_METRIC_LABELS[key]}")
        ax.set_ylabel(f"DQN  —  {_METRIC_LABELS[key]}")
        ax.set_title(
            f"{_METRIC_LABELS[key]}  —  DQN vs Greedy Min-Peak\n"
            f"(below diagonal = DQN wins  |  {n_dqn_wins}/{len(trace_ids)} traces)",
            fontsize=11,
        )
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()
        savefig(fig, os.path.join(out_dir, "ch6", "per_metric", f"per_metric_{key}.png"))


# Shared config for the two paired-diff figures.
_ALL_METRICS   = [
    "mean_peak", "max_peak", "p95_peak", "frac_over_90",
    "gini", "util_std",
    "jitter", "temporal_std", "rolling_std",
]
_METRIC_LABELS = {
    "mean_peak":    "Mean Peak",
    "max_peak":     "Max Peak",
    "p95_peak":     "P95 Peak",
    "frac_over_90": "Frac > 90%",
    "gini":         "Gini",
    "util_std":     "Util Std",
    "jitter":       "Jitter",
    "temporal_std": "Temporal Std",
    "rolling_std":  f"Rolling Std (w={_ROLLING_W})",
}
_BASELINES = [
    ("least_load",        "DQN vs Least Load"),
    ("greedy_min_peak",   "DQN vs Greedy Min-Peak"),
]


def plot_volatility_comparison(results_dir, out_dir):
    """Grouped bar chart of jitter / temporal_std / rolling_std across all 6 methods."""
    vol_metrics = [
        ("jitter",       f"Jitter  (mean |Δu|)"),
        ("temporal_std", "Temporal Std  (std of u over episode)"),
        ("rolling_std",  f"Rolling Std  (mean of {_ROLLING_W}-tick window std)"),
    ]
    trace_ids = sorted(
        int(os.path.basename(f).replace("trace_", "").replace(".csv", ""))
        for f in glob.glob(os.path.join(results_dir, "dqn", "trace_*.csv"))
    )

    # Collect mean ± std across traces for each method × metric.
    method_means = {m: {} for m in METHODS}
    method_stds  = {m: {} for m in METHODS}
    for method in METHODS:
        vals_per_metric = {k: [] for k, _ in vol_metrics}
        for tid in trace_ids:
            m = compute_all_trace_metrics(method, results_dir, tid)
            if m is None:
                continue
            for k, _ in vol_metrics:
                vals_per_metric[k].append(m[k])
        for k, _ in vol_metrics:
            v = vals_per_metric[k]
            method_means[method][k] = float(np.mean(v))  if v else float("nan")
            method_stds[method][k]  = float(np.std(v))   if v else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    x     = np.arange(len(METHODS))
    width = 0.65

    for ax, (key, label) in zip(axes, vol_metrics):
        means  = [method_means[m][key] for m in METHODS]
        stds   = [method_stds[m][key]  for m in METHODS]
        colors = [COLORS[m] for m in METHODS]
        bars   = ax.bar(x, means, width, color=colors, alpha=0.85,
                        yerr=stds, error_kw={"elinewidth": 1.2, "capsize": 4,
                                              "ecolor": "#333333"})
        # Highlight DQN bar with a border.
        dqn_idx = METHODS.index("dqn")
        bars[dqn_idx].set_edgecolor("black")
        bars[dqn_idx].set_linewidth(1.8)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Temporal Volatility Metrics — All Methods  (mean ± std across {len(trace_ids)} traces)\n"
        "lower = smoother utilisation",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "ch6", "summary", "volatility_comparison.png"))


def plot_summary_heatmap(summary_dir, out_dir):
    """Compact 2×9 heatmap of mean diffs from paired_diffs.csv — dissertation-friendly."""
    path = os.path.join(summary_dir, "paired_diffs.csv")
    if not os.path.exists(path):
        print("  Skipping summary heatmap — paired_diffs.csv not found.")
        return

    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[(row["comparison"], row["metric"])] = float(row["mean_diff"])

    comparisons = [k for k, _ in _BASELINES]
    comp_labels  = [v for _, v in _BASELINES]
    matrix = np.array([
        [data.get((f"dqn_vs_{comp}", m), float("nan")) for m in _ALL_METRICS]
        for comp in comparisons
    ])

    # Symmetric colour scale centred on 0.
    finite = np.abs(matrix[~np.isnan(matrix)])
    vmax   = float(finite.max()) if len(finite) > 0 else 0.05
    vmax   = vmax if vmax > 1e-6 else 0.05

    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(_ALL_METRICS)))
    ax.set_xticklabels([_METRIC_LABELS[m] for m in _ALL_METRICS],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(comparisons)))
    ax.set_yticklabels(comp_labels, fontsize=10)

    for i in range(len(comparisons)):
        for j, m in enumerate(sorted(_ALL_METRICS, key=_ALL_METRICS.index)):
            v = matrix[i, j]
            if not np.isnan(v):
                sign  = "▼" if v < 0 else "▲"
                color = "black"
                ax.text(j, i, f"{sign} {abs(v):.4f}", ha="center", va="center",
                        fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, fraction=0.03)
    cbar.set_label("Mean diff  (DQN − baseline)\nnegative = DQN wins", fontsize=8)

    ax.set_title(
        "Summary: DQN vs Baselines — mean metric difference across all traces\n"
        "Green = DQN wins  |  Red = baseline wins  |  ▼ / ▲ = direction",
        fontsize=10,
    )
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "ch6", "summary", "summary_heatmap.png"))


def plot_paired_diffs_diverging(summary_dir, out_dir):
    """
    Horizontal diverging bar chart.
    Each bar = mean(DQN - baseline) across all traces, error bar = std.
    Negative (left) = DQN better. Annotated with win count.
    """
    from matplotlib.patches import Patch

    path = os.path.join(summary_dir, "paired_diffs.csv")
    if not os.path.exists(path):
        print("  Skipping diverging diffs — paired_diffs.csv not found.")
        return

    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["comparison"], row["metric"])
            rows[key] = {
                "mean_diff": float(row["mean_diff"]),
                "std_diff":  float(row["std_diff"]),
                "dqn_wins":  int(row["dqn_wins"]),
                "n_traces":  int(row["n_traces"]),
            }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    y = np.arange(len(_ALL_METRICS))

    for ax, (comp_key, comp_label) in zip(axes, _BASELINES):
        full_key = f"dqn_vs_{comp_key}"
        for i, metric in enumerate(_ALL_METRICS):
            r = rows.get((full_key, metric))
            if r is None:
                continue
            diff  = r["mean_diff"]
            std   = r["std_diff"]
            wins  = r["dqn_wins"]
            n     = r["n_traces"]
            color = COLORS["dqn"] if diff < 0 else COLORS["greedy_min_peak"]
            ax.barh(i, diff, xerr=std, color=color, alpha=0.82,
                    error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "#333333"})
            # Win-count label just outside the bar tip.
            x_label = diff - std * 0.15 if diff < 0 else diff + std * 0.15
            ha = "right" if diff < 0 else "left"
            ax.text(x_label, i, f" {wins}/{n} ", va="center", ha=ha, fontsize=8.5,
                    color="black")

        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([_METRIC_LABELS[m] for m in _ALL_METRICS])
        ax.set_title(comp_label, fontweight="bold", fontsize=11)
        ax.set_xlabel("DQN − baseline\n← negative = DQN better", fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")

    legend_handles = [
        Patch(facecolor=COLORS["dqn"],            alpha=0.82, label="DQN better"),
        Patch(facecolor=COLORS["greedy_min_peak"], alpha=0.82, label="Baseline better"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    # n_traces comes from the CSV — use it directly.
    n_traces_label = next(
        (r["n_traces"] for r in rows.values() if r["n_traces"] > 0), "?"
    )
    fig.suptitle(
        f"Paired Differences: DQN vs Baselines  (mean ± std across {n_traces_label} traces)\n"
        "win count = traces where DQN wins that metric",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    savefig(fig, os.path.join(out_dir, "ch6", "summary", "paired_diffs_diverging.png"))


def _render_win_heatmap(matrices, trace_ids, out_path, chunk_label=""):
    """Render a win/loss heatmap for a subset of traces and save to out_path."""
    n_metrics = len(_ALL_METRICS)
    n_traces  = len(trace_ids)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, n_traces * 1.5), 9))

    for ax, (baseline, title) in zip(axes, _BASELINES):
        matrix = matrices[baseline]
        vmax = float(np.nanmax(np.abs(matrix))) if not np.all(np.isnan(matrix)) else 1.0
        vmax = vmax or 1.0
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")

        for i in range(n_metrics):
            for j in range(n_traces):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                txt_color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i, f"{val:+.4f}", ha="center", va="center",
                        fontsize=max(5, min(8, 60 // n_traces)),
                        color=txt_color, fontweight="bold")

        ax.set_xticks(range(n_traces))
        ax.set_xticklabels([f"T{t}" for t in trace_ids], fontsize=max(6, min(9, 60 // n_traces)))
        ax.set_yticks(range(n_metrics))
        ax.set_yticklabels([_METRIC_LABELS[m] for m in _ALL_METRICS], fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=11)
        plt.colorbar(im, ax=ax, label="DQN − baseline  (green = DQN wins)", shrink=0.85)

    suffix = f"  —  traces {chunk_label}" if chunk_label else ""
    fig.suptitle(
        f"Per-Trace Win/Loss: DQN vs Baselines{suffix}\n"
        "green = DQN better · red = baseline better · value = exact difference",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_path)


def plot_per_trace_win_heatmap(results_dir, out_dir):
    """Full heatmap (all traces) + 20 chunked heatmaps of 5 traces each."""
    trace_ids = sorted(
        int(os.path.basename(f).replace("trace_", "").replace(".csv", ""))
        for f in glob.glob(os.path.join(results_dir, "dqn", "trace_*.csv"))
    )
    if not trace_ids:
        print("  Skipping win heatmap — no DQN trace files found.")
        return

    # Pre-compute all metrics once to avoid re-reading CSVs for every chunk.
    all_dqn  = {}
    all_base = {b: {} for b, _ in _BASELINES}
    for tid in trace_ids:
        all_dqn[tid] = compute_all_trace_metrics("dqn", results_dir, tid)
        for baseline, _ in _BASELINES:
            all_base[baseline][tid] = compute_all_trace_metrics(baseline, results_dir, tid)

    def _build_matrices(tids):
        mats = {}
        for baseline, _ in _BASELINES:
            m = np.full((len(_ALL_METRICS), len(tids)), np.nan)
            for j, tid in enumerate(tids):
                dm = all_dqn.get(tid)
                bm = all_base[baseline].get(tid)
                if dm is None or bm is None:
                    continue
                for i, metric in enumerate(_ALL_METRICS):
                    m[i, j] = dm[metric] - bm[metric]
            mats[baseline] = m
        return mats

    hmap_dir   = os.path.join(out_dir, "ch6", "heatmaps")
    chunks_dir = os.path.join(hmap_dir, "chunks")

    # Full heatmap.
    _render_win_heatmap(_build_matrices(trace_ids), trace_ids,
                        os.path.join(hmap_dir, "per_trace_win_heatmap.png"))

    # Chunked heatmaps — 5 traces per figure.
    chunk_size = 5
    for start in range(0, len(trace_ids), chunk_size):
        chunk = trace_ids[start:start + chunk_size]
        label = f"{chunk[0]}–{chunk[-1]}"
        fname = f"heatmap_{chunk[0]}_{chunk[-1]}.png"
        _render_win_heatmap(_build_matrices(chunk), chunk,
                            os.path.join(chunks_dir, fname),
                            chunk_label=label)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate dissertation figures.")
    parser.add_argument("--training-csv",  default="Models/metrics/training_metrics.csv")
    parser.add_argument("--results-dir",   default="eval/results")
    parser.add_argument("--out-dir",       default="eval/figures")
    parser.add_argument("--trace",         type=int, default=1,
                        help="Trace ID for time-series and per-server plots (default: 1)")
    args = parser.parse_args()

    print("=== Chapter 4: Training Curves ===")
    train = load_training_csv(args.training_csv)
    if train is not None:
        plot_training_reward(train, args.out_dir)
        plot_epsilon_decay(train, args.out_dir)
        plot_training_fairness(train, args.out_dir)
    else:
        print(f"  WARNING: {args.training_csv} not found — skipping Chapter 4 plots.")

    print("\n=== Chapter 6: Evaluation Figures ===")
    plot_peak_timeseries(args.results_dir, args.out_dir, trace_id=args.trace)
    plot_peak_cdf(args.results_dir, args.out_dir)
    plot_action_distribution(args.results_dir, args.out_dir)
    plot_peak_boxplots(args.results_dir, args.out_dir)
    plot_server_utilization(args.results_dir, args.out_dir, trace_id=args.trace)
    plot_dqn_vs_greedy_scatter(args.results_dir, args.out_dir)
    plot_per_metric_comparison(args.results_dir, args.out_dir)
    summary_dir = os.path.join(args.results_dir, "..", "summary")
    plot_summary_heatmap(summary_dir, args.out_dir)
    plot_paired_diffs_diverging(summary_dir, args.out_dir)
    plot_per_trace_win_heatmap(args.results_dir, args.out_dir)
    plot_volatility_comparison(args.results_dir, args.out_dir)

    print(f"\nDone. Folder layout:")
    print(f"  {args.out_dir}/ch4/               training curves")
    print(f"  {args.out_dir}/ch6/peak/           peak timeseries, CDF, boxplots, scatter, actions")
    print(f"  {args.out_dir}/ch6/server/         per-server CPU & MEM utilisation")
    print(f"  {args.out_dir}/ch6/per_metric/     DQN vs Greedy per metric (9 figures)")
    print(f"  {args.out_dir}/ch6/summary/        summary heatmap, diverging bars, volatility")
    print(f"  {args.out_dir}/ch6/heatmaps/       full win/loss heatmap")
    print(f"  {args.out_dir}/ch6/heatmaps/chunks/  20 × 5-trace heatmaps")


if __name__ == "__main__":
    main()
