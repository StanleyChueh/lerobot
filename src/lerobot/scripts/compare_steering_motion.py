
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(np.float64)


def to_scalar(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().reshape(-1)[0].item()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.reshape(-1)[0].item()
        return x.reshape(-1)[0].item()
    return x


def safe_median_positive(x, default=1.0 / 30.0):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return default
    return float(np.median(x))


def resample_trace(trace, n_bins=100):
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) == 0:
        return np.full(n_bins, np.nan, dtype=np.float64)
    if len(trace) == 1:
        return np.full(n_bins, trace[0], dtype=np.float64)

    old_x = np.linspace(0.0, 1.0, len(trace))
    new_x = np.linspace(0.0, 1.0, n_bins)
    return np.interp(new_x, old_x, trace)


def compute_signal_metrics(signal, timestamps, prefix):
    out = {}

    if signal is None:
        return out

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]

    T = signal.shape[0]
    if T < 2:
        out[f"{prefix}_path_len"] = 0.0
        out[f"{prefix}_mean_speed"] = 0.0
        out[f"{prefix}_p95_speed"] = 0.0
        out[f"{prefix}_max_speed"] = 0.0
        out[f"{prefix}_mean_jerk"] = 0.0
        out[f"{prefix}_p95_jerk"] = 0.0
        out[f"{prefix}_speed_spikiness"] = 0.0
        out[f"{prefix}_endpoint_delta"] = 0.0
        out[f"{prefix}_trace_speed"] = np.array([0.0], dtype=np.float64)
        out[f"{prefix}_trace_jerk"] = np.array([0.0], dtype=np.float64)
        return out

    timestamps = np.asarray(timestamps, dtype=np.float64)
    dt = np.diff(timestamps)
    fallback_dt = safe_median_positive(dt, default=1.0 / 30.0)
    dt = np.where((dt > 0) & np.isfinite(dt), dt, fallback_dt)

    d1 = np.diff(signal, axis=0)
    step_norm = np.linalg.norm(d1, axis=1)
    speed = step_norm / dt

    if len(speed) >= 2:
        ds = np.diff(speed)
        dt2 = dt[1:]
        dt2 = np.where((dt2 > 0) & np.isfinite(dt2), dt2, fallback_dt)
        jerk = np.abs(ds) / dt2
    else:
        jerk = np.array([0.0], dtype=np.float64)

    median_speed = float(np.median(speed)) if len(speed) > 0 else 0.0
    out[f"{prefix}_path_len"] = float(np.sum(step_norm))
    out[f"{prefix}_mean_speed"] = float(np.mean(speed))
    out[f"{prefix}_p95_speed"] = float(np.percentile(speed, 95))
    out[f"{prefix}_max_speed"] = float(np.max(speed))
    out[f"{prefix}_mean_jerk"] = float(np.mean(jerk))
    out[f"{prefix}_p95_jerk"] = float(np.percentile(jerk, 95))
    out[f"{prefix}_speed_spikiness"] = float(np.max(speed) / (median_speed + 1e-8))
    out[f"{prefix}_endpoint_delta"] = float(np.linalg.norm(signal[-1] - signal[0]))
    out[f"{prefix}_trace_speed"] = speed
    out[f"{prefix}_trace_jerk"] = jerk
    return out


def bootstrap_mean_diff_ci(a, b, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)

    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = bb.mean() - aa.mean()

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return np.nan

    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return np.nan
    return float((b.mean() - a.mean()) / math.sqrt(pooled))


def load_episodes(repo_id, root=None, max_frames=None):
    ds = LeRobotDataset(repo_id, root=root)

    episodes = {}
    inferred_episode = 0
    prev_ts = None

    n = len(ds)
    if max_frames is not None:
        n = min(n, max_frames)

    for i in range(n):
        sample = ds[i]

        ts = to_scalar(sample.get("timestamp", i))
        ts = float(ts) if ts is not None else float(i)

        ep = sample.get("episode_index", None)
        if ep is not None:
            ep = int(to_scalar(ep))
        else:
            if prev_ts is not None and ts < prev_ts:
                inferred_episode += 1
            ep = inferred_episode
        prev_ts = ts

        action = to_numpy(sample.get("action"))
        state = to_numpy(sample.get("observation.state"))

        if ep not in episodes:
            episodes[ep] = {
                "timestamps": [],
                "actions": [],
                "states": [],
            }

        episodes[ep]["timestamps"].append(ts)
        if action is not None:
            episodes[ep]["actions"].append(action.reshape(-1))
        if state is not None:
            episodes[ep]["states"].append(state.reshape(-1))

    cleaned = {}
    for ep_idx, data in episodes.items():
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64) if len(data["actions"]) > 0 else None
        states = np.asarray(data["states"], dtype=np.float64) if len(data["states"]) > 0 else None

        cleaned[ep_idx] = {
            "timestamps": timestamps,
            "actions": actions,
            "states": states,
        }

    return cleaned


def compute_episode_metrics(ep_idx, episode):
    timestamps = episode["timestamps"]
    actions = episode["actions"]
    states = episode["states"]

    out = {
        "episode_index": ep_idx,
        "num_frames": int(len(timestamps)),
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0,
    }

    out.update(compute_signal_metrics(actions, timestamps, prefix="action"))
    out.update(compute_signal_metrics(states, timestamps, prefix="state"))
    return out


def summarize_metrics(rows, dataset_name):
    metric_names = [
        "duration_s",
        "action_path_len",
        "action_mean_speed",
        "action_p95_speed",
        "action_max_speed",
        "action_mean_jerk",
        "action_p95_jerk",
        "action_speed_spikiness",
        "state_path_len",
        "state_mean_speed",
        "state_p95_speed",
        "state_max_speed",
        "state_mean_jerk",
        "state_p95_jerk",
        "state_speed_spikiness",
        "state_endpoint_delta",
    ]

    summary = {}
    for name in metric_names:
        vals = np.array([r[name] for r in rows if name in r], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        summary[name] = {
            "dataset": dataset_name,
            "n_episodes": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
        }
    return summary


def compare_summaries(baseline_rows, steered_rows):
    metric_names = sorted(set().union(*[set(r.keys()) for r in baseline_rows + steered_rows]))
    ignore = {
        "episode_index",
        "num_frames",
        "action_trace_speed",
        "action_trace_jerk",
        "state_trace_speed",
        "state_trace_jerk",
    }

    comparisons = []
    for name in metric_names:
        if name in ignore:
            continue

        a = np.array(
            [r[name] for r in baseline_rows if name in r and is_finite_scalar(r[name])],
            dtype=np.float64,
        )
        b = np.array(
            [r[name] for r in steered_rows if name in r and is_finite_scalar(r[name])],
            dtype=np.float64,
        )

        if len(a) == 0 or len(b) == 0:
            continue

        ci_lo, ci_hi = bootstrap_mean_diff_ci(a, b, n_boot=3000, seed=42)
        comparisons.append({
            "metric": name,
            "baseline_mean": float(a.mean()),
            "steered_mean": float(b.mean()),
            "abs_diff": float(b.mean() - a.mean()),
            "ratio": float((b.mean() + 1e-8) / (a.mean() + 1e-8)),
            "cohens_d": cohens_d(a, b),
            "diff_ci95_lo": ci_lo,
            "diff_ci95_hi": ci_hi,
            "baseline_n": int(len(a)),
            "steered_n": int(len(b)),
        })

    return comparisons


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        path.write_text("")
        return

    fieldnames = sorted(set().union(*[set(r.keys()) for r in rows]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_boxplots(baseline_rows, steered_rows, out_path):
    metrics = [
        "action_mean_speed",
        "action_mean_jerk",
        "action_speed_spikiness",
        "state_mean_speed",
        "state_mean_jerk",
        "state_speed_spikiness",
    ]

    available = []
    for m in metrics:
        a = [r[m] for r in baseline_rows if m in r and np.isfinite(r[m])]
        b = [r[m] for r in steered_rows if m in r and np.isfinite(r[m])]
        if len(a) > 0 and len(b) > 0:
            available.append((m, a, b))

    if len(available) == 0:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 4 * len(available)))
    if len(available) == 1:
        axes = [axes]

    for ax, (metric, a, b) in zip(axes, available):
        ax.boxplot([a, b], labels=["baseline", "steered"])
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mean_trace(rows, prefix, out_path):
    speed_traces = []
    jerk_traces = []

    for r in rows:
        if f"{prefix}_trace_speed" in r:
            speed_traces.append(resample_trace(r[f"{prefix}_trace_speed"], n_bins=100))
        if f"{prefix}_trace_jerk" in r:
            jerk_traces.append(resample_trace(r[f"{prefix}_trace_jerk"], n_bins=100))

    if len(speed_traces) == 0 and len(jerk_traces) == 0:
        return None

    out = {}
    if len(speed_traces) > 0:
        speed_arr = np.vstack(speed_traces)
        out["speed_mean"] = np.nanmean(speed_arr, axis=0)
        out["speed_std"] = np.nanstd(speed_arr, axis=0)
    if len(jerk_traces) > 0:
        jerk_arr = np.vstack(jerk_traces)
        out["jerk_mean"] = np.nanmean(jerk_arr, axis=0)
        out["jerk_std"] = np.nanstd(jerk_arr, axis=0)
    return out


def plot_trace_comparison(baseline_rows, steered_rows, out_path):
    x = np.linspace(0.0, 1.0, 100)

    base_action = plot_mean_trace(baseline_rows, "action", None)
    steer_action = plot_mean_trace(steered_rows, "action", None)
    base_state = plot_mean_trace(baseline_rows, "state", None)
    steer_state = plot_mean_trace(steered_rows, "state", None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def draw(ax, base, steer, key, title):
        if base is None or steer is None or key not in base or key not in steer:
            ax.set_visible(False)
            return
        ax.plot(x, base[key], label="baseline")
        ax.plot(x, steer[key], label="steered")
        ax.fill_between(x, base[key] - base[key.replace("mean", "std")], base[key] + base[key.replace("mean", "std")], alpha=0.2)
        ax.fill_between(x, steer[key] - steer[key.replace("mean", "std")], steer[key] + steer[key.replace("mean", "std")], alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("normalized episode time")
        ax.grid(True, alpha=0.3)
        ax.legend()

    draw(axes[0, 0], base_action, steer_action, "speed_mean", "Action speed trace")
    draw(axes[0, 1], base_action, steer_action, "jerk_mean", "Action jerk trace")
    draw(axes[1, 0], base_state, steer_state, "speed_mean", "State speed trace")
    draw(axes[1, 1], base_state, steer_state, "jerk_mean", "State jerk trace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def is_finite_scalar(x):
    return np.isscalar(x) and np.isfinite(x)

def strip_trace_fields(rows):
    clean = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if k.endswith("_trace_speed") or k.endswith("_trace_jerk"):
                continue
            rr[k] = v
        clean.append(rr)
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-repo-id", type=str, required=True)
    parser.add_argument("--steered-repo-id", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="steering_compare")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline dataset: {args.baseline_repo_id}")
    baseline_eps = load_episodes(args.baseline_repo_id, root=args.root, max_frames=args.max_frames)

    print(f"Loading steered dataset: {args.steered_repo_id}")
    steered_eps = load_episodes(args.steered_repo_id, root=args.root, max_frames=args.max_frames)

    baseline_rows = [compute_episode_metrics(ep_idx, ep) for ep_idx, ep in sorted(baseline_eps.items())]
    steered_rows = [compute_episode_metrics(ep_idx, ep) for ep_idx, ep in sorted(steered_eps.items())]

    baseline_summary = summarize_metrics(baseline_rows, "baseline")
    steered_summary = summarize_metrics(steered_rows, "steered")
    comparisons = compare_summaries(baseline_rows, steered_rows)

    save_csv(out_dir / "baseline_episode_metrics.csv", strip_trace_fields(baseline_rows))
    save_csv(out_dir / "steered_episode_metrics.csv", strip_trace_fields(steered_rows))
    save_csv(out_dir / "comparison_summary.csv", comparisons)

    with (out_dir / "baseline_summary.json").open("w") as f:
        json.dump(baseline_summary, f, indent=2)
    with (out_dir / "steered_summary.json").open("w") as f:
        json.dump(steered_summary, f, indent=2)

    plot_metric_boxplots(baseline_rows, steered_rows, out_dir / "metric_boxplots.png")
    plot_trace_comparison(baseline_rows, steered_rows, out_dir / "trace_comparison.png")

    print("\nSaved:")
    print(f"  {out_dir / 'baseline_episode_metrics.csv'}")
    print(f"  {out_dir / 'steered_episode_metrics.csv'}")
    print(f"  {out_dir / 'comparison_summary.csv'}")
    print(f"  {out_dir / 'metric_boxplots.png'}")
    print(f"  {out_dir / 'trace_comparison.png'}")

    interesting = [
        "action_mean_speed",
        "action_mean_jerk",
        "action_speed_spikiness",
        "state_mean_speed",
        "state_mean_jerk",
        "state_speed_spikiness",
    ]
    comp_by_name = {row["metric"]: row for row in comparisons}

    print("\nQuick read:")
    for name in interesting:
        row = comp_by_name.get(name)
        if row is None:
            continue
        print(
            f"{name:>24}: "
            f"baseline={row['baseline_mean']:.4f}, "
            f"steered={row['steered_mean']:.4f}, "
            f"diff={row['abs_diff']:.4f}, "
            f"ratio={row['ratio']:.3f}, "
            f"d={row['cohens_d']:.3f}, "
            f"CI95=[{row['diff_ci95_lo']:.4f}, {row['diff_ci95_hi']:.4f}]"
        )


if __name__ == "__main__":
    main()