'''
 python src/lerobot/scripts/compare_steering_motion_eef.py \
  --repo-ids \
    ethanCSL/eval_koch_baseline \
    ethanCSL/eval_koch_high \
    ethanCSL/eval_koch_low \
  --labels baseline high low \
  --out-dir compare_baseline_high_low
'''
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import mujoco
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

def load_fk_model_from_local_xml(xml_name="follower.xml"):
    script_dir = Path(__file__).resolve().parent
    xml_path = script_dir / xml_name
    if not xml_path.exists():
        raise FileNotFoundError(f"Cannot find XML at: {xml_path}")

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def state_to_q_rad(state_vec, use_site_pose=True):
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)

    if state_vec.size < 5:
        raise ValueError(f"state size must be >= 5, got shape {state_vec.shape}")

    # 假設 observation.state 前 5~6 維就是 follower joint 狀態，且單位是 degree
    # 若你之後確認 state ordering 不同，只改這裡
    if state_vec.size >= 6:
        q_deg = state_vec[:6].copy()
    else:
        q_deg = np.concatenate([state_vec[:5], np.array([0.0], dtype=np.float64)], axis=0)

    q_rad = np.deg2rad(q_deg)

    # follower.xml 的 end_effector_site 掛在 link_5，下游 joint_6 不影響 site pose
    if use_site_pose:
        q_rad[5] = 0.0

    return q_rad


def compute_eef_pose_from_state_vector(state_vec, mj_model, mj_data, use_site_pose=True):
    q = state_to_q_rad(state_vec, use_site_pose=use_site_pose)

    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    for i, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_adr = mj_model.jnt_qposadr[joint_id]
        mj_data.qpos[qpos_adr] = float(q[i])

    mujoco.mj_fwdPosition(mj_model, mj_data)

    if use_site_pose:
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
        pos = mj_data.site_xpos[site_id].copy()
    else:
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        pos = mj_data.xpos[body_id].copy()

    return pos


def compute_eef_traces(states, xml_name="follower.xml", use_site_pose=True):
    if states is None or len(states) == 0:
        return None, None, None

    mj_model, mj_data = load_fk_model_from_local_xml(xml_name=xml_name)

    eef_positions = []
    for s in states:
        pos = compute_eef_pose_from_state_vector(
            s,
            mj_model=mj_model,
            mj_data=mj_data,
            use_site_pose=use_site_pose,
        )
        eef_positions.append(pos)

    eef_positions = np.asarray(eef_positions, dtype=np.float64)
    eef_height = eef_positions[:, 2].copy()
    eef_displacement = np.linalg.norm(eef_positions - eef_positions[0], axis=1)

    return eef_positions, eef_height, eef_displacement


def summarize_1d_trace(trace, prefix):
    out = {}
    if trace is None or len(trace) == 0:
        return out

    trace = np.asarray(trace, dtype=np.float64)
    out[f"{prefix}_mean"] = float(np.mean(trace))
    out[f"{prefix}_min"] = float(np.min(trace))
    out[f"{prefix}_max"] = float(np.max(trace))
    out[f"{prefix}_range"] = float(np.max(trace) - np.min(trace))
    out[f"{prefix}_final"] = float(trace[-1])
    out[f"{prefix}_trace"] = trace
    return out


def plot_1d_trace_bundle(rows, key):
    traces = []
    for r in rows:
        if key in r:
            traces.append(resample_trace(r[key], n_bins=100))

    if len(traces) == 0:
        return None

    arr = np.vstack(traces)
    return {
        "mean": np.nanmean(arr, axis=0),
        "std": np.nanstd(arr, axis=0),
    }


def plot_eef_trace_comparison_multi(rows_by_label, out_path):
    x = np.linspace(0.0, 1.0, 100)
    labels = list(rows_by_label.keys())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    def draw(ax, key, title, ylabel):
        drawn = False
        for label in labels:
            trace = plot_1d_trace_bundle(rows_by_label[label], key)
            if trace is None:
                continue
            ax.plot(x, trace["mean"], label=label)
            ax.fill_between(
                x,
                trace["mean"] - trace["std"],
                trace["mean"] + trace["std"],
                alpha=0.2,
            )
            drawn = True

        if drawn:
            ax.set_title(title)
            ax.set_xlabel("normalized episode time")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.set_visible(False)

    draw(axes[0], "eef_displacement_trace", "EEF displacement trace", "meters")
    draw(axes[1], "eef_height_trace", "EEF height trace", "meters")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def extract_height_trace_from_row(
    row,
    start_time=None,
    end_time=None,
    use_cm=True,
):
    if "eef_height_trace" not in row or "timestamps" not in row:
        return None

    h = np.asarray(row["eef_height_trace"], dtype=np.float64)
    t = np.asarray(row["timestamps"], dtype=np.float64)

    if start_time is not None or end_time is not None:
        mask = np.ones_like(t, dtype=bool)
        if start_time is not None:
            mask &= (t >= t[0] + start_time)
        if end_time is not None:
            mask &= (t <= t[0] + end_time)

        h = h[mask]

    if use_cm:
        h = h * 100.0

    return h


def plot_paper_low_high_height_trajectories(
    rows_by_label,
    out_path,
    low_label="low",
    high_label="high",
    n_episodes=10,
    start_time=None,
    end_time=None,
):
    if low_label not in rows_by_label or high_label not in rows_by_label:
        raise ValueError(
            f"rows_by_label must contain '{low_label}' and '{high_label}'. "
            f"Available labels: {list(rows_by_label.keys())}"
        )

    low_rows = rows_by_label[low_label][:n_episodes]
    high_rows = rows_by_label[high_label][:n_episodes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    panel_specs = [
        (axes[0], low_rows, "Low Intervention"),
        (axes[1], high_rows, "High Intervention"),
    ]

    for ax, rows, title in panel_specs:
        drawn = 0
        for row in rows:
            h = extract_height_trace_from_row(
                row,
                start_time=start_time,
                end_time=end_time,
                use_cm=True,
            )
            if h is None or len(h) == 0:
                continue

            x = np.arange(len(h))
            ax.plot(x, h, linewidth=1.6, alpha=0.95)
            drawn += 1

        ax.set_title(title)
        ax.set_xlabel("Action Step")
        ax.grid(True, alpha=0.25)

        if drawn > 0:
            ax.set_xlim(left=0)

    axes[0].set_ylabel("EE Height (cm)")

    fig.suptitle("Steering Intervention Trajectories", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

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

        eef_positions, eef_height, eef_displacement = compute_eef_traces(
            states,
            xml_name="follower.xml",
            use_site_pose=True,
        )

        cleaned[ep_idx] = {
            "timestamps": timestamps,
            "actions": actions,
            "states": states,
            "eef_positions": eef_positions,
            "eef_height": eef_height,
            "eef_displacement": eef_displacement,
        }

    return cleaned


def compute_episode_metrics(ep_idx, episode):
    timestamps = episode["timestamps"]
    actions = episode["actions"]
    states = episode["states"]
    eef_positions = episode.get("eef_positions", None)
    eef_height = episode.get("eef_height", None)
    eef_displacement = episode.get("eef_displacement", None)

    out = {
        "episode_index": ep_idx,
        "num_frames": int(len(timestamps)),
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0,
        "timestamps": timestamps,
    }

    out.update(compute_signal_metrics(actions, timestamps, prefix="action"))
    out.update(compute_signal_metrics(states, timestamps, prefix="state"))

    if eef_positions is not None:
        out.update(compute_signal_metrics(eef_positions, timestamps, prefix="eef_pos"))

    if eef_height is not None:
        out.update(summarize_1d_trace(eef_height, prefix="eef_height"))

    if eef_displacement is not None:
        out.update(summarize_1d_trace(eef_displacement, prefix="eef_displacement"))

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
        "eef_pos_path_len",
        "eef_pos_mean_speed",
        "eef_pos_p95_speed",
        "eef_pos_max_speed",
        "eef_pos_mean_jerk",
        "eef_pos_p95_jerk",
        "eef_pos_speed_spikiness",
        "eef_pos_endpoint_delta",
        "eef_height_mean",
        "eef_height_min",
        "eef_height_max",
        "eef_height_range",
        "eef_height_final",
        "eef_displacement_mean",
        "eef_displacement_min",
        "eef_displacement_max",
        "eef_displacement_range",
        "eef_displacement_final",
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


def compare_summaries_multi(rows_by_label):
    all_metric_keys = set()
    for rows in rows_by_label.values():
        for r in rows:
            all_metric_keys.update(r.keys())

    metric_names = sorted(all_metric_keys)

    ignore = {
        "episode_index",
        "num_frames",
        "timestamps",
        "action_trace_speed",
        "action_trace_jerk",
        "state_trace_speed",
        "state_trace_jerk",
        "eef_height_trace",
        "eef_displacement_trace",
    }

    labels = list(rows_by_label.keys())
    comparisons = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label_a = labels[i]
            label_b = labels[j]
            rows_a = rows_by_label[label_a]
            rows_b = rows_by_label[label_b]

            for name in metric_names:
                if name in ignore:
                    continue

                a = np.array(
                    [r[name] for r in rows_a if name in r and is_finite_scalar(r[name])],
                    dtype=np.float64,
                )
                b = np.array(
                    [r[name] for r in rows_b if name in r and is_finite_scalar(r[name])],
                    dtype=np.float64,
                )

                if len(a) == 0 or len(b) == 0:
                    continue

                ci_lo, ci_hi = bootstrap_mean_diff_ci(a, b, n_boot=3000, seed=42)
                comparisons.append({
                    "metric": name,
                    "label_a": label_a,
                    "label_b": label_b,
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "abs_diff": float(b.mean() - a.mean()),
                    "ratio": float((b.mean() + 1e-8) / (a.mean() + 1e-8)),
                    "cohens_d": cohens_d(a, b),
                    "diff_ci95_lo": ci_lo,
                    "diff_ci95_hi": ci_hi,
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
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


def plot_metric_boxplots_multi(rows_by_label, out_path):
    metrics = [
        "action_mean_speed",
        "action_mean_jerk",
        "action_speed_spikiness",
        "state_mean_speed",
        "state_mean_jerk",
        "state_speed_spikiness",
        "eef_height_range",
        "eef_displacement_final",
    ]

    labels = list(rows_by_label.keys())
    available = []

    for m in metrics:
        grouped = []
        valid_labels = []
        for label in labels:
            vals = [r[m] for r in rows_by_label[label] if m in r and np.isfinite(r[m])]
            if len(vals) > 0:
                grouped.append(vals)
                valid_labels.append(label)

        if len(grouped) >= 2:
            available.append((m, grouped, valid_labels))

    if len(available) == 0:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 4 * len(available)))
    if len(available) == 1:
        axes = [axes]

    for ax, (metric, grouped, valid_labels) in zip(axes, available):
        ax.boxplot(grouped, tick_labels=valid_labels)
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


def plot_trace_comparison_multi(rows_by_label, out_path):
    x = np.linspace(0.0, 1.0, 100)
    labels = list(rows_by_label.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    trace_specs = [
        ("action", "speed_mean", axes[0, 0], "Action speed trace"),
        ("action", "jerk_mean", axes[0, 1], "Action jerk trace"),
        ("state", "speed_mean", axes[1, 0], "State speed trace"),
        ("state", "jerk_mean", axes[1, 1], "State jerk trace"),
    ]

    for prefix, key, ax, title in trace_specs:
        drawn = False
        for label in labels:
            trace = plot_mean_trace(rows_by_label[label], prefix, None)
            if trace is None or key not in trace:
                continue
            ax.plot(x, trace[key], label=label)
            ax.fill_between(
                x,
                trace[key] - trace[key.replace("mean", "std")],
                trace[key] + trace[key.replace("mean", "std")],
                alpha=0.2,
            )
            drawn = True

        if drawn:
            ax.set_title(title)
            ax.set_xlabel("normalized episode time")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.set_visible(False)

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
            if k == "timestamps" or k.endswith("_trace_speed") or k.endswith("_trace_jerk") or k.endswith("_trace"):
                continue
            rr[k] = v
        clean.append(rr)
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-ids", type=str, nargs=3, required=True)
    parser.add_argument("--labels", type=str, nargs=3, default=["baseline", "high", "low"])
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="steering_compare")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = {}
    all_summaries = {}

    for label, repo_id in zip(args.labels, args.repo_ids):
        print(f"Loading {label} dataset: {repo_id}")
        eps = load_episodes(repo_id, root=args.root, max_frames=args.max_frames)
        rows = [compute_episode_metrics(ep_idx, ep) for ep_idx, ep in sorted(eps.items())]
        all_rows[label] = rows
        all_summaries[label] = summarize_metrics(rows, label)

        save_csv(out_dir / f"{label}_episode_metrics.csv", strip_trace_fields(rows))
        with (out_dir / f"{label}_summary.json").open("w") as f:
            json.dump(all_summaries[label], f, indent=2)

    comparisons = compare_summaries_multi(all_rows)
    save_csv(out_dir / "comparison_summary.csv", comparisons)

    plot_metric_boxplots_multi(all_rows, out_dir / "metric_boxplots.png")
    plot_trace_comparison_multi(all_rows, out_dir / "trace_comparison.png")
    plot_eef_trace_comparison_multi(all_rows, out_dir / "eef_trace_comparison.png")
    plot_paper_low_high_height_trajectories(
        all_rows,
        out_dir / "paper_low_high_height_10episodes.png",
        low_label="low",
        high_label="high",
        n_episodes=1,
        start_time=0,
        end_time=6,
    )

    print("\nSaved:")
    for label in args.labels:
        print(f"  {out_dir / f'{label}_episode_metrics.csv'}")
        print(f"  {out_dir / f'{label}_summary.json'}")
    print(f"  {out_dir / 'comparison_summary.csv'}")
    print(f"  {out_dir / 'metric_boxplots.png'}")
    print(f"  {out_dir / 'trace_comparison.png'}")
    print(f"  {out_dir / 'eef_trace_comparison.png'}")
    print(f"  {out_dir / 'paper_low_high_height_10episodes.png'}")

    interesting = [
        "action_mean_speed",
        "action_mean_jerk",
        "action_speed_spikiness",
        "state_mean_speed",
        "state_mean_jerk",
        "state_speed_spikiness",
        "eef_height_range",
        "eef_displacement_final",
    ]

    print("\nQuick read:")
    for row in comparisons:
        if row["metric"] not in interesting:
            continue
        print(
            f"{row['metric']:>24} | "
            f"{row['label_a']}={row['mean_a']:.4f}, "
            f"{row['label_b']}={row['mean_b']:.4f}, "
            f"diff={row['abs_diff']:.4f}, "
            f"ratio={row['ratio']:.3f}, "
            f"d={row['cohens_d']:.3f}, "
            f"CI95=[{row['diff_ci95_lo']:.4f}, {row['diff_ci95_hi']:.4f}]"
        )


if __name__ == "__main__":
    main()