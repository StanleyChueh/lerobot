#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import mujoco
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from aggregate_episode_chunk_height_compare import eef_z_from_state
except Exception as exc:
    raise RuntimeError(
        "Failed to import eef_z_from_state from aggregate_episode_chunk_height_compare.py. "
        "Put this script in src/lerobot/scripts/ and make sure aggregate_episode_chunk_height_compare.py exists."
    ) from exc


ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def to_vec6(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if isinstance(x, dict):
        vals = []
        for k in ACTION_KEYS:
            if k not in x:
                return None
            v = x[k]
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            vals.append(float(np.asarray(v).reshape(-1)[0]))
        return np.asarray(vals, dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        return None
    return arr[:6].copy()


def load_trace(trace_path: Path) -> list[dict[str, Any]]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return []
    trace = payload.get("trace", [])
    if not isinstance(trace, list):
        return []
    return trace


def load_all_action_obs_pairs(actual_run: Path, action_source: str):
    rows = []
    for trace_path in sorted(actual_run.glob("episode_*/policy_action_trace.pt")):
        ep = int(trace_path.parent.name.split("_")[-1])
        trace = load_trace(trace_path)

        for i, step in enumerate(trace):
            if not isinstance(step, dict):
                continue

            obs = to_vec6(step.get("observation_state"))
            act = to_vec6(step.get(action_source))

            if obs is None or act is None:
                continue

            row = {
                "episode_idx": ep,
                "step_idx": i,
            }

            for j, name in enumerate(JOINT_NAMES):
                row[f"obs_{name}"] = obs[j]
                row[f"act_{name}"] = act[j]

            rows.append(row)

    return pd.DataFrame(rows)


def fit_delay_and_affine_mapping(df: pd.DataFrame, max_delay: int):
    delay_scores = []

    for delay in range(max_delay + 1):
        per_joint_rmse = []
        per_joint_corr = []

        for name in JOINT_NAMES:
            pairs = []
            for ep, g in df.groupby("episode_idx"):
                g = g.sort_values("step_idx")
                act = g[f"act_{name}"].to_numpy(dtype=np.float64)
                obs = g[f"obs_{name}"].to_numpy(dtype=np.float64)

                if len(act) <= delay:
                    continue

                x = act[:-delay] if delay > 0 else act
                y = obs[delay:] if delay > 0 else obs

                if len(x) == len(y) and len(x) > 5:
                    pairs.append((x, y))

            if not pairs:
                continue

            x = np.concatenate([p[0] for p in pairs])
            y = np.concatenate([p[1] for p in pairs])

            if np.std(y) < 1e-8:
                continue

            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            pred = a * x + b
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1e-8 else np.nan

            scale = float(np.std(y)) + 1e-8
            per_joint_rmse.append(rmse / scale)
            if np.isfinite(corr):
                per_joint_corr.append(corr)

        score = float(np.mean(per_joint_rmse)) if per_joint_rmse else np.inf
        mean_corr = float(np.mean(per_joint_corr)) if per_joint_corr else np.nan
        delay_scores.append({
            "delay": delay,
            "normalized_rmse": score,
            "mean_corr": mean_corr,
        })

    delay_df = pd.DataFrame(delay_scores)
    best_delay = int(delay_df.sort_values("normalized_rmse").iloc[0]["delay"])

    mapping_rows = []
    for j, name in enumerate(JOINT_NAMES):
        pairs = []
        for ep, g in df.groupby("episode_idx"):
            g = g.sort_values("step_idx")
            act = g[f"act_{name}"].to_numpy(dtype=np.float64)
            obs = g[f"obs_{name}"].to_numpy(dtype=np.float64)

            if len(act) <= best_delay:
                continue

            x = act[:-best_delay] if best_delay > 0 else act
            y = obs[best_delay:] if best_delay > 0 else obs

            if len(x) == len(y) and len(x) > 5:
                pairs.append((x, y))

        if not pairs:
            mapping_rows.append({
                "joint_idx": j,
                "joint_name": name,
                "a": np.nan,
                "b": np.nan,
                "rmse": np.nan,
                "corr": np.nan,
                "obs_std": np.nan,
                "note": "no_pairs",
            })
            continue

        x = np.concatenate([p[0] for p in pairs])
        y = np.concatenate([p[1] for p in pairs])

        if np.std(y) < 1e-8:
            a = 0.0
            b = float(np.mean(y))
            pred = a * x + b
            note = "obs_constant"
        else:
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            pred = a * x + b
            note = "fit"

        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1e-8 and np.std(y) > 1e-8 else np.nan

        mapping_rows.append({
            "joint_idx": j,
            "joint_name": name,
            "a": float(a),
            "b": float(b),
            "rmse": rmse,
            "corr": corr,
            "obs_std": float(np.std(y)),
            "note": note,
        })

    mapping_df = pd.DataFrame(mapping_rows)
    return best_delay, delay_df, mapping_df


def map_action_vec_to_obs_convention(action_vec: np.ndarray, mapping_df: pd.DataFrame) -> np.ndarray:
    out = np.asarray(action_vec, dtype=np.float64).reshape(-1)[:6].copy()
    for _, row in mapping_df.iterrows():
        j = int(row["joint_idx"])
        a = row["a"]
        b = row["b"]
        if np.isfinite(a) and np.isfinite(b):
            out[j] = float(a) * out[j] + float(b)
    return out


def find_pred_summary(ep_dir: Path) -> Path | None:
    candidates = [
        ep_dir / "sequence_exact_predicted_action_chunk_max_fk_eef_z_summary.csv",
        ep_dir / "predicted_action_chunk_max_fk_eef_z_summary.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    hits = sorted(ep_dir.glob("*predicted*chunk*max*fk*eef*z*summary*.csv"))
    return hits[0] if hits else None


def load_raw_predicted_summary(pred_root: Path) -> pd.DataFrame:
    rows = []

    # Case 1: root-level single-sequence output
    root_summary = pred_root / "sequence_exact_predicted_action_chunk_max_fk_eef_z_summary.csv"
    if root_summary.exists():
        s = pd.read_csv(root_summary)
        if len(s) > 0:
            r = s.iloc[0].to_dict()
            rows.append({
                "episode_idx": 0,
                "raw_predicted_max_fk_eef_z": float(r.get("predicted_chunk_fk_eef_z_max_mean", np.nan)),
                "raw_predicted_max_fk_eef_z_std": float(r.get("predicted_chunk_fk_eef_z_max_std", np.nan)),
                "raw_predicted_max_fk_eef_z_range": float(r.get("predicted_chunk_fk_eef_z_max_range", np.nan)),
                "predicted_layout": "root_single_sequence",
            })

    # Case 2: per-episode output
    for ep_dir in sorted(pred_root.glob("episode_*")):
        if not ep_dir.is_dir():
            continue

        ep = int(ep_dir.name.split("_")[-1])
        p = find_pred_summary(ep_dir)
        if p is None:
            continue

        s = pd.read_csv(p)
        if len(s) == 0:
            continue

        r = s.iloc[0].to_dict()
        rows.append({
            "episode_idx": ep,
            "raw_predicted_max_fk_eef_z": float(r.get("predicted_chunk_fk_eef_z_max_mean", np.nan)),
            "raw_predicted_max_fk_eef_z_std": float(r.get("predicted_chunk_fk_eef_z_max_std", np.nan)),
            "raw_predicted_max_fk_eef_z_range": float(r.get("predicted_chunk_fk_eef_z_max_range", np.nan)),
            "predicted_layout": "episode_folder",
        })

    return pd.DataFrame(rows)


def find_pred_trials(ep_dir: Path) -> Path | None:
    candidates = [
        ep_dir / "sequence_exact_predicted_action_chunk_trials.csv",
        ep_dir / "predicted_action_chunk_trials.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    hits = sorted(ep_dir.glob("*predicted*action*chunk*trials*.csv"))
    return hits[0] if hits else None


def infer_action_columns(df: pd.DataFrame) -> list[str]:
    patterns = [
        re.compile(r"^action_values[._](\d+)$"),
        re.compile(r"^action[._](\d+)$"),
        re.compile(r"^dim[._](\d+)$"),
        re.compile(r"^joint[._](\d+)$"),
    ]

    found = []
    for c in df.columns:
        for pat in patterns:
            m = pat.match(c)
            if m:
                found.append((int(m.group(1)), c))
                break

    found = sorted(found, key=lambda x: x[0])
    cols = [c for idx, c in found if idx < 6]

    if len(cols) >= 6:
        return cols[:6]

    numeric_cols = []
    blocked = {
        "trial_idx", "trial", "repeat_idx", "step_idx", "chunk_step_idx",
        "episode_idx", "time_s", "chunk_index", "raw_chunk_id"
    }
    for c in df.columns:
        if c in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    if len(numeric_cols) >= 6:
        return numeric_cols[:6]

    raise ValueError(f"Could not infer 6 action columns from: {list(df.columns)}")


def infer_trial_column(df: pd.DataFrame) -> str | None:
    for c in ["trial_idx", "trial", "repeat_idx", "repeat", "sample_idx"]:
        if c in df.columns:
            return c
    return None


def compute_mapped_predicted_summary(pred_root: Path, mapping_df: pd.DataFrame, model, data) -> pd.DataFrame:
    rows = []

    pred_sources = []

    # Case 1: root-level single-sequence output
    root_trials = pred_root / "sequence_exact_predicted_action_chunk_trials.csv"
    if root_trials.exists():
        pred_sources.append((0, pred_root, root_trials, "root_single_sequence"))

    # Case 2: per-episode output
    for ep_dir in sorted(pred_root.glob("episode_*")):
        if not ep_dir.is_dir():
            continue
        ep = int(ep_dir.name.split("_")[-1])
        p = find_pred_trials(ep_dir)
        if p is not None:
            pred_sources.append((ep, ep_dir, p, "episode_folder"))

    for ep, ep_dir, p, layout in pred_sources:
        df = pd.read_csv(p)
        if len(df) == 0:
            continue

        action_cols = infer_action_columns(df)
        trial_col = infer_trial_column(df)

        max_per_trial = []
        if trial_col is None:
            groups = [(0, df)]
        else:
            groups = list(df.groupby(trial_col))

        for trial_id, g in groups:
            z_vals = []
            for _, row in g.iterrows():
                act = row[action_cols].to_numpy(dtype=np.float64)
                mapped = map_action_vec_to_obs_convention(act, mapping_df)
                z_vals.append(eef_z_from_state(mapped, model, data))

            if z_vals:
                max_per_trial.append(float(np.max(z_vals)))

        if not max_per_trial:
            rows.append({
                "episode_idx": ep,
                "mapped_predicted_max_fk_eef_z_mean": np.nan,
                "mapped_predicted_max_fk_eef_z_std": np.nan,
                "mapped_predicted_max_fk_eef_z_range": np.nan,
                "mapped_predicted_num_trials": 0,
                "mapped_predicted_source": str(p),
                "predicted_layout": layout,
            })
            continue

        vals = np.asarray(max_per_trial, dtype=np.float64)
        rows.append({
            "episode_idx": ep,
            "mapped_predicted_max_fk_eef_z_mean": float(np.mean(vals)),
            "mapped_predicted_max_fk_eef_z_std": float(np.std(vals, ddof=0)),
            "mapped_predicted_max_fk_eef_z_range": float(np.max(vals) - np.min(vals)),
            "mapped_predicted_num_trials": int(len(vals)),
            "mapped_predicted_source": str(p),
            "predicted_layout": layout,
        })

    return pd.DataFrame(rows)


def compute_actual_obs_summary(actual_run: Path, model, data, actual_chunk_index=None, actual_delay=0) -> pd.DataFrame:
    rows = []

    for trace_path in sorted(actual_run.glob("episode_*/policy_action_trace.pt")):
        ep = int(trace_path.parent.name.split("_")[-1])
        trace = load_trace(trace_path)

        z_vals = []
        for i, step in enumerate(trace):
            if actual_chunk_index is not None:
                if step.get("chunk_index") != actual_chunk_index:
                    continue

            obs_i = i + int(actual_delay)
            if obs_i >= len(trace):
                continue

            obs = to_vec6(trace[obs_i].get("observation_state"))
            if obs is None:
                continue

            z_vals.append(eef_z_from_state(obs, model, data))

        if z_vals:
            z = np.asarray(z_vals, dtype=np.float64)
            rows.append({
                "episode_idx": ep,
                "actual_obs_max_fk_eef_z": float(np.max(z)),
                "actual_obs_min_fk_eef_z": float(np.min(z)),
                "actual_obs_range_fk_eef_z": float(np.max(z) - np.min(z)),
                "actual_obs_num_points": int(len(z)),
                "actual_obs_chunk_index": actual_chunk_index,
                "actual_obs_delay": actual_delay,
            })

    return pd.DataFrame(rows)


def compute_sent_vs_obs_delay_summary(
    actual_run: Path,
    action_source: str,
    delay: int,
    mapping_df: pd.DataFrame,
    model,
    data,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    first_episode_detail = None

    for trace_path in sorted(actual_run.glob("episode_*/policy_action_trace.pt")):
        ep = int(trace_path.parent.name.split("_")[-1])
        trace = load_trace(trace_path)

        raw_z = []
        mapped_z = []
        obs_z = []
        steps = []

        for i, step in enumerate(trace):
            obs_idx = i + delay
            if obs_idx >= len(trace):
                break

            act = to_vec6(step.get(action_source))
            obs = to_vec6(trace[obs_idx].get("observation_state"))
            if act is None or obs is None:
                continue

            mapped = map_action_vec_to_obs_convention(act, mapping_df)

            raw_z.append(eef_z_from_state(act, model, data))
            mapped_z.append(eef_z_from_state(mapped, model, data))
            obs_z.append(eef_z_from_state(obs, model, data))
            steps.append(i)

        if raw_z:
            raw_z = np.asarray(raw_z, dtype=np.float64)
            mapped_z = np.asarray(mapped_z, dtype=np.float64)
            obs_z = np.asarray(obs_z, dtype=np.float64)

            rows.append({
                "episode_idx": ep,
                "delay": delay,
                "sent_raw_max_fk_eef_z": float(np.max(raw_z)),
                "sent_mapped_max_fk_eef_z": float(np.max(mapped_z)),
                "obs_delay_aligned_max_fk_eef_z": float(np.max(obs_z)),
                "sent_raw_minus_obs_max": float(np.max(raw_z) - np.max(obs_z)),
                "sent_mapped_minus_obs_max": float(np.max(mapped_z) - np.max(obs_z)),
                "num_points": int(len(raw_z)),
            })

            if first_episode_detail is None:
                first_episode_detail = (ep, np.asarray(steps), raw_z, mapped_z, obs_z)

    if first_episode_detail is not None:
        ep, steps, raw_z, mapped_z, obs_z = first_episode_detail
        plt.figure(figsize=(11, 5))
        plt.plot(steps, raw_z, label=f"{action_source} raw FK z")
        plt.plot(steps, mapped_z, label=f"{action_source} mapped-to-obs FK z")
        plt.plot(steps, obs_z, label=f"observation_state[t+{delay}] FK z")
        plt.xlabel("control step t")
        plt.ylabel("FK EEF z (m)")
        plt.title(f"C: delay-aligned {action_source} vs observation_state FK z | episode {ep:06d}, delay={delay}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"C_delay_aligned_{action_source}_vs_obs_fk_z_episode_{ep:06d}.png", dpi=160, bbox_inches="tight")
        plt.close()

    return pd.DataFrame(rows)


def plot_episode_lines(df: pd.DataFrame, out_dir: Path):
    # A
    if {"raw_predicted_max_fk_eef_z", "actual_obs_max_fk_eef_z"}.issubset(df.columns):
        valid = df.dropna(subset=["raw_predicted_max_fk_eef_z", "actual_obs_max_fk_eef_z"])
        plt.figure(figsize=(10, 5))
        plt.plot(valid["episode_idx"], valid["raw_predicted_max_fk_eef_z"], marker="o", label="raw predicted action chunk FK max z")
        plt.plot(valid["episode_idx"], valid["actual_obs_max_fk_eef_z"], marker="o", label="actual observation_state FK max z")
        plt.xlabel("episode index")
        plt.ylabel("Max FK EEF z (m)")
        plt.title("A: raw predicted action FK vs actual observation_state FK")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "A_raw_predicted_vs_actual_observation_fk_z.png", dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 5))
        diff = valid["raw_predicted_max_fk_eef_z"] - valid["actual_obs_max_fk_eef_z"]
        plt.plot(valid["episode_idx"], diff, marker="o", label="raw predicted - actual observation")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("episode index")
        plt.ylabel("Height difference (m)")
        plt.title("A: raw predicted minus actual observation FK max z")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "A_raw_predicted_minus_actual_observation_fk_z.png", dpi=160, bbox_inches="tight")
        plt.close()

    # B
    if {"mapped_predicted_max_fk_eef_z_mean", "actual_obs_max_fk_eef_z"}.issubset(df.columns):
        valid = df.dropna(subset=["mapped_predicted_max_fk_eef_z_mean", "actual_obs_max_fk_eef_z"])
        plt.figure(figsize=(10, 5))
        plt.plot(valid["episode_idx"], valid["mapped_predicted_max_fk_eef_z_mean"], marker="o", label="mapped predicted action chunk FK max z")
        plt.plot(valid["episode_idx"], valid["actual_obs_max_fk_eef_z"], marker="o", label="actual observation_state FK max z")
        plt.xlabel("episode index")
        plt.ylabel("Max FK EEF z (m)")
        plt.title("B: mapped predicted action FK vs actual observation_state FK")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "B_mapped_predicted_vs_actual_observation_fk_z.png", dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 5))
        diff = valid["mapped_predicted_max_fk_eef_z_mean"] - valid["actual_obs_max_fk_eef_z"]
        plt.plot(valid["episode_idx"], diff, marker="o", label="mapped predicted - actual observation")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("episode index")
        plt.ylabel("Height difference (m)")
        plt.title("B: mapped predicted minus actual observation FK max z")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "B_mapped_predicted_minus_actual_observation_fk_z.png", dpi=160, bbox_inches="tight")
        plt.close()


def plot_mapping(mapping_df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(8, 4))
    x = np.arange(len(mapping_df))
    plt.bar(x, mapping_df["a"].to_numpy(dtype=float))
    plt.xticks(x, mapping_df["joint_name"], rotation=30, ha="right")
    plt.ylabel("affine slope a")
    plt.title("Action to observation_state affine mapping slope")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "mapping_action_to_observation_slope.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(x, mapping_df["rmse"].to_numpy(dtype=float))
    plt.xticks(x, mapping_df["joint_name"], rotation=30, ha="right")
    plt.ylabel("RMSE")
    plt.title("Action to observation_state mapping RMSE")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "mapping_action_to_observation_rmse.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True, type=Path)
    ap.add_argument("--actual-run", required=True, type=Path)
    ap.add_argument("--xml", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--action-source", default="sent_action",
                    choices=["sent_action", "robot_action_to_send", "act_processed_policy", "policy_action_values"])
    ap.add_argument("--max-delay", type=int, default=20)
    ap.add_argument("--actual-chunk-index", type=int, default=None)
    ap.add_argument("--actual-delay", type=int, default=None)
    args = ap.parse_args()

    out_dir = args.out if args.out is not None else args.pred_root / "action_observation_fk_convention_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)

    print("[*] Loading action/observation pairs from trace...")
    pair_df = load_all_action_obs_pairs(args.actual_run, args.action_source)
    pair_df.to_csv(out_dir / "action_observation_pairs_by_step.csv", index=False)

    if len(pair_df) == 0:
        raise RuntimeError("No action/observation pairs found. Check --actual-run and --action-source.")

    print("[*] Fitting action -> observation_state affine mapping with delay search...")
    best_delay, delay_df, mapping_df = fit_delay_and_affine_mapping(pair_df, args.max_delay)
    delay_df.to_csv(out_dir / "delay_search_summary.csv", index=False)
    mapping_df.to_csv(out_dir / "action_to_observation_affine_mapping.csv", index=False)

    print(f"[*] Best delay: {best_delay} control steps")
    print("\n=== action -> observation_state mapping ===")
    print(mapping_df.to_string(index=False))

    plot_mapping(mapping_df, out_dir)

    print("[*] Loading raw predicted summary...")
    raw_pred_df = load_raw_predicted_summary(args.pred_root)

    print("[*] Computing actual observation_state FK summary...")
    actual_delay = best_delay if args.actual_delay is None else args.actual_delay

    actual_df = compute_actual_obs_summary(
        args.actual_run,
        model,
        data,
        actual_chunk_index=args.actual_chunk_index,
        actual_delay=actual_delay,
    )

    print("[*] Computing mapped-predicted-action FK summary...")
    mapped_pred_df = compute_mapped_predicted_summary(args.pred_root, mapping_df, model, data)

    print("[*] Computing delay-aligned sent/action vs observation FK summary...")
    sent_vs_obs_df = compute_sent_vs_obs_delay_summary(
        actual_run=args.actual_run,
        action_source=args.action_source,
        delay=best_delay,
        mapping_df=mapping_df,
        model=model,
        data=data,
        out_dir=out_dir,
    )

    if "episode_idx" not in raw_pred_df.columns:
        raise RuntimeError(
            f"No predicted summary found under {args.pred_root}. "
            "Expected either root-level sequence_exact_predicted_action_chunk_max_fk_eef_z_summary.csv "
            "or per-episode episode_*/sequence_exact_predicted_action_chunk_max_fk_eef_z_summary.csv."
        )

    summary = raw_pred_df.merge(actual_df, on="episode_idx", how="outer")
    summary = summary.merge(mapped_pred_df, on="episode_idx", how="outer")
    summary = summary.merge(sent_vs_obs_df, on="episode_idx", how="outer")

    summary = summary.sort_values("episode_idx")
    summary.to_csv(out_dir / "ABC_fk_comparison_summary.csv", index=False)

    plot_episode_lines(summary, out_dir)

    if len(sent_vs_obs_df):
        plt.figure(figsize=(10, 5))
        plt.plot(sent_vs_obs_df["episode_idx"], sent_vs_obs_df["sent_raw_minus_obs_max"], marker="o", label=f"{args.action_source} raw FK max - obs FK max")
        plt.plot(sent_vs_obs_df["episode_idx"], sent_vs_obs_df["sent_mapped_minus_obs_max"], marker="o", label=f"{args.action_source} mapped FK max - obs FK max")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("episode index")
        plt.ylabel("Height difference (m)")
        plt.title(f"C: delay-aligned {args.action_source} vs observation_state FK max z")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "C_delay_aligned_sent_vs_obs_max_fk_z_by_episode.png", dpi=160, bbox_inches="tight")
        plt.close()

    print("\n=== ABC summary ===")
    keep_cols = [
        "episode_idx",
        "raw_predicted_max_fk_eef_z",
        "mapped_predicted_max_fk_eef_z_mean",
        "actual_obs_max_fk_eef_z",
        "sent_raw_max_fk_eef_z",
        "sent_mapped_max_fk_eef_z",
        "obs_delay_aligned_max_fk_eef_z",
        "sent_raw_minus_obs_max",
        "sent_mapped_minus_obs_max",
    ]
    keep_cols = [c for c in keep_cols if c in summary.columns]
    print(summary[keep_cols].to_string(index=False))

    print("\nKey outputs:")
    for name in [
        "ABC_fk_comparison_summary.csv",
        "action_to_observation_affine_mapping.csv",
        "delay_search_summary.csv",
        "A_raw_predicted_vs_actual_observation_fk_z.png",
        "A_raw_predicted_minus_actual_observation_fk_z.png",
        "B_mapped_predicted_vs_actual_observation_fk_z.png",
        "B_mapped_predicted_minus_actual_observation_fk_z.png",
        "C_delay_aligned_sent_vs_obs_max_fk_z_by_episode.png",
    ]:
        p = out_dir / name
        if p.exists():
            print("  -", p)

    print("[DONE]")


if __name__ == "__main__":
    main()
