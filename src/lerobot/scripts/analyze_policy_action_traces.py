#!/usr/bin/env python3
"""
analyze_policy_action_traces.py

Analyze per-control-step policy_action_trace.pt files from normal high/low/baseline rollouts.

Primary use case:
  You recorded 10 high-intervention rollouts with action_trace.enabled=true.
  The EEF height is inconsistent again.
  Fixed-action replay was stable, so now we need to check whether normal policy
  rollouts generated different full action sequences.

Example:
  python src/lerobot/scripts/analyze_policy_action_traces.py \
    --run debug_runs/20260601_112414 \
    --xml src/lerobot/scripts/follower.xml \
    --out analysis_high_action_trace_20260601_112414 \
    --focus-chunks 0,1,2

Optional manual split:
  python src/lerobot/scripts/analyze_policy_action_traces.py \
    --run debug_runs/20260601_112414 \
    --xml src/lerobot/scripts/follower.xml \
    --out analysis_high_action_trace_20260601_112414 \
    --focus-chunks 0,1,2 \
    --manual-high-eef-episodes 1,2,5,6,7 \
    --manual-low-eef-episodes 0,3,4,8,9

Outputs:
  - episode_summary.csv
  - steps_per_chunk.csv
  - per_episode_chunk_action_summary.csv
  - top_mode_action_separators.csv
  - top_mode_state_separators.csv
  - action_trace_long.csv
  - diagnosis.md
  - plots/eef_z_by_step.png
  - plots/eef_z_by_chunk_step_focus.png
  - plots/action_<key>_by_step.png for focus action keys
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    import mujoco
except Exception as exc:
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None


# ==============================================================================
# EEF FK calibration copied from physical_neuron_finding_eef.py
# ==============================================================================

CALIB_REST_STATE_DEG = np.asarray([
    -11.20879121,
    97.00520833,
    17.25596857,
    100.0,
    -9.01098901,
    47.38863287,
], dtype=np.float64)

CALIB_TARGET_REST_DEG = np.asarray([
    91.7,
    15.0,
    40.0,
    65.0,
    0.0,
    -30.0,
], dtype=np.float64)

CALIB_ORDER = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
CALIB_SCALE = np.asarray([1.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
CALIB_SIGN = np.asarray([1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float64)


def state_to_q_rad(state_vec: np.ndarray) -> np.ndarray:
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)
    if state_vec.size < 6:
        raise ValueError(f"State must have >= 6 dims, got {state_vec.shape}")
    raw_deg = state_vec[:6].copy()
    raw_delta = raw_deg[CALIB_ORDER] - CALIB_REST_STATE_DEG[CALIB_ORDER]
    q_deg = CALIB_TARGET_REST_DEG + CALIB_SIGN * CALIB_SCALE * raw_delta
    return np.deg2rad(q_deg)


def load_fk_model(xml_path: Path):
    if mujoco is None:
        raise ImportError(
            "mujoco import failed. Run in the same environment where your EEF scripts work. "
            f"Original error: {MUJOCO_IMPORT_ERROR}"
        )
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def eef_z_from_state(state_vec: np.ndarray, model, data) -> float:
    q = state_to_q_rad(state_vec)
    mujoco.mj_resetData(model, data)

    for i, joint_name in enumerate(["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        data.qpos[model.jnt_qposadr[jid]] = float(q[i])

    mujoco.mj_forward(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    if sid < 0:
        raise ValueError("Site not found: end_effector_site")

    return float(data.site_xpos[sid][2])


# ==============================================================================
# General helpers
# ==============================================================================

def natural_sort_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}

    if value is None:
        return out

    if isinstance(value, dict):
        for k, v in value.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric(v, child))
        return out

    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            child = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_numeric(v, child))
        return out

    try:
        arr = to_numpy(value)
    except Exception:
        return out

    try:
        if not np.issubdtype(arr.dtype, np.number):
            return out
    except Exception:
        return out

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        out[prefix] = float(arr)
        return out

    flat = arr.reshape(-1)
    for i, v in enumerate(flat):
        key = f"{prefix}.{i}" if prefix else str(i)
        out[key] = float(v)

    return out


def parse_episode_list(text: str | None) -> list[int] | None:
    if text is None or str(text).strip() == "":
        return None
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_int_list(text: str | None) -> list[int]:
    if text is None or str(text).strip() == "":
        return []
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def infer_episode_idx_from_path(path: Path) -> int:
    m = re.search(r"episode_(\d+)", str(path))
    return int(m.group(1)) if m else -1


def load_trace_file(path: Path, model, data) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trace = payload["trace"]
    metadata = payload.get("metadata", {}) or {}
    ep_idx = infer_episode_idx_from_path(path)

    rows = []
    for row in trace:
        state = row.get("observation_state")
        state_arr = None
        eef_z = np.nan

        if state is not None:
            state_arr = np.asarray(to_numpy(state), dtype=np.float64).reshape(-1)
            eef_z = eef_z_from_state(state_arr, model, data)

        policy_flat = flatten_numeric(row.get("policy_action_values"), "policy")
        processed_flat = flatten_numeric(row.get("act_processed_policy"), "processed")
        to_send_flat = flatten_numeric(row.get("robot_action_to_send"), "to_send")
        sent_flat = flatten_numeric(row.get("sent_action"), "sent")

        out = {
            "path": str(path),
            "episode": f"episode_{ep_idx:06d}",
            "episode_idx": ep_idx,
            "trace_step": int(row.get("trace_step", -1)),
            "time_s": float(row.get("time_s", np.nan)),
            "chunk_index": int(row.get("chunk_index", -1)) if row.get("chunk_index") is not None else -1,
            "raw_chunk_id": int(row.get("raw_chunk_id", -1)) if row.get("raw_chunk_id") is not None else -1,
            "state_key": row.get("state_key"),
            "eef_z": float(eef_z),
            "task": row.get("task"),
            "metadata_intervention": metadata.get("intervention", ""),
        }

        if state_arr is not None:
            for i, v in enumerate(state_arr[:6]):
                out[f"state.{i}"] = float(v)

        out.update(policy_flat)
        out.update(processed_flat)
        out.update(to_send_flat)
        out.update(sent_flat)
        rows.append(out)

    df = pd.DataFrame(rows)

    # Step index within each generated chunk.
    df["chunk_step"] = df.groupby(["episode_idx", "chunk_index"]).cumcount()
    return df, metadata


def assign_modes(summary: pd.DataFrame, manual_high: list[int] | None, manual_low: list[int] | None) -> pd.DataFrame:
    summary = summary.copy()

    if manual_high is not None or manual_low is not None:
        high_set = set(manual_high or [])
        low_set = set(manual_low or [])
        modes = []
        for ep in summary["episode_idx"].astype(int):
            if ep in high_set:
                modes.append("high_eef_mode")
            elif ep in low_set:
                modes.append("low_eef_mode")
            else:
                modes.append("unlabeled")
        summary["eef_mode"] = modes
        summary["eef_mode_threshold"] = np.nan
        return summary

    threshold = float(summary["eef_z_max"].median())
    summary["eef_mode"] = np.where(summary["eef_z_max"] >= threshold, "high_eef_mode", "low_eef_mode")
    summary["eef_mode_threshold"] = threshold
    return summary


def robust_effect_rows(df: pd.DataFrame, group_col: str, value_cols: list[str], by_cols: list[str]) -> pd.DataFrame:
    rows = []

    for group_key, g in df.groupby(by_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        high = g[g[group_col] == "high_eef_mode"]
        low = g[g[group_col] == "low_eef_mode"]

        if len(high) == 0 or len(low) == 0:
            continue

        for col in value_cols:
            hv = high[col].to_numpy(dtype=np.float64)
            lv = low[col].to_numpy(dtype=np.float64)
            hv = hv[np.isfinite(hv)]
            lv = lv[np.isfinite(lv)]

            if len(hv) == 0 or len(lv) == 0:
                continue

            high_mean = float(np.mean(hv))
            low_mean = float(np.mean(lv))
            high_std = float(np.std(hv, ddof=0))
            low_std = float(np.std(lv, ddof=0))
            pooled = math.sqrt(max(high_std**2 + low_std**2, 1e-12) / 2.0)

            out = {by_cols[i]: group_key[i] for i in range(len(by_cols))}
            out.update({
                "key": col,
                "high_mean": high_mean,
                "low_mean": low_mean,
                "high_minus_low": high_mean - low_mean,
                "abs_high_minus_low": abs(high_mean - low_mean),
                "high_std": high_std,
                "low_std": low_std,
                "effect_z": abs(high_mean - low_mean) / pooled,
                "high_count": int(len(hv)),
                "low_count": int(len(lv)),
            })
            rows.append(out)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    return out_df.sort_values(["effect_z", "abs_high_minus_low"], ascending=False)


# ==============================================================================
# Analysis
# ==============================================================================

def make_episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ep_idx, episode), g in df.groupby(["episode_idx", "episode"]):
        g = g.sort_values("trace_step")
        z = g["eef_z"].to_numpy(dtype=np.float64)

        rows.append({
            "episode_idx": int(ep_idx),
            "episode": episode,
            "num_steps": int(len(g)),
            "num_chunks": int(g["chunk_index"].nunique()),
            "first_chunk": int(g["chunk_index"].min()),
            "last_chunk": int(g["chunk_index"].max()),
            "eef_z_start": float(z[0]),
            "eef_z_end": float(z[-1]),
            "eef_z_min": float(np.nanmin(z)),
            "eef_z_max": float(np.nanmax(z)),
            "eef_z_range": float(np.nanmax(z) - np.nanmin(z)),
            "eef_z_p90": float(np.nanpercentile(z, 90)),
            "peak_step": int(g.iloc[int(np.nanargmax(z))]["trace_step"]),
            "peak_chunk": int(g.iloc[int(np.nanargmax(z))]["chunk_index"]),
            "metadata_intervention": str(g["metadata_intervention"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values("episode_idx")


def steps_per_chunk(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    mode_map = summary.set_index("episode_idx")["eef_mode"].to_dict()
    out = (
        df.groupby(["episode_idx", "episode", "chunk_index"])
        .agg(
            num_steps=("trace_step", "count"),
            eef_z_start=("eef_z", "first"),
            eef_z_end=("eef_z", "last"),
            eef_z_max=("eef_z", "max"),
            first_trace_step=("trace_step", "min"),
            last_trace_step=("trace_step", "max"),
        )
        .reset_index()
    )
    out["eef_mode"] = out["episode_idx"].map(mode_map)
    return out.sort_values(["episode_idx", "chunk_index"])


def per_episode_chunk_action_summary(df: pd.DataFrame, action_cols: list[str], state_cols: list[str]) -> pd.DataFrame:
    agg = {}
    for col in action_cols + state_cols + ["eef_z"]:
        agg[f"{col}.mean"] = (col, "mean")
        agg[f"{col}.std"] = (col, "std")
        agg[f"{col}.first"] = (col, "first")
        agg[f"{col}.last"] = (col, "last")
        agg[f"{col}.min"] = (col, "min")
        agg[f"{col}.max"] = (col, "max")

    out = df.groupby(["episode_idx", "episode", "eef_mode", "chunk_index"]).agg(**agg).reset_index()
    return out.sort_values(["episode_idx", "chunk_index"])


def plot_eef_by_step(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(11, 6))
    for ep_idx, g in df.groupby("episode_idx"):
        g = g.sort_values("trace_step")
        label = f"ep{ep_idx:02d}-{g['eef_mode'].iloc[0].replace('_eef_mode','')}"
        plt.plot(g["trace_step"], g["eef_z"], label=label)
    plt.xlabel("Trace step")
    plt.ylabel("EEF z")
    plt.title("Normal policy rollouts: EEF z by control step")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "eef_z_by_step.png", dpi=160)
    plt.close()


def plot_eef_focus_chunks(df: pd.DataFrame, focus_chunks: list[int], out_dir: Path) -> None:
    if not focus_chunks:
        return
    sub = df[df["chunk_index"].isin(focus_chunks)].copy()
    if sub.empty:
        return

    plt.figure(figsize=(11, 6))
    for ep_idx, g in sub.groupby("episode_idx"):
        g = g.sort_values(["chunk_index", "chunk_step"])
        # Continuous x: chunk_index * 1000 + chunk_step, simple but readable.
        x = g["chunk_index"].to_numpy() * 1000 + g["chunk_step"].to_numpy()
        label = f"ep{ep_idx:02d}-{g['eef_mode'].iloc[0].replace('_eef_mode','')}"
        plt.plot(x, g["eef_z"], label=label)
    plt.xlabel("chunk_index * 1000 + chunk_step")
    plt.ylabel("EEF z")
    plt.title(f"EEF z in focus chunks {focus_chunks}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "eef_z_by_chunk_step_focus.png", dpi=160)
    plt.close()


def plot_action_keys(df: pd.DataFrame, action_cols: list[str], out_dir: Path, max_keys: int = 12) -> None:
    for col in action_cols[:max_keys]:
        plt.figure(figsize=(11, 6))
        for ep_idx, g in df.groupby("episode_idx"):
            g = g.sort_values("trace_step")
            label = f"ep{ep_idx:02d}-{g['eef_mode'].iloc[0].replace('_eef_mode','')}"
            plt.plot(g["trace_step"], g[col], label=label)
        plt.xlabel("Trace step")
        plt.ylabel(col)
        safe_col = col.replace(".", "_").replace("/", "_")
        plt.title(f"{col} by step")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"action_{safe_col}_by_step.png", dpi=160)
        plt.close()


def write_report(
    out_dir: Path,
    run: str,
    summary: pd.DataFrame,
    steps: pd.DataFrame,
    action_sep: pd.DataFrame,
    state_sep: pd.DataFrame,
    focus_chunks: list[int],
) -> None:
    lines = []
    lines.append("# Policy action trace diagnosis")
    lines.append("")
    lines.append(f"Run: `{run}`")
    lines.append("")
    lines.append("## Episode summary")
    lines.append("")
    lines.append("```")
    lines.append(summary.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Steps per chunk")
    lines.append("")
    lines.append("```")
    lines.append(steps.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Top high-EEF vs low-EEF action separators")
    lines.append("")
    lines.append("These compare full per-control-step command traces, grouped by EEF mode.")
    lines.append("")
    lines.append("```")
    if action_sep.empty:
        lines.append("(empty)")
    else:
        lines.append(action_sep.head(40).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Top high-EEF vs low-EEF state/EEF separators")
    lines.append("")
    lines.append("```")
    if state_sep.empty:
        lines.append("(empty)")
    else:
        lines.append(state_sep.head(40).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation rules")
    lines.append("")
    lines.append("1. If action separators are large in chunk 0 or chunk 1, the high/low lift split is caused by different policy-generated command sequences before lift.")
    lines.append("2. If sent_action and to_send differences are small but EEF/state separators are large, contact/execution dynamics are still involved.")
    lines.append("3. If step counts per chunk differ, compare by chunk_step within each chunk, not global trace_step.")
    lines.append("4. Focus on the earliest chunk with large separation; that is the likely branching point.")
    lines.append("")
    lines.append(f"Focus chunks analyzed: `{focus_chunks}`")
    lines.append("")
    (out_dir / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory containing episode_*/policy_action_trace.pt")
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--focus-chunks", default="0,1,2")
    parser.add_argument("--manual-high-eef-episodes", default=None)
    parser.add_argument("--manual-low-eef-episodes", default=None)
    parser.add_argument("--plot-max-action-keys", type=int, default=12)
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model, data = load_fk_model(Path(args.xml))

    paths = sorted(run_dir.glob("episode_*/policy_action_trace.pt"), key=lambda p: natural_sort_key(str(p)))
    if not paths:
        raise FileNotFoundError(f"No policy_action_trace.pt found under {run_dir}")

    dfs = []
    metadata_rows = []
    for p in paths:
        df, meta = load_trace_file(p, model, data)
        dfs.append(df)
        metadata_rows.append({"path": str(p), **meta})

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out_dir / "action_trace_long.csv", index=False)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(out_dir / "trace_metadata.csv", index=False)

    summary = make_episode_summary(all_df)
    summary = assign_modes(
        summary,
        manual_high=parse_episode_list(args.manual_high_eef_episodes),
        manual_low=parse_episode_list(args.manual_low_eef_episodes),
    )
    summary.to_csv(out_dir / "episode_summary.csv", index=False)

    mode_map = summary.set_index("episode_idx")["eef_mode"].to_dict()
    all_df["eef_mode"] = all_df["episode_idx"].map(mode_map)
    all_df.to_csv(out_dir / "action_trace_long_with_modes.csv", index=False)

    focus_chunks = parse_int_list(args.focus_chunks)
    focus_df = all_df[all_df["chunk_index"].isin(focus_chunks)].copy() if focus_chunks else all_df.copy()

    steps = steps_per_chunk(all_df, summary)
    steps.to_csv(out_dir / "steps_per_chunk.csv", index=False)

    action_cols = [
        c for c in all_df.columns
        if c.startswith("sent.") or c.startswith("to_send.") or c.startswith("policy.") or c.startswith("processed.")
    ]
    action_cols = sorted(action_cols, key=natural_sort_key)

    state_cols = sorted([c for c in all_df.columns if c.startswith("state.")], key=natural_sort_key)

    per_chunk = per_episode_chunk_action_summary(focus_df, action_cols, state_cols)
    per_chunk.to_csv(out_dir / "per_episode_chunk_action_summary.csv", index=False)

    # Compare per-step values grouped by chunk_index and chunk_step.
    action_sep = robust_effect_rows(
        focus_df,
        group_col="eef_mode",
        value_cols=action_cols,
        by_cols=["chunk_index", "chunk_step"],
    )
    action_sep.to_csv(out_dir / "top_mode_action_separators.csv", index=False)

    state_sep = robust_effect_rows(
        focus_df,
        group_col="eef_mode",
        value_cols=state_cols + ["eef_z"],
        by_cols=["chunk_index", "chunk_step"],
    )
    state_sep.to_csv(out_dir / "top_mode_state_separators.csv", index=False)

    # Also aggregate by chunk to find broad differences.
    chunk_action_sep = robust_effect_rows(
        per_chunk,
        group_col="eef_mode",
        value_cols=[c for c in per_chunk.columns if any(c.startswith(ac + ".") for ac in action_cols)],
        by_cols=["chunk_index"],
    )
    chunk_action_sep.to_csv(out_dir / "top_chunk_level_action_separators.csv", index=False)

    plot_eef_by_step(all_df, plot_dir)
    plot_eef_focus_chunks(all_df, focus_chunks, plot_dir)

    # Prefer sent actions for plot keys.
    sent_cols = sorted([c for c in action_cols if c.startswith("sent.")], key=natural_sort_key)
    plot_action_keys(focus_df, sent_cols, plot_dir, max_keys=args.plot_max_action_keys)

    write_report(
        out_dir=out_dir,
        run=str(run_dir),
        summary=summary,
        steps=steps,
        action_sep=action_sep,
        state_sep=state_sep,
        focus_chunks=focus_chunks,
    )

    print("[DONE] Policy action trace analysis complete.")
    print(f"[DONE] Output directory: {out_dir.resolve()}")
    print()
    print("Episodes:")
    print(summary[["episode_idx", "num_steps", "eef_z_max", "peak_step", "peak_chunk", "eef_mode"]].to_string(index=False))
    print()
    print("Key outputs:")
    print(f"  - {out_dir / 'diagnosis.md'}")
    print(f"  - {out_dir / 'episode_summary.csv'}")
    print(f"  - {out_dir / 'steps_per_chunk.csv'}")
    print(f"  - {out_dir / 'top_mode_action_separators.csv'}")
    print(f"  - {out_dir / 'top_mode_state_separators.csv'}")
    print(f"  - {plot_dir / 'eef_z_by_step.png'}")


if __name__ == "__main__":
    main()
