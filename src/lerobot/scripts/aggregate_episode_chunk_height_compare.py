#!/usr/bin/env python3
"""
aggregate_episode_chunk_height_compare.py

Aggregate per-episode predicted chunk max FK EEF height, optionally compare it
with actual rollout EEF height from the original debug run.

This version removes hardcoded paths.

Examples:

Predicted only:
python src/lerobot/scripts/aggregate_episode_chunk_height_compare.py \
  --pred-root analysis_sequence_chunk50_20260529_112158_high_10eps

Predicted vs actual from the original debug run:
python src/lerobot/scripts/aggregate_episode_chunk_height_compare.py \
  --pred-root analysis_sequence_chunk50_20260529_112158_high_10eps \
  --actual-run debug_runs/20260529_112158_high \
  --xml src/lerobot/scripts/follower.xml \
  --actual-source auto
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


PRED_SUMMARY_CANDIDATES = [
    "sequence_exact_predicted_action_chunk_max_fk_eef_z_summary.csv",
    "predicted_action_chunk_max_fk_eef_z_summary.csv",
]

PRED_MEAN_COL_CANDIDATES = [
    "predicted_chunk_fk_eef_z_max_mean",
    "fk_eef_z_max_mean",
    "predicted_max_fk_eef_z_mean",
]

PRED_STD_COL_CANDIDATES = [
    "predicted_chunk_fk_eef_z_max_std",
    "fk_eef_z_max_std",
    "predicted_max_fk_eef_z_std",
]

PRED_RANGE_COL_CANDIDATES = [
    "predicted_chunk_fk_eef_z_max_range",
    "fk_eef_z_max_range",
    "predicted_max_fk_eef_z_range",
]

ACTUAL_SUMMARY_CANDIDATES = [
    "actual_rollout_eef_z_summary.csv",
    "actual_lift_eef_z_summary.csv",
    "sequence_actual_eef_z_summary.csv",
    "actual_chunk_eef_z_summary.csv",
]

ACTUAL_VALUE_CANDIDATES = [
    "actual_eef_z_max",
    "actual_lift_eef_z_max",
    "rollout_eef_z_max",
    "actual_chunk_eef_z_max",
    "actual_max_eef_z",
    "eef_z_max",
    "max_eef_z",
]

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Calibration copied from the existing analysis scripts.
CALIB_REST_STATE_DEG = np.asarray(
    [-11.20879121, 97.00520833, 17.25596857, 100.0, -9.01098901, 47.38863287],
    dtype=np.float64,
)
CALIB_TARGET_REST_DEG = np.asarray(
    [91.7, 15.0, 40.0, 65.0, 0.0, -30.0],
    dtype=np.float64,
)
CALIB_ORDER = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
CALIB_SCALE = np.asarray([1.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
CALIB_SIGN = np.asarray([1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate predicted max FK EEF height across per-episode sequence replay outputs, "
            "and optionally compare with actual rollout EEF height from a debug_runs folder."
        )
    )
    parser.add_argument("--pred-root", required=True, help="Root containing episode_* analysis folders.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to --pred-root.")
    parser.add_argument(
        "--actual-run",
        default=None,
        help="Original debug run folder containing episode_*/ debug data. If omitted, only predicted values are aggregated.",
    )
    parser.add_argument("--xml", default=None, help="MuJoCo XML path, required when --actual-run is used.")
    parser.add_argument(
        "--actual-source",
        choices=[
            "auto",
            "action_trace",
            "action_trace_hybrid_wrist",
            "action_trace_leader_wrist",
            "action_trace_leader_all_joints",
            "saved_chunks",
            "summary",
            "none",
        ],
        default="auto",
        help=(
            "Actual EEF source. auto = action_trace if present, else saved_chunks, else summary. "
            "action_trace = policy_action_trace.pt. saved_chunks = debug_chunk_rawid_* snapshots."
        ),
    )
    parser.add_argument("--action-trace-name", default="policy_action_trace.pt")
    parser.add_argument("--chunk-glob", default="debug_chunk_rawid_*_observation_frame.pt")
    parser.add_argument("--title-prefix", default="")
    return parser.parse_args()


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_existing_file(directory: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    return None


def episode_idx_from_path(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", str(path))
    return int(match.group(1)) if match else None


def load_predicted_summary(ep_dir: Path) -> dict[str, Any] | None:
    pred_file = find_existing_file(ep_dir, PRED_SUMMARY_CANDIDATES)
    if pred_file is None:
        print(f"[WARN] Missing predicted summary in {ep_dir}")
        return None

    pred_df = pd.read_csv(pred_file)
    if len(pred_df) == 0:
        print(f"[WARN] Empty predicted summary: {pred_file}")
        return None

    mean_col = pick_col(pred_df, PRED_MEAN_COL_CANDIDATES)
    std_col = pick_col(pred_df, PRED_STD_COL_CANDIDATES)
    range_col = pick_col(pred_df, PRED_RANGE_COL_CANDIDATES)

    if mean_col is None:
        print(f"[WARN] Cannot find predicted mean column in {pred_file}. Columns={pred_df.columns.tolist()}")

    return {
        "predicted_summary_file": str(pred_file),
        "predicted_chunk_fk_eef_z_max_mean": float(pred_df.iloc[0][mean_col]) if mean_col else np.nan,
        "predicted_chunk_fk_eef_z_max_std": float(pred_df.iloc[0][std_col]) if std_col else np.nan,
        "predicted_chunk_fk_eef_z_max_range": float(pred_df.iloc[0][range_col]) if range_col else np.nan,
    }


def try_find_actual_value_from_summary(ep_dir: Path) -> tuple[float, str | None]:
    actual_file = find_existing_file(ep_dir, ACTUAL_SUMMARY_CANDIDATES)
    if actual_file is None:
        return np.nan, None
    df = pd.read_csv(actual_file)
    if len(df) == 0:
        return np.nan, str(actual_file)
    for col in ACTUAL_VALUE_CANDIDATES:
        if col in df.columns:
            return float(df.iloc[0][col]), str(actual_file)
    return np.nan, str(actual_file)


def state_to_q_rad(state_vec: np.ndarray) -> np.ndarray:
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)
    raw_deg = state_vec[:6].copy()
    raw_delta = raw_deg[CALIB_ORDER] - CALIB_REST_STATE_DEG[CALIB_ORDER]
    q_deg = CALIB_TARGET_REST_DEG + CALIB_SIGN * CALIB_SCALE * raw_delta
    return np.deg2rad(q_deg)


def eef_z_from_state(state_vec: np.ndarray, model: Any, data: Any) -> float:
    import mujoco
    q = state_to_q_rad(state_vec)
    mujoco.mj_resetData(model, data)
    for i, joint_name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        data.qpos[model.jnt_qposadr[jid]] = float(q[i])
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    if sid < 0:
        raise ValueError("Site not found: end_effector_site")
    return float(data.site_xpos[sid][2])


def to_numpy_safe(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None

def action_dict_to_vec(action_dict: Any) -> np.ndarray | None:
    if not isinstance(action_dict, dict):
        return None

    vals = []
    for key in ACTION_KEYS:
        if key not in action_dict:
            return None
        v = action_dict[key]
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        vals.append(float(arr[0]))

    return np.asarray(vals, dtype=np.float64)

def leader_action_to_vec(step: dict[str, Any]) -> np.ndarray | None:
    leader_action = step.get("leader_action", None)
    if not isinstance(leader_action, dict):
        return None

    vals = []
    for key in ACTION_KEYS:
        if key not in leader_action:
            return None

        v = leader_action[key]
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()

        vals.append(float(np.asarray(v, dtype=np.float64).reshape(-1)[0]))

    return np.asarray(vals, dtype=np.float64)


def build_hybrid_state_with_sent_wrist(step: dict[str, Any]) -> np.ndarray | None:
    """
    Hybrid FK estimate:
      - joints 0,1,2,4,5 from actual follower observation_state
      - joint 3 / wrist_flex from sent_action

    This is NOT true actual FK. It is a command-corrected estimate for diagnosing
    the fixed FK offset caused by invalid wrist_flex feedback.
    """
    obs = to_numpy_safe(step.get("observation_state", None))
    if obs is None:
        return None

    sent = action_dict_to_vec(step.get("sent_action", None))
    if sent is None:
        return None

    hybrid = np.asarray(obs, dtype=np.float64).reshape(-1)[:6].copy()
    hybrid[3] = float(sent[3])
    return hybrid


def find_observation_state(obj: Any) -> np.ndarray | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) == "observation.state":
                arr = to_numpy_safe(value)
                if arr is not None:
                    return arr
        for _, value in obj.items():
            out = find_observation_state(value)
            if out is not None:
                return out
        for key, value in obj.items():
            if "state" in str(key):
                arr = to_numpy_safe(value)
                if arr is not None:
                    return arr
    return None


def load_mujoco_model(xml_path: Path):
    import mujoco
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def actual_from_action_trace(ep_dir: Path, action_trace_name: str, model: Any, data: Any) -> dict[str, Any]:
    trace_path = ep_dir / action_trace_name
    if not trace_path.exists():
        return {"actual_rollout_eef_z_max": np.nan, "actual_source": "action_trace_missing", "actual_num_points": 0}
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    trace = payload.get("trace", []) if isinstance(payload, dict) else []
    z_values = []
    for step in trace:
        if not isinstance(step, dict):
            continue
        arr = to_numpy_safe(step.get("observation_state", None))
        if arr is None:
            continue
        try:
            z_values.append(eef_z_from_state(arr, model, data))
        except Exception as exc:
            warnings.warn(f"Failed FK for {trace_path}: {exc}")
    if not z_values:
        return {
            "actual_rollout_eef_z_max": np.nan,
            "actual_rollout_eef_z_min": np.nan,
            "actual_rollout_eef_z_range": np.nan,
            "actual_source": "action_trace_no_state",
            "actual_num_points": 0,
        }
    z = np.asarray(z_values, dtype=float)
    return {
        "actual_rollout_eef_z_max": float(np.max(z)),
        "actual_rollout_eef_z_min": float(np.min(z)),
        "actual_rollout_eef_z_range": float(np.max(z) - np.min(z)),
        "actual_source": "action_trace",
        "actual_num_points": int(len(z)),
    }

def actual_from_action_trace_hybrid_wrist(
    ep_dir: Path,
    action_trace_name: str,
    model: Any,
    data: Any,
) -> dict[str, Any]:
    trace_path = ep_dir / action_trace_name
    if not trace_path.exists():
        return {
            "actual_rollout_eef_z_max": np.nan,
            "actual_source": "action_trace_hybrid_wrist_missing",
            "actual_num_points": 0,
        }

    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    trace = payload.get("trace", []) if isinstance(payload, dict) else []

    z_values = []
    for step in trace:
        if not isinstance(step, dict):
            continue

        hybrid_state = build_hybrid_state_with_sent_wrist(step)
        if hybrid_state is None:
            continue

        try:
            z_values.append(eef_z_from_state(hybrid_state, model, data))
        except Exception as exc:
            warnings.warn(f"Failed hybrid wrist FK for {trace_path}: {exc}")

    if not z_values:
        return {
            "actual_rollout_eef_z_max": np.nan,
            "actual_rollout_eef_z_min": np.nan,
            "actual_rollout_eef_z_range": np.nan,
            "actual_source": "action_trace_hybrid_wrist_no_state",
            "actual_num_points": 0,
        }

    z = np.asarray(z_values, dtype=float)
    return {
        "actual_rollout_eef_z_max": float(np.max(z)),
        "actual_rollout_eef_z_min": float(np.min(z)),
        "actual_rollout_eef_z_range": float(np.max(z) - np.min(z)),
        "actual_source": "action_trace_hybrid_wrist",
        "actual_num_points": int(len(z)),
    }


def actual_from_action_trace_leader_wrist(
    ep_dir: Path,
    action_trace_name: str,
    model: Any,
    data: Any,
) -> dict[str, Any]:
    trace_path = ep_dir / action_trace_name
    if not trace_path.exists():
        return {
            "actual_rollout_eef_z_max": np.nan,
            "actual_rollout_eef_z_min": np.nan,
            "actual_rollout_eef_z_range": np.nan,
            "actual_source": "action_trace_leader_wrist_missing",
            "actual_num_points": 0,
        }

    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    trace = payload.get("trace", []) if isinstance(payload, dict) else []

    z_values = []
    for step in trace:
        if not isinstance(step, dict):
            continue

        obs = to_numpy_safe(step.get("observation_state", None))
        if obs is None:
            continue

        leader_wrist = step.get("leader_wrist_flex", None)

        if leader_wrist is None:
            leader_action = step.get("leader_action", None)
            if isinstance(leader_action, dict):
                leader_wrist = leader_action.get("wrist_flex.pos", None)

        if leader_wrist is None:
            continue

        if torch.is_tensor(leader_wrist):
            leader_wrist = leader_wrist.detach().cpu().numpy()

        leader_wrist = float(np.asarray(leader_wrist).reshape(-1)[0])

        hybrid_state = np.asarray(obs, dtype=np.float64).reshape(-1)[:6].copy()
        hybrid_state[3] = leader_wrist

        try:
            z_values.append(eef_z_from_state(hybrid_state, model, data))
        except Exception as exc:
            warnings.warn(f"Failed leader-wrist FK for {trace_path}: {exc}")

    if not z_values:
        return {
            "actual_rollout_eef_z_max": np.nan,
            "actual_rollout_eef_z_min": np.nan,
            "actual_rollout_eef_z_range": np.nan,
            "actual_source": "action_trace_leader_wrist_no_state",
            "actual_num_points": 0,
        }

    z = np.asarray(z_values, dtype=float)
    return {
        "actual_rollout_eef_z_max": float(np.max(z)),
        "actual_rollout_eef_z_min": float(np.min(z)),
        "actual_rollout_eef_z_range": float(np.max(z) - np.min(z)),
        "actual_source": "action_trace_leader_wrist",
        "actual_num_points": int(len(z)),
    }


def actual_from_action_trace_leader_all_joints(
    ep_dir: Path,
    action_trace_name: str,
    model: Any,
    data: Any,
) -> dict[str, Any]:
    trace_path = ep_dir / action_trace_name
    if not trace_path.exists():
        return {
            "actual_rollout_eef_z_max": np.nan,
            "actual_rollout_eef_z_min": np.nan,
            "actual_rollout_eef_z_range": np.nan,
            "actual_source": "action_trace_leader_all_joints_missing",
            "actual_num_points": 0,
        }

    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    trace = payload.get("trace", []) if isinstance(payload, dict) else []

    z_values = []
    for step in trace:
        if not isinstance(step, dict):
            continue

        leader_vec = leader_action_to_vec(step)
        if leader_vec is None:
            continue

        try:
            z_values.append(eef_z_from_state(leader_vec, model, data))
        except Exception as exc:
            warnings.warn(f"Failed leader-all-joints FK for {trace_path}: {exc}")

    if not z_values:
        return {
            "actual_rollout_eef_z_max": np.nan,
            "actual_rollout_eef_z_min": np.nan,
            "actual_rollout_eef_z_range": np.nan,
            "actual_source": "action_trace_leader_all_joints_no_state",
            "actual_num_points": 0,
        }

    z = np.asarray(z_values, dtype=float)
    return {
        "actual_rollout_eef_z_max": float(np.max(z)),
        "actual_rollout_eef_z_min": float(np.min(z)),
        "actual_rollout_eef_z_range": float(np.max(z) - np.min(z)),
        "actual_source": "action_trace_leader_all_joints",
        "actual_num_points": int(len(z)),
    }


def actual_from_saved_chunks(ep_dir: Path, chunk_glob: str, model: Any, data: Any) -> dict[str, Any]:
    chunk_files = sorted(ep_dir.glob(chunk_glob))
    z_values = []
    chunk_ids = []
    for path in chunk_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = find_observation_state(payload)
        if state is None:
            continue
        try:
            z_values.append(eef_z_from_state(state, model, data))
        except Exception as exc:
            warnings.warn(f"Failed FK for {path}: {exc}")
            continue
        match = re.search(r"debug_chunk_rawid_(\d+)_", path.name)
        chunk_ids.append(int(match.group(1)) if match else len(chunk_ids))
    if not z_values:
        return {
            "actual_rollout_eef_z_max": np.nan,
            "actual_rollout_eef_z_min": np.nan,
            "actual_rollout_eef_z_range": np.nan,
            "actual_rollout_eef_z_argmax_chunk": -1,
            "actual_source": "saved_chunks_no_state",
            "actual_num_points": 0,
        }
    z = np.asarray(z_values, dtype=float)
    argmax_i = int(np.argmax(z))
    return {
        "actual_rollout_eef_z_max": float(np.max(z)),
        "actual_rollout_eef_z_min": float(np.min(z)),
        "actual_rollout_eef_z_range": float(np.max(z) - np.min(z)),
        "actual_rollout_eef_z_argmax_chunk": int(chunk_ids[argmax_i]) if chunk_ids else -1,
        "actual_source": "saved_chunks",
        "actual_num_points": int(len(z)),
    }


def get_actual_for_episode(
    analysis_ep_dir: Path,
    actual_ep_dir: Path | None,
    actual_source: str,
    action_trace_name: str,
    chunk_glob: str,
    model: Any | None,
    data: Any | None,
) -> dict[str, Any]:
    if actual_source == "none":
        return {"actual_rollout_eef_z_max": np.nan, "actual_source": "none", "actual_num_points": 0}

    if actual_source == "summary":
        value, source_file = try_find_actual_value_from_summary(analysis_ep_dir)
        return {
            "actual_rollout_eef_z_max": value,
            "actual_source": "summary" if source_file else "summary_missing",
            "actual_summary_file": source_file,
            "actual_num_points": 1 if np.isfinite(value) else 0,
        }

    if actual_ep_dir is None:
        return {"actual_rollout_eef_z_max": np.nan, "actual_source": "actual_run_missing", "actual_num_points": 0}

    if actual_source == "action_trace_hybrid_wrist":
        return actual_from_action_trace_hybrid_wrist(
            actual_ep_dir,
            action_trace_name,
            model,
            data,
        )
        
    if actual_source == "action_trace_leader_wrist":
        return actual_from_action_trace_leader_wrist(
            actual_ep_dir,
            action_trace_name,
            model,
            data,
        )
    
    if actual_source == "action_trace_hybrid_wrist":
        return actual_from_action_trace_hybrid_wrist(
            actual_ep_dir,
            action_trace_name,
            model,
            data,
        )

    if actual_source == "action_trace_leader_wrist":
        return actual_from_action_trace_leader_wrist(
            actual_ep_dir,
            action_trace_name,
            model,
            data,
        )

    if actual_source == "action_trace_leader_all_joints":
        return actual_from_action_trace_leader_all_joints(
            actual_ep_dir,
            action_trace_name,
            model,
            data,
        )
            

    if actual_source in {"auto", "action_trace"}:
        result = actual_from_action_trace(actual_ep_dir, action_trace_name, model, data)
        if actual_source == "action_trace" or np.isfinite(result.get("actual_rollout_eef_z_max", np.nan)):
            return result

    if actual_source in {"auto", "saved_chunks"}:
        result = actual_from_saved_chunks(actual_ep_dir, chunk_glob, model, data)
        if actual_source == "saved_chunks" or np.isfinite(result.get("actual_rollout_eef_z_max", np.nan)):
            return result

    value, source_file = try_find_actual_value_from_summary(analysis_ep_dir)
    return {
        "actual_rollout_eef_z_max": value,
        "actual_source": "summary" if source_file else "actual_missing",
        "actual_summary_file": source_file,
        "actual_num_points": 1 if np.isfinite(value) else 0,
    }


def save_plots(df: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    prefix = f"{title_prefix}: " if title_prefix else ""

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode_idx"], df["predicted_chunk_fk_eef_z_max_mean"], marker="o", label="predicted 50-action chunk max FK EEF z")
    if not df["predicted_chunk_fk_eef_z_max_std"].isna().all():
        plt.errorbar(df["episode_idx"], df["predicted_chunk_fk_eef_z_max_mean"], yerr=df["predicted_chunk_fk_eef_z_max_std"].fillna(0), fmt="none", capsize=3)
    plt.xlabel("episode index")
    plt.ylabel("predicted max FK EEF z")
    plt.title(prefix + "Cross-episode predicted max FK EEF z")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "all_episodes_predicted_max_fk_eef_z.png", dpi=160, bbox_inches="tight")
    plt.close()

    if "actual_rollout_eef_z_max" in df.columns and not df["actual_rollout_eef_z_max"].isna().all():
        valid = df.dropna(subset=["predicted_chunk_fk_eef_z_max_mean", "actual_rollout_eef_z_max"]).copy()

        # Difference: predicted - actual
        valid["predicted_minus_actual_eef_z_m"] = (
            valid["predicted_chunk_fk_eef_z_max_mean"] - valid["actual_rollout_eef_z_max"]
        )
        valid["predicted_minus_actual_eef_z_cm"] = valid["predicted_minus_actual_eef_z_m"] * 100.0

        # Plot 1: predicted vs actual by episode, unit explicitly in meters
        plt.figure(figsize=(10, 5))
        plt.plot(
            df["episode_idx"],
            df["predicted_chunk_fk_eef_z_max_mean"],
            marker="o",
            label="predicted 50-action chunk max FK EEF z",
        )
        plt.plot(
            df["episode_idx"],
            df["actual_rollout_eef_z_max"],
            marker="o",
            label="actual rollout max FK EEF z",
        )
        plt.xlabel("episode index")
        plt.ylabel("Max FK EEF z (m)")
        plt.title(prefix + "Predicted vs actual max FK EEF z by episode")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "predicted_vs_actual_max_fk_eef_z_by_episode.png", dpi=160, bbox_inches="tight")
        plt.close()

        # Plot 2: predicted-vs-actual scatter, unit explicitly in meters
        if len(valid) > 0:
            plt.figure(figsize=(6, 5))
            plt.scatter(
                valid["predicted_chunk_fk_eef_z_max_mean"],
                valid["actual_rollout_eef_z_max"],
            )
            for _, row in valid.iterrows():
                plt.text(
                    row["predicted_chunk_fk_eef_z_max_mean"],
                    row["actual_rollout_eef_z_max"],
                    str(int(row["episode_idx"])),
                    fontsize=8,
                )

            corr = valid["predicted_chunk_fk_eef_z_max_mean"].corr(valid["actual_rollout_eef_z_max"])
            plt.xlabel("Predicted 50-action chunk max FK EEF z (m)")
            plt.ylabel("Actual rollout max FK EEF z (m)")
            plt.title(prefix + f"Predicted vs actual max EEF z | corr={corr:.3f}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "predicted_vs_actual_max_fk_eef_z_scatter.png", dpi=160, bbox_inches="tight")
            plt.close()

            # Plot 3: direct difference plot
            plt.figure(figsize=(10, 5))
            plt.plot(
                valid["episode_idx"],
                valid["predicted_minus_actual_eef_z_m"],
                marker="o",
                label="predicted - actual max FK EEF z",
            )
            plt.axhline(0.0, linestyle="--")
            plt.xlabel("episode index")
            plt.ylabel("Height difference (m)")
            plt.title(prefix + "Predicted minus actual max FK EEF z by episode")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "predicted_minus_actual_max_fk_eef_z_by_episode.png", dpi=160, bbox_inches="tight")
            plt.close()

            print("\n=== Predicted vs actual max FK EEF z difference ===")
            for _, row in valid.iterrows():
                print(
                    f"episode {int(row['episode_idx']):06d}: "
                    f"predicted={row['predicted_chunk_fk_eef_z_max_mean']:.6f} m, "
                    f"actual={row['actual_rollout_eef_z_max']:.6f} m, "
                    f"diff={row['predicted_minus_actual_eef_z_m']:.6f} m "
                    f"({row['predicted_minus_actual_eef_z_cm']:.2f} cm)"
                )


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir) if args.out_dir else pred_root
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_root.exists():
        raise FileNotFoundError(f"--pred-root does not exist: {pred_root}")

    actual_run = Path(args.actual_run) if args.actual_run else None
    need_fk = args.actual_source not in {"none", "summary"} and actual_run is not None
    model = data = None
    if need_fk:
        if args.xml is None:
            raise ValueError("--xml is required when computing actual EEF from debug run states.")
        model, data = load_mujoco_model(Path(args.xml))

    episode_dirs = sorted(pred_root.glob("episode_*"))
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found in {pred_root}")

    rows = []
    for analysis_ep_dir in episode_dirs:
        ep_idx = episode_idx_from_path(analysis_ep_dir)
        if ep_idx is None:
            continue
        pred = load_predicted_summary(analysis_ep_dir)
        if pred is None:
            continue
        actual_ep_dir = actual_run / f"episode_{ep_idx:06d}" if actual_run is not None else None
        actual = get_actual_for_episode(
            analysis_ep_dir=analysis_ep_dir,
            actual_ep_dir=actual_ep_dir,
            actual_source=args.actual_source,
            action_trace_name=args.action_trace_name,
            chunk_glob=args.chunk_glob,
            model=model,
            data=data,
        )
        row = {"episode_idx": int(ep_idx), "episode_dir": analysis_ep_dir.name}
        row.update(pred)
        row.update(actual)
        rows.append(row)

    all_df = pd.DataFrame(rows).sort_values("episode_idx")
    out_csv = out_dir / "all_episodes_chunk_height_comparison.csv"
    all_df.to_csv(out_csv, index=False)

    print(f"[*] Saved summary CSV: {out_csv}")
    display_cols = [
        "episode_idx",
        "episode_dir",
        "predicted_chunk_fk_eef_z_max_mean",
        "predicted_chunk_fk_eef_z_max_std",
        "predicted_chunk_fk_eef_z_max_range",
        "actual_rollout_eef_z_max",
        "actual_source",
        "actual_num_points",
    ]
    display_cols = [c for c in display_cols if c in all_df.columns]
    print(all_df[display_cols].to_string(index=False))

    pred_range = all_df["predicted_chunk_fk_eef_z_max_mean"].max() - all_df["predicted_chunk_fk_eef_z_max_mean"].min()
    print(f"\nPredicted cross-episode range: {pred_range:.8f} m = {pred_range * 100:.4f} cm")

    if "actual_rollout_eef_z_max" in all_df.columns and not all_df["actual_rollout_eef_z_max"].isna().all():
        actual_range = all_df["actual_rollout_eef_z_max"].max() - all_df["actual_rollout_eef_z_max"].min()
        corr = all_df["predicted_chunk_fk_eef_z_max_mean"].corr(all_df["actual_rollout_eef_z_max"])
        print(f"Actual cross-episode range: {actual_range:.8f} m = {actual_range * 100:.4f} cm")
        print(f"Predicted-vs-actual Pearson corr: {corr:.6f}")

    save_plots(all_df, out_dir, title_prefix=args.title_prefix)

    print("\nKey outputs:")
    print(f"  - {out_csv}")
    print(f"  - {out_dir / 'all_episodes_predicted_max_fk_eef_z.png'}")
    if "actual_rollout_eef_z_max" in all_df.columns and not all_df["actual_rollout_eef_z_max"].isna().all():
        print(f"  - {out_dir / 'predicted_vs_actual_max_fk_eef_z_by_episode.png'}")
        print(f"  - {out_dir / 'predicted_vs_actual_max_fk_eef_z_scatter.png'}")
    print("[*] Done.")


if __name__ == "__main__":
    main()