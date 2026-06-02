#!/usr/bin/env python3
"""
analyze_phase_eef_intervention.py

Phase-aware analysis for baseline/high/low LeRobot debug rollouts.

This script uses EEF z height computed from observation.state via MuJoCo FK, so
rollouts are compared by physical phase rather than raw chunk index only.

Expected input:
  debug_runs/xxx_baseline/episode_000000/debug_chunk*_observation_frame.pt
  debug_runs/xxx_high/episode_*/debug_chunk*_observation_frame.pt
  debug_runs/xxx_low/episode_*/debug_chunk*_observation_frame.pt

It ignores episode_*/policy_internal/*.pt.

Example:
  python src/lerobot/scripts/analyze_high_low_baseline.py \
    --baseline debug_runs/20260529_110106_baseline \
    --high debug_runs/20260529_112158_high \
    --low debug_runs/20260529_112911_low \
    --xml src/lerobot/scripts/follower.xml \
    --out analysis_phase_eef \
    --height-action-dims 1,2 \
    --include-images
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    import mujoco
except Exception as e:
    mujoco = None
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# FK calibration copied from physical_neuron_finding_eef.py
# -----------------------------------------------------------------------------

CALIB_REST_STATE_DEG = np.asarray(
    [-11.20879121, 97.00520833, 17.25596857, 100.0, -9.01098901, 47.38863287],
    dtype=np.float64,
)
CALIB_TARGET_REST_DEG = np.asarray([91.7, 15.0, 40.0, 65.0, 0.0, -30.0], dtype=np.float64)
CALIB_ORDER = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
CALIB_SCALE = np.asarray([1.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
CALIB_SIGN = np.asarray([1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float64)


def state_to_q_rad(state_vec: np.ndarray) -> np.ndarray:
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)
    if state_vec.size < 6:
        raise ValueError(f"observation.state must have at least 6 dims, got {state_vec.shape}")
    raw_deg = state_vec[:6].copy()
    raw_delta = raw_deg[CALIB_ORDER] - CALIB_REST_STATE_DEG[CALIB_ORDER]
    q_deg = CALIB_TARGET_REST_DEG + CALIB_SIGN * CALIB_SCALE * raw_delta
    return np.deg2rad(q_deg)


def load_fk_model(xml_path: Path):
    if mujoco is None:
        raise ImportError(
            "Could not import mujoco. Run this script in the same env where your "
            f"physical_neuron_finding_eef.py works. Original error: {MUJOCO_IMPORT_ERROR}"
        )
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def compute_eef_z(state_vec: np.ndarray, model, data) -> float:
    q = state_to_q_rad(state_vec)
    mujoco.mj_resetData(model, data)
    for i, joint_name in enumerate(["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint not found in XML: {joint_name}")
        data.qpos[model.jnt_qposadr[jid]] = float(q[i])
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    if sid < 0:
        raise ValueError("Site not found in XML: end_effector_site")
    return float(data.site_xpos[sid][2])


# -----------------------------------------------------------------------------
# Loading / flattening helpers
# -----------------------------------------------------------------------------

def natural_key(s: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten_numeric(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(flatten_numeric(v, f"{prefix}.{i}" if prefix else str(i)))
        return out
    try:
        arr = to_numpy(value)
        if not np.issubdtype(arr.dtype, np.number):
            return out
    except Exception:
        return out
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        out[prefix] = float(arr)
    else:
        for i, v in enumerate(arr.reshape(-1)):
            out[f"{prefix}.{i}" if prefix else str(i)] = float(v)
    return out


def image_feature(img: Any, size: tuple[int, int] = (32, 24)) -> np.ndarray | None:
    try:
        arr = to_numpy(img)
    except Exception:
        return None
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and np.nanmax(arr) <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 2:
        gray = arr
    else:
        return None
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0).reshape(-1)


def episode_idx(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def save_idx_from_name(path: Path) -> int:
    m = re.search(r"_(\d+)_observation_frame\.pt$", path.name)
    return int(m.group(1)) if m else -1


def find_chunk_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("episode_*/debug_chunk*_observation_frame.pt"), key=lambda p: natural_key(str(p)))
    return [p for p in paths if "policy_internal" not in p.parts]


def matching_images(pt_path: Path) -> list[Path]:
    stem = pt_path.name.replace("_observation_frame.pt", "")
    return sorted(pt_path.parent.glob(f"{stem}_observation_images_*.png"), key=lambda p: natural_key(p.name))


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_condition(condition: str, root: Path, model, data, include_images: bool) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    action_rows: list[dict] = []
    image_features: dict[str, np.ndarray] = {}
    paths = find_chunk_paths(root)
    if not paths:
        raise FileNotFoundError(f"No debug chunk .pt files found under {root}")

    for p in paths:
        obj = load_pt(p)
        meta = obj.get("metadata", {}) or {}
        state = None
        for key in ["observation.state", "observation_state", "state"]:
            if key in obj:
                state = obj[key]
                break
        if state is None:
            for key in obj.keys():
                if isinstance(key, str) and key.endswith("state"):
                    state = obj[key]
                    break
        if state is None:
            raise KeyError(f"No observation.state found in {p}")

        state_vec = np.asarray(to_numpy(state), dtype=np.float64).reshape(-1)
        z = compute_eef_z(state_vec, model, data)
        ep = p.parent.name
        sidx = int(meta.get("save_idx", obj.get("chunk_id", save_idx_from_name(p))))
        raw_id = meta.get("raw_chunk_id", obj.get("chunk_id", None))

        action_dict = flatten_numeric(obj.get("action_values", None), "action_values")
        obs_dict = {}
        img_feats = []
        for k, v in obj.items():
            if k in {"chunk_id", "keys", "action_values", "metadata"}:
                continue
            if isinstance(k, str) and k.startswith("observation.images."):
                if include_images:
                    feat = image_feature(v)
                    if feat is not None:
                        img_feats.append(feat)
                continue
            obs_dict.update(flatten_numeric(v, k))
        if include_images and img_feats:
            image_features[str(p)] = np.concatenate(img_feats).astype(np.float32)
        else:
            image_features[str(p)] = np.zeros((0,), dtype=np.float32)

        rows.append({
            "condition": condition,
            "condition_root": str(root),
            "episode": ep,
            "episode_idx": episode_idx(ep),
            "save_idx": sidx,
            "raw_chunk_id": raw_id,
            "eef_z": z,
            "path": str(p),
            "filename": p.name,
            "task": str(meta.get("task", "")),
            "metadata_intervention": str(meta.get("intervention", "unknown")),
            "state_dim": len(state_vec),
            "num_action_dims": len(action_dict),
            "num_obs_dims": len(obs_dict) + len(image_features[str(p)]),
        })

        for key, value in action_dict.items():
            dim = int(key.split(".")[-1]) if key.split(".")[-1].isdigit() else -1
            action_rows.append({
                "condition": condition,
                "episode": ep,
                "episode_idx": episode_idx(ep),
                "save_idx": sidx,
                "action_key": key,
                "action_dim": dim,
                "action_value": value,
                "path": str(p),
            })

    return rows, action_rows, image_features


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------

def add_episode_phase(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (_, _), g in df.sort_values(["condition", "episode_idx", "save_idx"]).groupby(["condition", "episode"], sort=False):
        g = g.copy().sort_values("save_idx")
        z = g["eef_z"].to_numpy(float)
        zmin, zmax = float(np.min(z)), float(np.max(z))
        zrng = max(zmax - zmin, 1e-8)
        z0 = float(z[0])
        g["eef_z_start"] = z0
        g["eef_z_delta_from_start"] = g["eef_z"] - z0
        g["eef_z_min"] = zmin
        g["eef_z_max"] = zmax
        g["eef_z_range"] = zmax - zmin
        g["eef_z_norm_in_episode"] = (g["eef_z"] - zmin) / zrng
        peak_idx = g["eef_z"].idxmax()
        g["peak_chunk"] = int(g.loc[peak_idx, "save_idx"])
        phase = []
        for v in g["eef_z_norm_in_episode"]:
            if v < 0.20:
                phase.append("low_height")
            elif v < 0.55:
                phase.append("mid_height")
            elif v < 0.85:
                phase.append("high_height")
            else:
                phase.append("near_peak")
        g["eef_height_phase"] = phase
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def make_episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, ep), g in df.groupby(["condition", "episode"]):
        g = g.sort_values("save_idx")
        z = g["eef_z"].to_numpy(float)
        rows.append({
            "condition": cond,
            "episode": ep,
            "episode_idx": int(g["episode_idx"].iloc[0]),
            "num_chunks": len(g),
            "first_chunk": int(g["save_idx"].min()),
            "last_chunk": int(g["save_idx"].max()),
            "eef_z_start": float(z[0]),
            "eef_z_end": float(z[-1]),
            "eef_z_min": float(np.min(z)),
            "eef_z_max": float(np.max(z)),
            "eef_z_range": float(np.max(z) - np.min(z)),
            "eef_z_p90": float(np.percentile(z, 90)),
            "peak_chunk": int(g.loc[g["eef_z"].idxmax(), "save_idx"]),
            "phase_sequence": " > ".join(g["eef_height_phase"].tolist()),
        })
    return pd.DataFrame(rows).sort_values(["condition", "episode_idx"])


def high_bimodal(ep: pd.DataFrame) -> pd.DataFrame:
    high = ep[ep["condition"] == "high"].copy()
    if high.empty:
        return high
    thr = float(high["eef_z_p90"].median())
    high["high_eef_threshold"] = thr
    high["high_eef_group"] = np.where(high["eef_z_p90"] >= thr, "high_eef_mode", "low_eef_mode")
    return high.sort_values(["high_eef_group", "episode_idx"])


def select_phase_aligned(df: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    rows = []
    for (_, _), g in df.groupby(["condition", "episode"]):
        g = g.sort_values("save_idx").copy()
        for t in targets:
            idx = (g["eef_z_norm_in_episode"] - t).abs().idxmin()
            row = g.loc[idx].to_dict()
            row["phase_target"] = t
            row["phase_error"] = abs(float(row["eef_z_norm_in_episode"]) - t)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["phase_target", "condition", "episode_idx"])


def phase_action_stats(selected: pd.DataFrame, action_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["condition", "episode", "episode_idx", "save_idx", "phase_target", "phase_error", "eef_z", "path"]
    merged = selected[cols].merge(action_df, on=["condition", "episode", "episode_idx", "save_idx", "path"], how="left")
    stats = (
        merged.groupby(["condition", "phase_target", "action_dim", "action_key"])["action_value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    stats["range"] = stats["max"] - stats["min"]
    return merged, stats.sort_values(["std", "range"], ascending=False)


def zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (x - mu) / sd


def pairwise_l2(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0, 0))
    x2 = np.sum(x * x, axis=1, keepdims=True)
    d2 = x2 + x2.T - 2.0 * x @ x.T
    return np.sqrt(np.maximum(d2, 0.0))


def make_matrix(rows: pd.DataFrame, long_df: pd.DataFrame, value_col: str, key_col: str) -> tuple[np.ndarray, list[str]]:
    keys = sorted(long_df[key_col].dropna().unique(), key=natural_key)
    path_to_i = {p: i for i, p in enumerate(rows["path"].tolist())}
    mat = np.zeros((len(rows), len(keys)), dtype=np.float32)
    key_to_j = {k: j for j, k in enumerate(keys)}
    for r in long_df.itertuples(index=False):
        if r.path in path_to_i and getattr(r, key_col) in key_to_j:
            mat[path_to_i[r.path], key_to_j[getattr(r, key_col)]] = float(getattr(r, value_col))
    return mat, keys


def phase_mismatch(selected: pd.DataFrame, action_df: pd.DataFrame, image_features: dict[str, np.ndarray], include_images: bool, max_pairs: int) -> pd.DataFrame:
    action_mat, _ = make_matrix(selected, action_df, "action_value", "action_key")
    action_z = zscore(action_mat)

    obs_cols = selected[["eef_z", "eef_z_norm_in_episode", "phase_target"]].to_numpy(np.float32)
    if include_images:
        max_dim = max((v.size for v in image_features.values()), default=0)
        img_mat = np.zeros((len(selected), max_dim), dtype=np.float32)
        for i, p in enumerate(selected["path"]):
            v = image_features.get(p, np.zeros((0,), dtype=np.float32))
            if v.size:
                img_mat[i, : v.size] = v
        obs_mat = np.concatenate([obs_cols, img_mat], axis=1)
    else:
        obs_mat = obs_cols
    obs_z = zscore(obs_mat)

    od = pairwise_l2(obs_z)
    ad = pairwise_l2(action_z)

    recs = selected.reset_index(drop=True)
    pairs = []
    for i, a in recs.iterrows():
        candidates = [j for j, b in recs.iterrows() if j != i and b["episode"] != a["episode"] and b["phase_target"] == a["phase_target"]]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: od[i, k])
        b = recs.iloc[j]
        pairs.append({
            "condition": a["condition"],
            "episode": a["episode"],
            "save_idx": a["save_idx"],
            "phase_target": a["phase_target"],
            "eef_z": a["eef_z"],
            "path": a["path"],
            "nearest_condition": b["condition"],
            "nearest_episode": b["episode"],
            "nearest_save_idx": b["save_idx"],
            "nearest_eef_z": b["eef_z"],
            "nearest_path": b["path"],
            "obs_distance": float(od[i, j]),
            "action_distance": float(ad[i, j]),
            "same_condition": int(a["condition"] == b["condition"]),
        })
    out = pd.DataFrame(pairs)
    if out.empty:
        return out
    out["obs_distance_rank_pct"] = out["obs_distance"].rank(pct=True)
    out["action_distance_rank_pct"] = out["action_distance"].rank(pct=True)
    out["mismatch_score"] = (1.0 - out["obs_distance_rank_pct"]) * out["action_distance_rank_pct"]
    return out.sort_values("mismatch_score", ascending=False).head(max_pairs)


def copy_mismatch_images(mismatch: pd.DataFrame, out_dir: Path, n: int) -> None:
    if mismatch.empty or n <= 0:
        return
    dst_root = out_dir / "inspect_phase_mismatch_images"
    dst_root.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(mismatch.head(n).itertuples(index=False), start=1):
        a = Path(row.path)
        b = Path(row.nearest_path)
        folder = dst_root / f"rank_{rank:03d}_{row.condition}_{row.episode}_c{int(row.save_idx)}__vs__{row.nearest_condition}_{row.nearest_episode}_c{int(row.nearest_save_idx)}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "pair_info.json").write_text(json.dumps(row._asdict(), indent=2, default=str))
        for prefix, p in [("A", a), ("B", b)]:
            for img in matching_images(p):
                shutil.copy2(img, folder / f"{prefix}_{img.name}")


# -----------------------------------------------------------------------------
# Plots and report
# -----------------------------------------------------------------------------

def plot_eef_chunk(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(10, 5))
    for cond, g in df.groupby("condition"):
        s = g.groupby("save_idx")["eef_z"].agg(["mean", "std"]).reset_index().sort_values("save_idx")
        x, y, e = s["save_idx"].to_numpy(), s["mean"].to_numpy(), s["std"].fillna(0).to_numpy()
        plt.plot(x, y, marker="o", label=cond)
        plt.fill_between(x, y - e, y + e, alpha=0.15)
    plt.xlabel("raw chunk index")
    plt.ylabel("EEF z height")
    plt.title("EEF height by raw chunk index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "eef_height_by_raw_chunk.png", dpi=160)
    plt.close()


def plot_episode_eef(ep: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(10, 5))
    off = {"baseline": -0.18, "high": 0.0, "low": 0.18}
    for cond, g in ep.groupby("condition"):
        plt.scatter(g["episode_idx"] + off.get(cond, 0), g["eef_z_p90"], label=cond)
    plt.xlabel("episode index")
    plt.ylabel("episode EEF p90 height from saved chunks")
    plt.title("Episode-level EEF height")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "episode_eef_p90_height.png", dpi=160)
    plt.close()


def plot_action_dims(merged: pd.DataFrame, dims: list[int], out: Path) -> None:
    for dim in dims:
        sub = merged[merged["action_dim"] == dim]
        if sub.empty:
            continue
        plt.figure(figsize=(10, 5))
        for cond, g in sub.groupby("condition"):
            s = g.groupby("phase_target")["action_value"].agg(["mean", "std"]).reset_index().sort_values("phase_target")
            x, y, e = s["phase_target"].to_numpy(), s["mean"].to_numpy(), s["std"].fillna(0).to_numpy()
            plt.plot(x, y, marker="o", label=cond)
            plt.fill_between(x, y - e, y + e, alpha=0.15)
        plt.xlabel("normalized EEF phase target")
        plt.ylabel(f"action dim {dim}")
        plt.title(f"Phase-aligned action dim {dim}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"phase_aligned_action_dim_{dim}.png", dpi=160)
        plt.close()


def plot_mismatch(mismatch: pd.DataFrame, out: Path) -> None:
    if mismatch.empty:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(mismatch["obs_distance"], mismatch["action_distance"])
    plt.xlabel("phase-aware observation distance")
    plt.ylabel("action distance")
    plt.title("Same-phase observation/action mismatch")
    plt.tight_layout()
    plt.savefig(out / "phase_obs_distance_vs_action_distance.png", dpi=160)
    plt.close()


def write_report(out: Path, df: pd.DataFrame, ep: pd.DataFrame, high_modes: pd.DataFrame, stats: pd.DataFrame, mismatch: pd.DataFrame, dims: list[int]) -> None:
    lines = []
    lines.append("# Phase-aware EEF intervention diagnosis")
    lines.append("")
    lines.append("This analysis uses EEF z height from observation.state via MuJoCo FK, so it can diagnose whether raw chunk index was comparing different task phases.")
    lines.append("")
    lines.append("## Loaded data")
    lines.append("```")
    lines.append(df.groupby("condition")["path"].count().to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Episode EEF summary")
    lines.append("```")
    lines.append(ep[["condition", "episode", "num_chunks", "eef_z_p90", "eef_z_max", "eef_z_range", "peak_chunk", "phase_sequence"]].to_string(index=False))
    lines.append("```")
    lines.append("")
    if not high_modes.empty:
        lines.append("## High intervention EEF modes")
        lines.append("```")
        lines.append(high_modes[["episode", "episode_idx", "eef_z_p90", "eef_z_max", "eef_z_range", "peak_chunk", "high_eef_group"]].to_string(index=False))
        lines.append("```")
        lines.append("")
    lines.append("## Most unstable phase-aligned action dimensions")
    lines.append("```")
    lines.append(stats.head(30).to_string(index=False))
    lines.append("```")
    lines.append("")
    if not mismatch.empty:
        lines.append("## Same-phase similar observation / different action pairs")
        lines.append("```")
        cols = ["condition", "episode", "save_idx", "phase_target", "eef_z", "nearest_condition", "nearest_episode", "nearest_save_idx", "nearest_eef_z", "obs_distance", "action_distance", "mismatch_score"]
        lines.append(mismatch.head(20)[cols].to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("1. If `eef_height_by_raw_chunk.png` has high spread at a chunk index, raw chunk alignment is invalid.")
    lines.append("2. If high episodes split into high_eef_mode and low_eef_mode, high intervention is causing task-progress / EEF-height bifurcation.")
    lines.append("3. If phase-aligned action variance is still high, the policy/intervention is unstable even after EEF-phase alignment.")
    lines.append("4. If phase-aligned variance becomes small, the previous chunk-index variance was mainly phase misalignment.")
    lines.append("5. Inspect `inspect_phase_mismatch_images/` to see whether same-phase images are visually similar.")
    if dims:
        lines.append(f"6. Candidate height/action dims plotted: {dims}")
    (out / "diagnosis_phase_eef.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--high", required=True)
    parser.add_argument("--low", required=True)
    parser.add_argument("--xml", required=True, help="Path to follower.xml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--phase-targets", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--height-action-dims", default="")
    parser.add_argument("--max-mismatch-pairs", type=int, default=100)
    parser.add_argument("--copy-top-images", type=int, default=12)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    xml = Path(args.xml)
    if not xml.exists():
        xml = Path.cwd() / args.xml
    model, data = load_fk_model(xml)

    all_rows, all_actions, image_features = [], [], {}
    for cond, root in [("baseline", args.baseline), ("high", args.high), ("low", args.low)]:
        rows, actions, imgs = load_condition(cond, Path(root), model, data, args.include_images)
        all_rows.extend(rows)
        all_actions.extend(actions)
        image_features.update(imgs)

    df = add_episode_phase(pd.DataFrame(all_rows))
    action_df = pd.DataFrame(all_actions)
    df.to_csv(out / "chunk_eef_table.csv", index=False)
    action_df.to_csv(out / "actions_long.csv", index=False)

    ep = make_episode_summary(df)
    ep.to_csv(out / "episode_eef_summary.csv", index=False)
    high_modes = high_bimodal(ep)
    high_modes.to_csv(out / "high_episode_bimodal_report.csv", index=False)

    targets = [float(x.strip()) for x in args.phase_targets.split(",") if x.strip()]
    selected = select_phase_aligned(df, targets)
    selected.to_csv(out / "phase_aligned_chunks.csv", index=False)
    merged, stats = phase_action_stats(selected, action_df)
    merged.to_csv(out / "phase_aligned_actions_long.csv", index=False)
    stats.to_csv(out / "phase_aligned_action_stats.csv", index=False)

    dims = [int(x.strip()) for x in args.height_action_dims.split(",") if x.strip()]

    # Key map.
    action_keys = sorted(action_df["action_key"].dropna().unique(), key=natural_key)
    pd.DataFrame([{"action_dim": int(k.split(".")[-1]), "action_key": k} for k in action_keys]).to_csv(out / "action_key_map.csv", index=False)

    mismatch = phase_mismatch(selected, action_df, image_features, args.include_images, args.max_mismatch_pairs)
    mismatch.to_csv(out / "same_phase_obs_action_mismatch.csv", index=False)
    copy_mismatch_images(mismatch, out, args.copy_top_images)

    plot_eef_chunk(df, plots)
    plot_episode_eef(ep, plots)
    plot_action_dims(merged, dims, plots)
    plot_mismatch(mismatch, plots)
    write_report(out, df, ep, high_modes, stats, mismatch, dims)

    print("[DONE] Phase-aware EEF analysis complete")
    print(f"[DONE] Output: {out.resolve()}")
    print("\nLoaded chunks:")
    print(df.groupby("condition")["path"].count().to_string())
    print("\nEpisodes:")
    print(df.groupby("condition")["episode"].nunique().to_string())
    print("\nKey outputs:")
    print(f"  - {out / 'diagnosis_phase_eef.md'}")
    print(f"  - {out / 'chunk_eef_table.csv'}")
    print(f"  - {out / 'episode_eef_summary.csv'}")
    print(f"  - {out / 'high_episode_bimodal_report.csv'}")
    print(f"  - {out / 'phase_aligned_action_stats.csv'}")
    print(f"  - {out / 'same_phase_obs_action_mismatch.csv'}")
    print(f"  - {plots / 'eef_height_by_raw_chunk.png'}")
    print(f"  - {plots / 'episode_eef_p90_height.png'}")


if __name__ == "__main__":
    main()