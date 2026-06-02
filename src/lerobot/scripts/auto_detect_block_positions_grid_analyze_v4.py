#!/usr/bin/env python3
"""
auto_detect_block_positions_grid_analyze_v4.py

Automatic dataset robustness audit for RANDOMLY placed red-cube pick-and-place datasets.

Compared with v2, this version adds continuous spatial/grid analysis:
  - Detect red cube center (cx, cy) from top camera using traditional CV.
  - Divide workspace ROI into grid cells.
  - Analyze dataset health per grid cell, not only k-means clusters.
  - Identify weak spatial regions:
      * too few demos
      * inconsistent EEF peak height
      * multimodal pre-lift wrist/elbow strategy
      * outlier episodes
  - Generate heatmaps and actionable recording advice.

Recommended command:
  python src/lerobot/scripts/auto_detect_block_positions_grid_analyze_v4.py \
    --local-dataset ~/.cache/huggingface/lerobot/ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
    --xml src/lerobot/scripts/follower.xml \
    --out analysis_dataset_grid_robustness_test2 \
    --camera-key observation.images.top \
    --roi 250,120,570,340 \
    --grid-cols 5 \
    --grid-rows 4 \
    --detect-frame-indices 0,3,6,9 \
    --pre-peak-window 45:35

Outputs:
  - frame_long.csv
  - episode_metrics.csv
  - cube_detection.csv
  - grid_cell_metrics.csv
  - grid_cell_advice.csv
  - outlier_episodes.csv
  - dataset_health_summary.json
  - diagnosis.md
  - overlays/*.png
  - plots/grid_count_heatmap.png
  - plots/grid_eef_range_heatmap.png
  - plots/grid_wrist_iqr_heatmap.png
  - plots/grid_elbow_iqr_heatmap.png
  - plots/grid_advice_severity_heatmap.png
  - plots/cube_positions_grid.png
  - plots/prelift_wrist_elbow_by_grid.png

Notes:
  - Supports LeRobot v3 chunked videos using meta/episodes:
      videos/{camera_key}/chunk_index
      videos/{camera_key}/file_index
      videos/{camera_key}/from_timestamp
  - Uses ffmpeg CLI to decode AV1 videos.
  - Uses traditional CV, not VLM:
      ROI crop -> HSV red threshold -> morphology -> contour -> cube center.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    snapshot_download = None
    HF_IMPORT_ERROR = exc
else:
    HF_IMPORT_ERROR = None

try:
    import mujoco
except Exception as exc:
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None


# ==============================================================================
# FK calibration
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

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


# ==============================================================================
# Generic helpers
# ==============================================================================

def natural_sort_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if text is None or str(text).strip() == "":
        return None
    vals = [int(x.strip()) for x in str(text).split(",")]
    if len(vals) != 4:
        raise ValueError("--roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI: {text}")
    return x1, y1, x2, y2


def parse_pre_peak_window(text: str) -> tuple[int, int]:
    """
    Parse '45:35' as offsets before peak: [peak-45, peak-35], inclusive.
    """
    a, b = str(text).split(":", 1)
    a, b = int(a), int(b)
    return max(a, b), min(a, b)


def parse_vector_cell(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().reshape(-1)

    arr = np.asarray(x)
    if arr.dtype == object and arr.ndim == 0:
        x = arr.item()
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().reshape(-1)
        arr = np.asarray(x)

    if arr.dtype.kind in {"U", "S"}:
        import ast
        return np.asarray(ast.literal_eval(str(x)), dtype=np.float64).reshape(-1)

    return np.asarray(arr, dtype=np.float64).reshape(-1)


def find_col(columns: list[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"None of candidates {candidates} found. Available columns: {columns}")


def iqr_series(x: pd.Series) -> float:
    vals = x.dropna().to_numpy(dtype=np.float64)
    if vals.size == 0:
        return np.nan
    return float(np.percentile(vals, 75) - np.percentile(vals, 25))


def robust_mad_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if not np.isfinite(mad) or mad < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return 0.6745 * (values - med) / mad


# ==============================================================================
# Dataset loading
# ==============================================================================

def download_or_use_repo(repo_id: str | None, local_dataset: str | None, cache_dir: str | None, include_videos: bool) -> Path:
    if local_dataset is not None:
        p = Path(local_dataset).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Local dataset path not found: {p}")
        return p

    if repo_id is None:
        raise ValueError("Provide either --repo-id or --local-dataset.")

    if snapshot_download is None:
        raise ImportError(
            "huggingface_hub is not available. Install it or run inside your lerobot env. "
            f"Original error: {HF_IMPORT_ERROR}"
        )

    allow = ["meta/**", "data/**/*.parquet", "*.json", "README.md"]
    if include_videos:
        allow += ["videos/**/*.mp4", "videos/**/*.avi", "videos/**/*.mov", "videos/**/*.mkv"]

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=cache_dir,
        allow_patterns=allow,
        ignore_patterns=[] if include_videos else ["videos/**", "*.mp4", "*.avi", "*.png", "*.jpg", "*.jpeg"],
    )
    return Path(local)


def load_lerobot_parquets(repo_path: Path) -> pd.DataFrame:
    files = sorted(repo_path.glob("data/**/*.parquet"), key=lambda p: natural_sort_key(str(p)))
    if not files:
        files = sorted(repo_path.glob("**/*.parquet"), key=lambda p: natural_sort_key(str(p)))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {repo_path}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["_source_file"] = str(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_episode_metadata(repo_path: Path) -> pd.DataFrame:
    files = sorted(repo_path.glob("meta/episodes/**/*.parquet"), key=lambda p: natural_sort_key(str(p)))
    if not files:
        raise FileNotFoundError(f"No meta/episodes parquet files found under {repo_path}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["_episode_meta_file"] = str(f)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    if "episode_index" not in out.columns:
        raise KeyError("episode metadata must contain episode_index")
    return out.sort_values("episode_index").reset_index(drop=True)


def load_dataset_info(repo_path: Path) -> dict[str, Any]:
    p = repo_path / "meta/info.json"
    if not p.exists():
        p = repo_path / "info.json"
    if not p.exists():
        raise FileNotFoundError(f"Could not find meta/info.json or info.json under {repo_path}")
    return json.loads(p.read_text())


# ==============================================================================
# FK / EEF
# ==============================================================================

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
            "mujoco import failed. Run in your lerobot env. "
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


def normalize_dataset_df(raw: pd.DataFrame, model, data) -> pd.DataFrame:
    cols = list(raw.columns)
    ep_col = find_col(cols, ["episode_index", "episode.idx", "episode"])
    frame_col = find_col(cols, ["frame_index", "frame.idx", "index"])
    state_col = find_col(cols, ["observation.state", "observation_state", "state"])
    action_col = find_col(cols, ["action", "actions"])

    rows = []
    for i, row in raw.iterrows():
        state = parse_vector_cell(row[state_col])
        action = parse_vector_cell(row[action_col])
        if state.size < 6 or action.size < 6:
            continue

        ep = int(np.asarray(row[ep_col]).reshape(-1)[0])
        frame = int(np.asarray(row[frame_col]).reshape(-1)[0])

        out = {
            "episode_idx": ep,
            "frame_index": frame,
            "source_row": int(i),
            "eef_z": eef_z_from_state(state[:6], model, data),
        }
        for j, name in enumerate(JOINT_NAMES):
            out[f"state.{name}"] = float(state[j])
            out[f"action.{name}"] = float(action[j])
        rows.append(out)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable rows after parsing dataset parquet.")
    df = df.sort_values(["episode_idx", "frame_index"]).reset_index(drop=True)
    df["episode_step"] = df.groupby("episode_idx").cumcount()
    return df


# ==============================================================================
# Video frame extraction + cube detection
# ==============================================================================

def episode_video_path_and_start_time(
    repo_path: Path,
    ep_row: pd.Series,
    camera_key: str,
    info: dict[str, Any],
) -> tuple[Path, float, float]:
    prefix = f"videos/{camera_key}"
    chunk_col = f"{prefix}/chunk_index"
    file_col = f"{prefix}/file_index"
    from_col = f"{prefix}/from_timestamp"
    to_col = f"{prefix}/to_timestamp"

    missing = [c for c in [chunk_col, file_col, from_col, to_col] if c not in ep_row.index]
    if missing:
        raise KeyError(
            f"Missing video metadata columns for camera_key={camera_key}: {missing}. "
            f"Available video columns include: {[c for c in ep_row.index if str(c).startswith('videos/')]}"
        )

    chunk_idx = int(ep_row[chunk_col])
    file_idx = int(ep_row[file_col])
    from_ts = float(ep_row[from_col])
    to_ts = float(ep_row[to_col])

    template = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    rel = template.format(video_key=camera_key, chunk_index=chunk_idx, file_index=file_idx)
    video_path = repo_path / rel

    if not video_path.exists():
        video_path = repo_path / "videos" / camera_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found for episode {int(ep_row['episode_index'])}: {video_path}")

    return video_path, from_ts, to_ts


def read_video_frame_at_time(video_path: Path, timestamp_s: float) -> np.ndarray | None:
    """
    Extract one frame using ffmpeg CLI. This avoids OpenCV AV1 decode issues.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_png = Path(f.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-ss",
            f"{max(float(timestamp_s), 0.0):.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(tmp_png),
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            return None

        if not tmp_png.exists() or tmp_png.stat().st_size == 0:
            return None

        return cv2.imread(str(tmp_png), cv2.IMREAD_COLOR)

    finally:
        try:
            tmp_png.unlink(missing_ok=True)
        except Exception:
            pass


def detect_red_cube_center(
    frame_bgr: np.ndarray,
    min_area: float = 80.0,
    roi: tuple[int, int, int, int] | None = None,
    max_area: float | None = None,
    min_square_ratio: float = 0.45,
) -> dict[str, Any]:
    """
    Traditional CV red-cube detector.

    Steps:
      - optional ROI crop
      - HSV red threshold
      - morphology
      - contour selection by area and shape compactness

    min_square_ratio:
      area / bbox_area lower bound.
      Helps reject long red artifacts.
    """
    if roi is not None:
        x1, y1, x2, y2 = roi
        crop = frame_bgr[y1:y2, x1:x2].copy()
        offset_x, offset_y = x1, y1
    else:
        crop = frame_bgr
        offset_x, offset_y = 0, 0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 70, 40], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([165, 70, 40], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -np.inf

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(max(w * h, 1))
        fill_ratio = area / bbox_area
        aspect = float(w) / float(max(h, 1))
        aspect_score = math.exp(-abs(math.log(max(aspect, 1e-6))))

        if fill_ratio < min_square_ratio:
            continue

        # Prefer compact square-ish red objects, not large red regions.
        score = area * fill_ratio * aspect_score

        if score > best_score:
            best_score = score
            best = (contour, area, x, y, w, h, fill_ratio, aspect)

    if best is None:
        return {
            "found": False,
            "cx": np.nan,
            "cy": np.nan,
            "area": 0.0,
            "bbox_x": np.nan,
            "bbox_y": np.nan,
            "bbox_w": np.nan,
            "bbox_h": np.nan,
            "fill_ratio": np.nan,
            "aspect": np.nan,
        }

    contour, area, x, y, w, h, fill_ratio, aspect = best
    M = cv2.moments(contour)
    if abs(M["m00"]) < 1e-9:
        return {
            "found": False,
            "cx": np.nan,
            "cy": np.nan,
            "area": area,
            "bbox_x": np.nan,
            "bbox_y": np.nan,
            "bbox_w": np.nan,
            "bbox_h": np.nan,
            "fill_ratio": fill_ratio,
            "aspect": aspect,
        }

    cx = float(M["m10"] / M["m00"] + offset_x)
    cy = float(M["m01"] / M["m00"] + offset_y)

    return {
        "found": True,
        "cx": cx,
        "cy": cy,
        "area": area,
        "bbox_x": float(x + offset_x),
        "bbox_y": float(y + offset_y),
        "bbox_w": float(w),
        "bbox_h": float(h),
        "fill_ratio": float(fill_ratio),
        "aspect": float(aspect),
    }


def draw_detection_overlay(frame_bgr: np.ndarray, det: dict[str, Any], label: str, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    out = frame_bgr.copy()

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)

    if det.get("found", False):
        cx, cy = int(round(det["cx"])), int(round(det["cy"]))
        x, y = int(round(det["bbox_x"])), int(round(det["bbox_y"]))
        w, h = int(round(det["bbox_w"])), int(round(det["bbox_h"]))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 6, (255, 0, 0), -1)

    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def detect_episode_position(
    repo_path: Path,
    ep_row: pd.Series,
    camera_key: str,
    frame_indices: list[int],
    fps: float,
    min_area: float,
    max_area: float | None,
    roi: tuple[int, int, int, int] | None,
    overlay_dir: Path,
    info: dict[str, Any],
) -> dict[str, Any]:
    episode_idx = int(ep_row["episode_index"])

    try:
        video_path, from_ts, to_ts = episode_video_path_and_start_time(repo_path, ep_row, camera_key, info)
    except Exception as exc:
        return {
            "episode_idx": episode_idx,
            "video_path": "",
            "found": False,
            "cx": np.nan,
            "cy": np.nan,
            "area": 0.0,
            "fill_ratio": np.nan,
            "aspect": np.nan,
            "num_found_frames": 0,
            "reason": f"video_metadata_or_path_error: {exc}",
        }

    detections = []
    first_overlay_written = False

    for local_fi in frame_indices:
        timestamp_s = from_ts + float(local_fi) / float(fps)
        if timestamp_s > to_ts:
            continue

        frame = read_video_frame_at_time(video_path, timestamp_s)
        if frame is None:
            continue

        det = detect_red_cube_center(
            frame,
            min_area=min_area,
            max_area=max_area,
            roi=roi,
        )
        det["local_frame_index"] = int(local_fi)
        det["timestamp_s"] = float(timestamp_s)
        detections.append(det)

        if not first_overlay_written:
            label = (
                f"ep={episode_idx} frame={local_fi} t={timestamp_s:.3f} "
                f"found={det.get('found')} cx={det.get('cx', np.nan):.1f} cy={det.get('cy', np.nan):.1f}"
            )
            overlay = draw_detection_overlay(frame, det, label, roi=roi)
            overlay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_dir / f"episode_{episode_idx:06d}_detect.png"), overlay)
            first_overlay_written = True

    found = [d for d in detections if d.get("found", False)]
    if not found:
        reason = "red_cube_not_found" if detections else "no_decoded_frames"
        return {
            "episode_idx": episode_idx,
            "video_path": str(video_path),
            "found": False,
            "cx": np.nan,
            "cy": np.nan,
            "area": max([d.get("area", 0.0) for d in detections], default=0.0),
            "fill_ratio": np.nan,
            "aspect": np.nan,
            "num_found_frames": 0,
            "reason": reason,
        }

    return {
        "episode_idx": episode_idx,
        "video_path": str(video_path),
        "found": True,
        "cx": float(np.median([d["cx"] for d in found])),
        "cy": float(np.median([d["cy"] for d in found])),
        "area": float(np.median([d["area"] for d in found])),
        "fill_ratio": float(np.median([d.get("fill_ratio", np.nan) for d in found])),
        "aspect": float(np.median([d.get("aspect", np.nan) for d in found])),
        "num_found_frames": int(len(found)),
        "reason": "",
    }


# ==============================================================================
# Episode metrics
# ==============================================================================

def summarize_episodes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep, g in df.groupby("episode_idx"):
        g = g.sort_values("episode_step")
        z = g["eef_z"].to_numpy(dtype=np.float64)
        peak_i = int(np.nanargmax(z))
        rows.append({
            "episode_idx": int(ep),
            "num_steps": int(len(g)),
            "eef_z_start": float(z[0]),
            "eef_z_min": float(np.nanmin(z)),
            "eef_z_max": float(np.nanmax(z)),
            "eef_z_end": float(z[-1]),
            "eef_z_range": float(np.nanmax(z) - np.nanmin(z)),
            "peak_episode_step": int(g.iloc[peak_i]["episode_step"]),
            "peak_frame_index": int(g.iloc[peak_i]["frame_index"]),
        })
    return pd.DataFrame(rows).sort_values("episode_idx")


def summarize_window(df: pd.DataFrame, summary: pd.DataFrame, offsets: tuple[int, int]) -> pd.DataFrame:
    peak_map = summary.set_index("episode_idx")["peak_episode_step"].to_dict()
    rows = []

    for ep, g in df.groupby("episode_idx"):
        g = g.sort_values("episode_step")
        far, near = offsets
        peak = int(peak_map[int(ep)])
        start = max(0, peak - far)
        end = max(0, peak - near)
        sub = g[(g["episode_step"] >= start) & (g["episode_step"] <= end)].copy()
        if sub.empty:
            continue

        row = {
            "episode_idx": int(ep),
            "window_start_step": int(start),
            "window_end_step": int(end),
            "num_window_steps": int(len(sub)),
            "eef_z_window_mean": float(sub["eef_z"].mean()),
            "eef_z_window_max": float(sub["eef_z"].max()),
        }

        for joint in JOINT_NAMES:
            for kind in ["action", "state"]:
                col = f"{kind}.{joint}"
                row[f"{col}.mean"] = float(sub[col].mean())
                row[f"{col}.std"] = float(sub[col].std(ddof=0))
                row[f"{col}.min"] = float(sub[col].min())
                row[f"{col}.max"] = float(sub[col].max())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("episode_idx")


# ==============================================================================
# Grid analysis
# ==============================================================================

def assign_grid_cells(
    metrics: pd.DataFrame,
    roi: tuple[int, int, int, int],
    grid_cols: int,
    grid_rows: int,
) -> pd.DataFrame:
    out = metrics.copy()
    x1, y1, x2, y2 = roi
    w = x2 - x1
    h = y2 - y1

    gx = np.floor((out["cx"].to_numpy(dtype=np.float64) - x1) / max(w, 1) * grid_cols).astype(int)
    gy = np.floor((out["cy"].to_numpy(dtype=np.float64) - y1) / max(h, 1) * grid_rows).astype(int)
    gx = np.clip(gx, 0, grid_cols - 1)
    gy = np.clip(gy, 0, grid_rows - 1)

    out["grid_x"] = gx
    out["grid_y"] = gy
    out["grid_id"] = [f"cell_x{int(x):02d}_y{int(y):02d}" for x, y in zip(gx, gy)]
    out["grid_center_cx"] = x1 + (gx + 0.5) * (w / grid_cols)
    out["grid_center_cy"] = y1 + (gy + 0.5) * (h / grid_rows)
    return out


def grid_cell_summary(metrics: pd.DataFrame, grid_cols: int, grid_rows: int) -> pd.DataFrame:
    rows = []

    for gy in range(grid_rows):
        for gx in range(grid_cols):
            grid_id = f"cell_x{gx:02d}_y{gy:02d}"
            g = metrics[metrics["grid_id"] == grid_id].copy()

            if g.empty:
                rows.append({
                    "grid_id": grid_id,
                    "grid_x": gx,
                    "grid_y": gy,
                    "num_episodes": 0,
                    "episode_ids": "",
                    "cx_mean": np.nan,
                    "cy_mean": np.nan,
                    "eef_z_max_mean": np.nan,
                    "eef_z_max_std": np.nan,
                    "eef_z_max_min": np.nan,
                    "eef_z_max_max": np.nan,
                    "eef_z_max_range": np.nan,
                    "eef_z_max_iqr": np.nan,
                    "prelift_wrist_mean": np.nan,
                    "prelift_wrist_std": np.nan,
                    "prelift_wrist_iqr": np.nan,
                    "prelift_wrist_range": np.nan,
                    "prelift_elbow_mean": np.nan,
                    "prelift_elbow_std": np.nan,
                    "prelift_elbow_iqr": np.nan,
                    "prelift_elbow_range": np.nan,
                    "prelift_gripper_mean": np.nan,
                    "prelift_gripper_std": np.nan,
                })
                continue

            wrist = g["action.wrist_flex.pos.mean"]
            elbow = g["action.elbow_flex.pos.mean"]

            rows.append({
                "grid_id": grid_id,
                "grid_x": gx,
                "grid_y": gy,
                "num_episodes": int(len(g)),
                "episode_ids": ",".join(str(int(x)) for x in sorted(g["episode_idx"].tolist())),
                "cx_mean": float(g["cx"].mean()),
                "cy_mean": float(g["cy"].mean()),
                "eef_z_max_mean": float(g["eef_z_max"].mean()),
                "eef_z_max_std": float(g["eef_z_max"].std(ddof=0)),
                "eef_z_max_min": float(g["eef_z_max"].min()),
                "eef_z_max_max": float(g["eef_z_max"].max()),
                "eef_z_max_range": float(g["eef_z_max"].max() - g["eef_z_max"].min()),
                "eef_z_max_iqr": iqr_series(g["eef_z_max"]),
                "prelift_wrist_mean": float(wrist.mean()),
                "prelift_wrist_std": float(wrist.std(ddof=0)),
                "prelift_wrist_iqr": iqr_series(wrist),
                "prelift_wrist_range": float(wrist.max() - wrist.min()),
                "prelift_elbow_mean": float(elbow.mean()),
                "prelift_elbow_std": float(elbow.std(ddof=0)),
                "prelift_elbow_iqr": iqr_series(elbow),
                "prelift_elbow_range": float(elbow.max() - elbow.min()),
                "prelift_gripper_mean": float(g["action.gripper.pos.mean"].mean()),
                "prelift_gripper_std": float(g["action.gripper.pos.mean"].std(ddof=0)),
            })

    return pd.DataFrame(rows)


def make_grid_advice(
    grid: pd.DataFrame,
    min_episodes_per_cell: int,
    max_eef_std: float,
    max_eef_range: float,
    max_wrist_iqr: float,
    max_elbow_iqr: float,
) -> pd.DataFrame:
    rows = []

    for _, r in grid.iterrows():
        flags = []
        advice = []
        severity = 0

        n = int(r["num_episodes"])

        if n == 0:
            flags.append("EMPTY_CELL")
            advice.append("record demos in this spatial cell if it is inside the intended workspace")
            severity = max(severity, 3)

        elif n < min_episodes_per_cell:
            flags.append("LOW_COUNT")
            advice.append(f"resume recording: add at least {min_episodes_per_cell - n} demos in this cell")
            severity = max(severity, 2)

        if n >= 2:
            if pd.notna(r["eef_z_max_std"]) and r["eef_z_max_std"] > max_eef_std:
                flags.append("HIGH_EEF_STD")
                advice.append("inspect/re-record episodes with abnormal EEF peak height")
                severity = max(severity, 2)

            if pd.notna(r["eef_z_max_range"]) and r["eef_z_max_range"] > max_eef_range:
                flags.append("HIGH_EEF_RANGE")
                advice.append("same cell contains mixed lift outcomes; curate or re-record inconsistent demos")
                severity = max(severity, 3)

            if pd.notna(r["prelift_wrist_iqr"]) and r["prelift_wrist_iqr"] > max_wrist_iqr:
                flags.append("WRIST_MULTIMODAL")
                advice.append("pre-lift wrist style varies too much; re-record with consistent wrist strategy")
                severity = max(severity, 3)

            if pd.notna(r["prelift_elbow_iqr"]) and r["prelift_elbow_iqr"] > max_elbow_iqr:
                flags.append("ELBOW_MULTIMODAL")
                advice.append("pre-lift elbow style varies too much; re-record with consistent elbow strategy")
                severity = max(severity, 3)

        if not flags:
            status = "OK"
            advice_text = "cell appears healthy"
        elif "EMPTY_CELL" in flags:
            status = "EMPTY_RECORD_IF_NEEDED"
            advice_text = "; ".join(advice)
        elif "LOW_COUNT" in flags and len(flags) == 1:
            status = "RESUME_RECORDING"
            advice_text = "; ".join(advice)
        elif any(f in flags for f in ["WRIST_MULTIMODAL", "ELBOW_MULTIMODAL", "HIGH_EEF_RANGE"]):
            status = "CURATE_OR_RERECORD"
            advice_text = "; ".join(advice)
        else:
            status = "INSPECT"
            advice_text = "; ".join(advice)

        rows.append({
            "grid_id": r["grid_id"],
            "grid_x": int(r["grid_x"]),
            "grid_y": int(r["grid_y"]),
            "status": status,
            "severity": int(severity),
            "flags": ",".join(flags),
            "advice": advice_text,
            "num_episodes": n,
            "episode_ids": r.get("episode_ids", ""),
            "eef_z_max_mean": r["eef_z_max_mean"],
            "eef_z_max_std": r["eef_z_max_std"],
            "eef_z_max_range": r["eef_z_max_range"],
            "prelift_wrist_iqr": r["prelift_wrist_iqr"],
            "prelift_elbow_iqr": r["prelift_elbow_iqr"],
        })

    return pd.DataFrame(rows)


def find_outlier_episodes(
    metrics: pd.DataFrame,
    min_cell_count_for_outliers: int = 4,
    mad_z_threshold: float = 2.5,
) -> pd.DataFrame:
    rows = []
    cols = {
        "eef_z_max": "EEF_PEAK_OUTLIER",
        "action.wrist_flex.pos.mean": "WRIST_OUTLIER",
        "action.elbow_flex.pos.mean": "ELBOW_OUTLIER",
        "peak_episode_step": "PEAK_TIMING_OUTLIER",
    }

    for grid_id, g in metrics.groupby("grid_id"):
        if len(g) < min_cell_count_for_outliers:
            continue

        for col, flag in cols.items():
            vals = g[col].to_numpy(dtype=np.float64)
            z = robust_mad_z(vals)
            for i, (_, r) in enumerate(g.iterrows()):
                if abs(z[i]) >= mad_z_threshold:
                    rows.append({
                        "episode_idx": int(r["episode_idx"]),
                        "grid_id": grid_id,
                        "flag": flag,
                        "metric": col,
                        "value": float(r[col]),
                        "robust_z": float(z[i]),
                        "eef_z_max": float(r["eef_z_max"]),
                        "prelift_wrist": float(r["action.wrist_flex.pos.mean"]),
                        "prelift_elbow": float(r["action.elbow_flex.pos.mean"]),
                        "peak_episode_step": int(r["peak_episode_step"]),
                    })

    if not rows:
        return pd.DataFrame(columns=[
            "episode_idx", "grid_id", "flag", "metric", "value", "robust_z",
            "eef_z_max", "prelift_wrist", "prelift_elbow", "peak_episode_step",
        ])

    return pd.DataFrame(rows).sort_values(["grid_id", "episode_idx", "flag"])


def dataset_health_summary(
    grid: pd.DataFrame,
    advice: pd.DataFrame,
    metrics: pd.DataFrame,
    outliers: pd.DataFrame,
    min_episodes_per_cell: int,
) -> dict[str, Any]:
    nonempty = grid[grid["num_episodes"] > 0]
    total_cells = int(len(grid))
    nonempty_cells = int(len(nonempty))
    low_count_cells = int((grid["num_episodes"] < min_episodes_per_cell).sum())
    empty_cells = int((grid["num_episodes"] == 0).sum())
    curate_cells = int((advice["status"] == "CURATE_OR_RERECORD").sum())
    ok_cells = int((advice["status"] == "OK").sum())

    total_episodes = int(metrics["episode_idx"].nunique())

    # Simple health score, not a scientific metric.
    coverage_score = nonempty_cells / max(total_cells, 1)
    count_score = 1.0 - low_count_cells / max(total_cells, 1)
    consistency_score = 1.0 - curate_cells / max(total_cells, 1)
    outlier_score = 1.0 - min(len(outliers) / max(total_episodes, 1), 1.0)

    health_score = 100.0 * (
        0.30 * coverage_score +
        0.25 * count_score +
        0.35 * consistency_score +
        0.10 * outlier_score
    )

    return {
        "total_episodes": total_episodes,
        "total_grid_cells": total_cells,
        "nonempty_grid_cells": nonempty_cells,
        "empty_grid_cells": empty_cells,
        "low_count_grid_cells": low_count_cells,
        "curate_or_rerecord_grid_cells": curate_cells,
        "ok_grid_cells": ok_cells,
        "num_outlier_episode_flags": int(len(outliers)),
        "coverage_score_0_1": coverage_score,
        "count_balance_score_0_1": count_score,
        "consistency_score_0_1": consistency_score,
        "outlier_score_0_1": outlier_score,
        "dataset_health_score_0_100": health_score,
        "note": (
            "Health score is a heuristic. Use grid_cell_advice.csv and "
            "outlier_episodes.csv for actual recording decisions."
        ),
    }



# ==============================================================================
# Data-driven auto-region analysis
# ==============================================================================

def load_reference_top_frame(
    repo_path: Path,
    ep_meta_by_idx: pd.DataFrame,
    episode_indices: list[int],
    camera_key: str,
    frame_indices: list[int],
    fps: float,
    info: dict[str, Any],
) -> np.ndarray | None:
    """
    Load one representative top-camera frame for RGB-background plotting.
    """
    for ep in episode_indices:
        if ep not in ep_meta_by_idx.index:
            continue

        try:
            ep_row = ep_meta_by_idx.loc[ep]
            video_path, from_ts, to_ts = episode_video_path_and_start_time(
                repo_path, ep_row, camera_key, info
            )
        except Exception:
            continue

        for local_fi in frame_indices:
            timestamp_s = from_ts + float(local_fi) / float(fps)
            if timestamp_s > to_ts:
                continue
            frame = read_video_frame_at_time(video_path, timestamp_s)
            if frame is not None:
                return frame

    return None


def kmeans_numpy(
    points: np.ndarray,
    k: int,
    max_iter: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Small deterministic k-means implementation to avoid extra dependencies.
    Returns labels, centers, inertia.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if k < 1:
        raise ValueError("k must be >= 1")
    if n < k:
        raise ValueError(f"Need at least k={k} points, got n={n}")

    # K-means++ style initialization.
    centers = [points[int(rng.integers(0, n))]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        if float(d2.sum()) <= 1e-12:
            idx = int(rng.integers(0, n))
        else:
            idx = int(rng.choice(n, p=d2 / d2.sum()))
        centers.append(points[idx])

    centers = np.asarray(centers, dtype=np.float64)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)

        new_centers = centers.copy()
        for c in range(k):
            mask = new_labels == c
            if mask.any():
                new_centers[c] = points[mask].mean(axis=0)

        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            break

        labels = new_labels
        centers = new_centers

    dist = ((points - centers[labels]) ** 2).sum(axis=1)
    inertia = float(dist.sum())

    # Stable order by center x then y, so region ids are repeatable.
    order = sorted(range(k), key=lambda i: (centers[i, 0], centers[i, 1]))
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(x)] for x in labels], dtype=int)
    centers = centers[order]

    return labels, centers, inertia


def best_kmeans_numpy(points: np.ndarray, k: int, num_seeds: int = 8) -> tuple[np.ndarray, np.ndarray, float]:
    best = None
    for seed in range(num_seeds):
        labels, centers, inertia = kmeans_numpy(points, k=k, seed=seed)
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    assert best is not None
    return best


def silhouette_score_numpy(points: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean silhouette score without sklearn.
    If k==1 or a cluster has only one sample, singleton samples get score 0.
    """
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    unique = sorted(np.unique(labels).tolist())

    if len(unique) <= 1:
        return 0.0

    # Pairwise Euclidean distances.
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    scores = []

    for i in range(len(points)):
        own = labels[i]
        own_mask = labels == own

        if own_mask.sum() <= 1:
            scores.append(0.0)
            continue

        a = float(d[i, own_mask].sum() / max(int(own_mask.sum()) - 1, 1))

        b_vals = []
        for other in unique:
            if other == own:
                continue
            m = labels == other
            if m.any():
                b_vals.append(float(d[i, m].mean()))

        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        scores.append(0.0 if denom <= 1e-12 else (b - a) / denom)

    return float(np.mean(scores))


def choose_auto_region_count(
    points: np.ndarray,
    fixed_k: int,
    min_k: int,
    max_k: int,
    target_episodes_per_region: int,
    min_episodes_per_region: int,
) -> tuple[int, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Choose a data-driven number of regions.

    Selection is heuristic:
      - Prefer good silhouette.
      - Penalize clusters below min_episodes_per_region.
      - Mildly prefer a region count whose mean count is near target_episodes_per_region.

    This is not a ground-truth estimator of "true positions"; it produces useful
    diagnostic regions for recording/curation decisions.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)

    if n < 2:
        labels = np.zeros(n, dtype=int)
        centers = points.copy()
        candidates = pd.DataFrame([{
            "k": 1,
            "score": 0.0,
            "silhouette": 0.0,
            "min_cluster_count": n,
            "low_count_fraction": 0.0,
            "mean_cluster_count": n,
            "inertia": 0.0,
        }])
        return 1, labels, centers, candidates

    if fixed_k and fixed_k > 0:
        k = int(np.clip(fixed_k, 1, n))
        labels, centers, inertia = best_kmeans_numpy(points, k=k)
        counts = np.bincount(labels, minlength=k)
        candidates = pd.DataFrame([{
            "k": k,
            "score": np.nan,
            "silhouette": silhouette_score_numpy(points, labels),
            "min_cluster_count": int(counts.min()),
            "low_count_fraction": float((counts < min_episodes_per_region).mean()),
            "mean_cluster_count": float(counts.mean()),
            "inertia": inertia,
            "selection": "fixed_by_user",
        }])
        return k, labels, centers, candidates

    min_k = max(2, int(min_k))
    max_k = min(int(max_k), n)
    if max_k < min_k:
        min_k = max_k

    rows = []
    best = None

    for k in range(min_k, max_k + 1):
        labels, centers, inertia = best_kmeans_numpy(points, k=k)
        counts = np.bincount(labels, minlength=k)

        sil = silhouette_score_numpy(points, labels)
        low_frac = float((counts < min_episodes_per_region).mean())
        mean_count = float(counts.mean())
        target_penalty = abs(mean_count - float(target_episodes_per_region)) / max(float(target_episodes_per_region), 1.0)

        # Silhouette is usually [-1,1]. Penalties are deliberately moderate.
        score = sil - 0.35 * low_frac - 0.08 * target_penalty

        row = {
            "k": int(k),
            "score": float(score),
            "silhouette": float(sil),
            "min_cluster_count": int(counts.min()),
            "max_cluster_count": int(counts.max()),
            "low_count_fraction": low_frac,
            "mean_cluster_count": mean_count,
            "target_count_penalty": float(target_penalty),
            "inertia": float(inertia),
        }
        rows.append(row)

        if best is None or score > best[0]:
            best = (score, k, labels, centers)

    candidates = pd.DataFrame(rows).sort_values("score", ascending=False)
    assert best is not None
    _, selected_k, selected_labels, selected_centers = best

    return int(selected_k), selected_labels, selected_centers, candidates


def assign_auto_regions(
    metrics: pd.DataFrame,
    labels: np.ndarray,
    centers: np.ndarray,
) -> pd.DataFrame:
    out = metrics.copy()
    out["auto_region_idx"] = labels.astype(int)
    out["auto_region_id"] = [f"region_{int(x):02d}" for x in labels]
    center_x = {i: float(centers[i, 0]) for i in range(len(centers))}
    center_y = {i: float(centers[i, 1]) for i in range(len(centers))}
    out["auto_region_center_cx"] = out["auto_region_idx"].map(center_x)
    out["auto_region_center_cy"] = out["auto_region_idx"].map(center_y)
    return out


def auto_region_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for region_id, g in metrics.groupby("auto_region_id"):
        wrist = g["action.wrist_flex.pos.mean"]
        elbow = g["action.elbow_flex.pos.mean"]

        rows.append({
            "auto_region_id": region_id,
            "auto_region_idx": int(g["auto_region_idx"].iloc[0]),
            "num_episodes": int(len(g)),
            "episode_ids": ",".join(str(int(x)) for x in sorted(g["episode_idx"].tolist())),
            "center_cx": float(g["auto_region_center_cx"].iloc[0]),
            "center_cy": float(g["auto_region_center_cy"].iloc[0]),
            "cx_mean": float(g["cx"].mean()),
            "cy_mean": float(g["cy"].mean()),
            "cx_std": float(g["cx"].std(ddof=0)),
            "cy_std": float(g["cy"].std(ddof=0)),
            "eef_z_max_mean": float(g["eef_z_max"].mean()),
            "eef_z_max_std": float(g["eef_z_max"].std(ddof=0)),
            "eef_z_max_min": float(g["eef_z_max"].min()),
            "eef_z_max_max": float(g["eef_z_max"].max()),
            "eef_z_max_range": float(g["eef_z_max"].max() - g["eef_z_max"].min()),
            "eef_z_max_iqr": iqr_series(g["eef_z_max"]),
            "prelift_wrist_mean": float(wrist.mean()),
            "prelift_wrist_std": float(wrist.std(ddof=0)),
            "prelift_wrist_iqr": iqr_series(wrist),
            "prelift_wrist_range": float(wrist.max() - wrist.min()),
            "prelift_elbow_mean": float(elbow.mean()),
            "prelift_elbow_std": float(elbow.std(ddof=0)),
            "prelift_elbow_iqr": iqr_series(elbow),
            "prelift_elbow_range": float(elbow.max() - elbow.min()),
            "prelift_gripper_mean": float(g["action.gripper.pos.mean"].mean()),
            "prelift_gripper_std": float(g["action.gripper.pos.mean"].std(ddof=0)),
        })

    return pd.DataFrame(rows).sort_values("auto_region_idx")


def make_auto_region_advice(
    summary: pd.DataFrame,
    min_episodes_per_region: int,
    max_eef_std: float,
    max_eef_range: float,
    max_wrist_iqr: float,
    max_elbow_iqr: float,
) -> pd.DataFrame:
    rows = []

    for _, r in summary.iterrows():
        flags = []
        advice = []
        severity = 0

        n = int(r["num_episodes"])
        needed = max(0, int(min_episodes_per_region) - n)

        if n < min_episodes_per_region:
            flags.append("LOW_COUNT")
            advice.append(f"resume recording: add at least {needed} demos around this data-driven region")
            severity = max(severity, 2)

        if n >= 2:
            if pd.notna(r["eef_z_max_std"]) and r["eef_z_max_std"] > max_eef_std:
                flags.append("HIGH_EEF_STD")
                advice.append("inspect/re-record episodes with abnormal EEF peak height")
                severity = max(severity, 2)

            if pd.notna(r["eef_z_max_range"]) and r["eef_z_max_range"] > max_eef_range:
                flags.append("HIGH_EEF_RANGE")
                advice.append("region contains mixed lift outcomes; curate or re-record inconsistent demos")
                severity = max(severity, 3)

            if pd.notna(r["prelift_wrist_iqr"]) and r["prelift_wrist_iqr"] > max_wrist_iqr:
                flags.append("WRIST_MULTIMODAL")
                advice.append("pre-lift wrist style varies too much; re-record with consistent wrist strategy")
                severity = max(severity, 3)

            if pd.notna(r["prelift_elbow_iqr"]) and r["prelift_elbow_iqr"] > max_elbow_iqr:
                flags.append("ELBOW_MULTIMODAL")
                advice.append("pre-lift elbow style varies too much; re-record with consistent elbow strategy")
                severity = max(severity, 3)

        if not flags:
            status = "OK"
            advice_text = "region appears healthy"
        elif "LOW_COUNT" in flags and len(flags) == 1:
            status = "RESUME_RECORDING"
            advice_text = "; ".join(advice)
        elif any(f in flags for f in ["WRIST_MULTIMODAL", "ELBOW_MULTIMODAL", "HIGH_EEF_RANGE"]):
            status = "CURATE_OR_RERECORD"
            advice_text = "; ".join(advice)
        else:
            status = "INSPECT"
            advice_text = "; ".join(advice)

        rows.append({
            "auto_region_id": r["auto_region_id"],
            "auto_region_idx": int(r["auto_region_idx"]),
            "status": status,
            "severity": int(severity),
            "flags": ",".join(flags),
            "advice": advice_text,
            "needed_demos": int(needed),
            "num_episodes": n,
            "episode_ids": r["episode_ids"],
            "center_cx": r["center_cx"],
            "center_cy": r["center_cy"],
            "eef_z_max_mean": r["eef_z_max_mean"],
            "eef_z_max_std": r["eef_z_max_std"],
            "eef_z_max_range": r["eef_z_max_range"],
            "prelift_wrist_iqr": r["prelift_wrist_iqr"],
            "prelift_elbow_iqr": r["prelift_elbow_iqr"],
        })

    return pd.DataFrame(rows).sort_values(["severity", "auto_region_idx"], ascending=[False, True])


def find_auto_region_outliers(
    metrics: pd.DataFrame,
    min_region_count_for_outliers: int = 4,
    mad_z_threshold: float = 2.5,
) -> pd.DataFrame:
    rows = []
    cols = {
        "eef_z_max": "EEF_PEAK_OUTLIER",
        "action.wrist_flex.pos.mean": "WRIST_OUTLIER",
        "action.elbow_flex.pos.mean": "ELBOW_OUTLIER",
        "peak_episode_step": "PEAK_TIMING_OUTLIER",
    }

    for region_id, g in metrics.groupby("auto_region_id"):
        if len(g) < min_region_count_for_outliers:
            continue

        for col, flag in cols.items():
            vals = g[col].to_numpy(dtype=np.float64)
            z = robust_mad_z(vals)
            for i, (_, r) in enumerate(g.iterrows()):
                if abs(z[i]) >= mad_z_threshold:
                    rows.append({
                        "episode_idx": int(r["episode_idx"]),
                        "auto_region_id": region_id,
                        "flag": flag,
                        "metric": col,
                        "value": float(r[col]),
                        "robust_z": float(z[i]),
                        "eef_z_max": float(r["eef_z_max"]),
                        "prelift_wrist": float(r["action.wrist_flex.pos.mean"]),
                        "prelift_elbow": float(r["action.elbow_flex.pos.mean"]),
                        "peak_episode_step": int(r["peak_episode_step"]),
                    })

    if not rows:
        return pd.DataFrame(columns=[
            "episode_idx", "auto_region_id", "flag", "metric", "value", "robust_z",
            "eef_z_max", "prelift_wrist", "prelift_elbow", "peak_episode_step",
        ])

    return pd.DataFrame(rows).sort_values(["auto_region_id", "episode_idx", "flag"])


def auto_region_health_summary(
    auto_summary: pd.DataFrame,
    auto_advice: pd.DataFrame,
    auto_outliers: pd.DataFrame,
    selected_k: int,
) -> dict[str, Any]:
    total_regions = int(len(auto_summary))
    low_count_regions = int((auto_advice["flags"].fillna("").str.contains("LOW_COUNT")).sum())
    curate_regions = int((auto_advice["status"] == "CURATE_OR_RERECORD").sum())
    ok_regions = int((auto_advice["status"] == "OK").sum())

    count_score = 1.0 - low_count_regions / max(total_regions, 1)
    consistency_score = 1.0 - curate_regions / max(total_regions, 1)
    outlier_score = 1.0 - min(len(auto_outliers) / max(int(auto_summary["num_episodes"].sum()), 1), 1.0)
    health_score = 100.0 * (0.35 * count_score + 0.50 * consistency_score + 0.15 * outlier_score)

    return {
        "selected_auto_regions": int(selected_k),
        "total_auto_regions": total_regions,
        "ok_auto_regions": ok_regions,
        "low_count_auto_regions": low_count_regions,
        "curate_or_rerecord_auto_regions": curate_regions,
        "num_auto_region_outlier_flags": int(len(auto_outliers)),
        "count_score_0_1": float(count_score),
        "consistency_score_0_1": float(consistency_score),
        "outlier_score_0_1": float(outlier_score),
        "auto_region_health_score_0_100": float(health_score),
        "note": (
            "Auto-regions are data-driven diagnostic regions, not ground-truth block positions. "
            "Use them with fixed-grid and RGB overlays."
        ),
    }


# ==============================================================================
# Plots
# ==============================================================================

def matrix_from_grid(grid: pd.DataFrame, value_col: str, grid_cols: int, grid_rows: int) -> np.ndarray:
    mat = np.full((grid_rows, grid_cols), np.nan, dtype=np.float64)
    for _, r in grid.iterrows():
        mat[int(r["grid_y"]), int(r["grid_x"])] = r[value_col]
    return mat


def save_heatmap(
    mat: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    value_format: str = ".2f",
) -> None:
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mat, origin="upper", cmap=cmap)
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("grid_x")
    plt.ylabel("grid_y")

    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            v = mat[y, x]
            text = "nan" if not np.isfinite(v) else format(v, value_format)
            plt.text(x, y, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_cube_positions_grid(
    metrics: pd.DataFrame,
    roi: tuple[int, int, int, int],
    grid_cols: int,
    grid_rows: int,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 7))
    for grid_id, g in metrics.groupby("grid_id"):
        plt.scatter(g["cx"], g["cy"], label=grid_id, s=70)
        for _, r in g.iterrows():
            plt.text(r["cx"], r["cy"], str(int(r["episode_idx"])), fontsize=8)

    x1, y1, x2, y2 = roi
    for i in range(grid_cols + 1):
        x = x1 + (x2 - x1) * i / grid_cols
        plt.axvline(x, linewidth=0.8)
    for j in range(grid_rows + 1):
        y = y1 + (y2 - y1) * j / grid_rows
        plt.axhline(y, linewidth=0.8)

    plt.xlim(x1 - 10, x2 + 10)
    plt.ylim(y2 + 10, y1 - 10)
    plt.xlabel("cube cx")
    plt.ylabel("cube cy")
    plt.title("Cube positions with grid cells")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_prelift_wrist_elbow(metrics: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    for grid_id, g in metrics.groupby("grid_id"):
        plt.scatter(
            g["action.wrist_flex.pos.mean"],
            g["action.elbow_flex.pos.mean"],
            label=grid_id,
            s=70,
        )
        for _, r in g.iterrows():
            plt.text(
                r["action.wrist_flex.pos.mean"],
                r["action.elbow_flex.pos.mean"],
                str(int(r["episode_idx"])),
                fontsize=8,
            )

    plt.xlabel("pre-lift action.wrist_flex.pos.mean")
    plt.ylabel("pre-lift action.elbow_flex.pos.mean")
    plt.title("Pre-lift wrist/elbow style by grid cell")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_grid_recording_gap_overlay_on_rgb(
    background_bgr: np.ndarray | None,
    roi: tuple[int, int, int, int],
    grid: pd.DataFrame,
    advice: pd.DataFrame,
    grid_cols: int,
    grid_rows: int,
    min_episodes_per_cell: int,
    out_path: Path,
) -> None:
    """
    Overlay fixed-grid recording gap on top-camera RGB.
    """
    if background_bgr is None:
        return

    bg_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)

    merged = grid.merge(
        advice[["grid_id", "status", "severity", "flags"]],
        on="grid_id",
        how="left",
    ).copy()

    merged["recording_gap"] = (min_episodes_per_cell - merged["num_episodes"]).clip(lower=0)
    max_gap = int(max(merged["recording_gap"].max(), 1))

    x1, y1, x2, y2 = roi
    cell_w = (x2 - x1) / float(grid_cols)
    cell_h = (y2 - y1) / float(grid_rows)

    cmap = plt.get_cmap("YlOrRd")
    norm = plt.Normalize(vmin=0, vmax=max_gap)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bg_rgb)
    ax.set_title("Fixed-grid recording-gap heatmap over top-camera RGB")

    ax.add_patch(
        Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="cyan", linewidth=2.5,
        )
    )

    for _, r in merged.iterrows():
        gx = int(r["grid_x"])
        gy = int(r["grid_y"])
        px = x1 + gx * cell_w
        py = y1 + gy * cell_h

        n = int(r["num_episodes"])
        gap = int(r["recording_gap"])
        status = str(r.get("status", ""))

        if gap > 0:
            face = cmap(norm(gap))
            alpha = 0.42
        else:
            face = (0.0, 0.0, 0.0, 0.0)
            alpha = 0.10

        if status == "CURATE_OR_RERECORD":
            edgecolor = "red"
            linewidth = 2.2
        elif status == "INSPECT":
            edgecolor = "orange"
            linewidth = 2.0
        else:
            edgecolor = "white"
            linewidth = 1.2

        ax.add_patch(
            Rectangle(
                (px, py), cell_w, cell_h,
                facecolor=face, edgecolor=edgecolor,
                linewidth=linewidth, alpha=alpha,
            )
        )

        text_lines = [f"n={n}"]
        if gap > 0:
            text_lines.append(f"need +{gap}")
        if status == "CURATE_OR_RERECORD":
            text_lines.append("curate")
        elif status == "INSPECT":
            text_lines.append("inspect")

        ax.text(
            px + cell_w / 2.0,
            py + cell_h / 2.0,
            "\n".join(text_lines),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.35, edgecolor="none"),
        )

    for i in range(grid_cols + 1):
        x = x1 + (x2 - x1) * i / grid_cols
        ax.plot([x, x], [y1, y2], color="cyan", linewidth=0.8, alpha=0.8)

    for j in range(grid_rows + 1):
        y = y1 + (y2 - y1) * j / grid_rows
        ax.plot([x1, x2], [y, y], color="cyan", linewidth=0.8, alpha=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("additional demos needed to reach fixed-grid min count")

    ax.set_xlim(0, bg_rgb.shape[1])
    ax.set_ylim(bg_rgb.shape[0], 0)
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_auto_region_recording_gap_overlay_on_rgb(
    background_bgr: np.ndarray | None,
    roi: tuple[int, int, int, int],
    metrics: pd.DataFrame,
    auto_summary: pd.DataFrame,
    auto_advice: pd.DataFrame,
    min_episodes_per_region: int,
    out_path: Path,
) -> None:
    """
    Overlay data-driven k-means/Voronoi-like regions on top-camera RGB.

    Each pixel inside the ROI is assigned to the nearest auto-region center.
    Color intensity indicates how many demos are still needed to reach the
    minimum per-region count. Red borders/text identify regions that need
    curation or re-recording due to inconsistency.
    """
    if background_bgr is None or auto_summary.empty:
        return

    bg_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    h, w = bg_rgb.shape[:2]

    x1, y1, x2, y2 = roi
    centers = auto_summary.sort_values("auto_region_idx")[["center_cx", "center_cy"]].to_numpy(dtype=np.float64)
    region_ids = auto_summary.sort_values("auto_region_idx")["auto_region_id"].tolist()

    advice_by_region = auto_advice.set_index("auto_region_id").to_dict(orient="index")

    # Build a full-image overlay map. Use NaN outside ROI.
    value_map = np.full((h, w), np.nan, dtype=np.float64)
    idx_map = np.full((h, w), -1, dtype=np.int32)

    yy, xx = np.mgrid[y1:y2, x1:x2]
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float64)
    d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1)

    for region_idx, region_id in enumerate(region_ids):
        n = int(auto_summary.loc[auto_summary["auto_region_id"] == region_id, "num_episodes"].iloc[0])
        gap = max(0, int(min_episodes_per_region) - n)
        mask = nearest == region_idx
        flat_values = value_map[y1:y2, x1:x2].reshape(-1)
        flat_idxs = idx_map[y1:y2, x1:x2].reshape(-1)
        flat_values[mask] = float(gap)
        flat_idxs[mask] = int(region_idx)

    max_gap = max(int(np.nanmax(value_map)) if np.isfinite(value_map).any() else 1, 1)
    cmap = plt.get_cmap("YlOrRd")
    norm = plt.Normalize(vmin=0, vmax=max_gap)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bg_rgb)
    ax.set_title("Data-driven region recording-gap overlay")

    overlay = np.ma.masked_invalid(value_map)
    alpha = np.where(np.isfinite(value_map), 0.40, 0.0)
    ax.imshow(overlay, cmap=cmap, norm=norm, alpha=alpha)

    # ROI border.
    ax.add_patch(
        Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="cyan", linewidth=2.5,
        )
    )

    # Draw detected cube points.
    for _, r in metrics.iterrows():
        ax.scatter(r["cx"], r["cy"], s=24, c="deepskyblue", edgecolors="black", linewidths=0.4)
        ax.text(r["cx"] + 2, r["cy"] + 2, str(int(r["episode_idx"])), fontsize=6, color="black")

    # Draw centers and annotation.
    for _, r in auto_summary.iterrows():
        region_id = r["auto_region_id"]
        adv = advice_by_region.get(region_id, {})
        n = int(r["num_episodes"])
        gap = max(0, int(min_episodes_per_region) - n)
        status = str(adv.get("status", ""))
        cx = float(r["center_cx"])
        cy = float(r["center_cy"])

        edgecolor = "red" if status == "CURATE_OR_RERECORD" else "white"
        ax.scatter(cx, cy, s=180, marker="X", c="black", edgecolors=edgecolor, linewidths=2.0)

        text_lines = [region_id, f"n={n}"]
        if gap > 0:
            text_lines.append(f"need +{gap}")
        if status == "CURATE_OR_RERECORD":
            text_lines.append("curate")
        elif status == "INSPECT":
            text_lines.append("inspect")

        ax.text(
            cx,
            cy,
            "\n".join(text_lines),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.45, edgecolor="none"),
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("additional demos needed to reach auto-region min count")

    ax.set_xlim(0, bg_rgb.shape[1])
    ax.set_ylim(bg_rgb.shape[0], 0)
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_auto_region_positions(
    metrics: pd.DataFrame,
    auto_summary: pd.DataFrame,
    roi: tuple[int, int, int, int],
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 7))

    for region_id, g in metrics.groupby("auto_region_id"):
        plt.scatter(g["cx"], g["cy"], label=region_id, s=70)
        for _, r in g.iterrows():
            plt.text(r["cx"], r["cy"], str(int(r["episode_idx"])), fontsize=8)

    plt.scatter(
        auto_summary["center_cx"],
        auto_summary["center_cy"],
        marker="X",
        s=180,
        linewidths=2,
        label="auto-region centers",
    )

    x1, y1, x2, y2 = roi
    plt.axvline(x1, linewidth=1.0)
    plt.axvline(x2, linewidth=1.0)
    plt.axhline(y1, linewidth=1.0)
    plt.axhline(y2, linewidth=1.0)

    plt.xlim(x1 - 10, x2 + 10)
    plt.ylim(y2 + 10, y1 - 10)
    plt.xlabel("cube cx")
    plt.ylabel("cube cy")
    plt.title("Data-driven auto-regions from cube positions")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_plots(
    plot_dir: Path,
    metrics: pd.DataFrame,
    grid: pd.DataFrame,
    advice: pd.DataFrame,
    roi: tuple[int, int, int, int],
    grid_cols: int,
    grid_rows: int,
    background_bgr: np.ndarray | None,
    min_episodes_per_cell: int,
    auto_region_summary_df: pd.DataFrame,
    auto_region_advice_df: pd.DataFrame,
    min_episodes_per_auto_region: int,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_cube_positions_grid(
        metrics,
        roi=roi,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        out_path=plot_dir / "cube_positions_grid.png",
    )

    plot_prelift_wrist_elbow(metrics, plot_dir / "prelift_wrist_elbow_by_grid.png")

    save_heatmap(
        matrix_from_grid(grid, "num_episodes", grid_cols, grid_rows),
        "Number of demos per fixed grid cell",
        plot_dir / "grid_count_heatmap.png",
        value_format=".0f",
    )
    save_heatmap(
        matrix_from_grid(grid, "eef_z_max_range", grid_cols, grid_rows),
        "EEF peak height range per fixed grid cell",
        plot_dir / "grid_eef_range_heatmap.png",
        value_format=".3f",
    )
    save_heatmap(
        matrix_from_grid(grid, "prelift_wrist_iqr", grid_cols, grid_rows),
        "Pre-lift wrist_flex IQR per fixed grid cell",
        plot_dir / "grid_wrist_iqr_heatmap.png",
        value_format=".1f",
    )
    save_heatmap(
        matrix_from_grid(grid, "prelift_elbow_iqr", grid_cols, grid_rows),
        "Pre-lift elbow_flex IQR per fixed grid cell",
        plot_dir / "grid_elbow_iqr_heatmap.png",
        value_format=".1f",
    )
    save_heatmap(
        matrix_from_grid(advice, "severity", grid_cols, grid_rows),
        "Advice severity per fixed grid cell",
        plot_dir / "grid_advice_severity_heatmap.png",
        value_format=".0f",
    )

    plot_grid_recording_gap_overlay_on_rgb(
        background_bgr=background_bgr,
        roi=roi,
        grid=grid,
        advice=advice,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        min_episodes_per_cell=min_episodes_per_cell,
        out_path=plot_dir / "grid_recording_gap_overlay.png",
    )

    if not auto_region_summary_df.empty:
        plot_auto_region_positions(
            metrics=metrics,
            auto_summary=auto_region_summary_df,
            roi=roi,
            out_path=plot_dir / "auto_region_positions.png",
        )

        plot_auto_region_recording_gap_overlay_on_rgb(
            background_bgr=background_bgr,
            roi=roi,
            metrics=metrics,
            auto_summary=auto_region_summary_df,
            auto_advice=auto_region_advice_df,
            min_episodes_per_region=min_episodes_per_auto_region,
            out_path=plot_dir / "auto_region_recording_gap_overlay.png",
        )


# ==============================================================================
# Report
# ==============================================================================

def write_report(
    out_dir: Path,
    dataset_name: str,
    camera_key: str,
    roi: tuple[int, int, int, int],
    grid: pd.DataFrame,
    advice: pd.DataFrame,
    outliers: pd.DataFrame,
    health: dict[str, Any],
    auto_region_selection: dict[str, Any],
    auto_region_candidates: pd.DataFrame,
    auto_region_summary_df: pd.DataFrame,
    auto_region_advice_df: pd.DataFrame,
    auto_region_outliers: pd.DataFrame,
    auto_region_health: dict[str, Any],
) -> None:
    lines = []
    lines.append("# Random-placement dataset robustness diagnosis")
    lines.append("")
    lines.append(f"Dataset: `{dataset_name}`")
    lines.append(f"Camera key: `{camera_key}`")
    lines.append(f"ROI: `{roi}`")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("The script uses traditional computer vision, not a VLM:")
    lines.append("")
    lines.append("```")
    lines.append("top-camera frame -> ROI crop -> HSV red threshold -> contour -> cube center")
    lines.append("```")
    lines.append("")
    lines.append("It then performs two analyses:")
    lines.append("")
    lines.append("```")
    lines.append("1. Fixed-grid analysis: human-readable coverage map.")
    lines.append("2. Data-driven auto-region analysis: regions inferred from the dataset cube positions.")
    lines.append("```")
    lines.append("")
    lines.append("## Fixed-grid health summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(health, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Data-driven auto-region selection")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(auto_region_selection, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Candidate region-count scores:")
    lines.append("")
    lines.append("```")
    if auto_region_candidates.empty:
        lines.append("(none)")
    else:
        lines.append(auto_region_candidates.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Data-driven auto-region health summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(auto_region_health, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Data-driven auto-region advice")
    lines.append("")
    auto_cols = [
        "auto_region_id", "status", "flags", "needed_demos", "num_episodes",
        "eef_z_max_range", "prelift_wrist_iqr", "prelift_elbow_iqr",
        "episode_ids", "advice",
    ]
    lines.append("```")
    if auto_region_advice_df.empty:
        lines.append("(none)")
    else:
        lines.append(auto_region_advice_df[auto_cols].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Fixed-grid advice")
    lines.append("")
    advice_cols = [
        "grid_id", "grid_x", "grid_y", "status", "flags", "num_episodes",
        "eef_z_max_range", "prelift_wrist_iqr", "prelift_elbow_iqr", "episode_ids", "advice",
    ]
    lines.append("```")
    lines.append(advice[advice_cols].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Data-driven auto-region metrics")
    lines.append("")
    auto_metric_cols = [
        "auto_region_id", "num_episodes", "episode_ids",
        "center_cx", "center_cy",
        "eef_z_max_mean", "eef_z_max_std", "eef_z_max_range",
        "prelift_wrist_mean", "prelift_wrist_iqr",
        "prelift_elbow_mean", "prelift_elbow_iqr",
    ]
    lines.append("```")
    if auto_region_summary_df.empty:
        lines.append("(none)")
    else:
        lines.append(auto_region_summary_df[auto_metric_cols].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Fixed-grid metrics")
    lines.append("")
    grid_cols = [
        "grid_id", "num_episodes", "episode_ids",
        "eef_z_max_mean", "eef_z_max_std", "eef_z_max_range",
        "prelift_wrist_mean", "prelift_wrist_iqr",
        "prelift_elbow_mean", "prelift_elbow_iqr",
    ]
    lines.append("```")
    lines.append(grid[grid_cols].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Auto-region outlier episodes")
    lines.append("")
    lines.append("```")
    if auto_region_outliers.empty:
        lines.append("(none)")
    else:
        lines.append(auto_region_outliers.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Fixed-grid outlier episodes")
    lines.append("")
    lines.append("```")
    if outliers.empty:
        lines.append("(none)")
    else:
        lines.append(outliers.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## How to act on this report")
    lines.append("")
    lines.append("- Use `auto_region_recording_gap_overlay.png` to see dataset-driven regions directly on the RGB image.")
    lines.append("- Use `grid_recording_gap_overlay.png` for a fixed, human-readable workspace map.")
    lines.append("- `LOW_COUNT` means resume recording in that region.")
    lines.append("- `HIGH_EEF_RANGE`, `WRIST_MULTIMODAL`, or `ELBOW_MULTIMODAL` means curate/re-record; adding random demos alone may worsen multimodality.")
    lines.append("- Data-driven auto-regions are diagnostic partitions, not physical ground-truth block positions.")
    lines.append("")

    (out_dir / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--local-dataset", default=None)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)

    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--roi", required=True, help="Workspace ROI: x1,y1,x2,y2")
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--grid-rows", type=int, default=4)

    parser.add_argument("--auto-region-k", type=int, default=0, help="If >0, force this many data-driven regions.")
    parser.add_argument("--auto-region-min-k", type=int, default=3)
    parser.add_argument("--auto-region-max-k", type=int, default=12)
    parser.add_argument("--auto-region-target-episodes", type=int, default=8)
    parser.add_argument("--min-episodes-per-auto-region", type=int, default=6)

    parser.add_argument("--detect-frame-indices", default="0,3,6,9")
    parser.add_argument("--red-min-area", type=float, default=80.0)
    parser.add_argument("--red-max-area", type=float, default=None)

    parser.add_argument("--pre-peak-window", default="45:35")
    parser.add_argument("--min-episodes-per-cell", type=int, default=5)
    parser.add_argument("--max-eef-std", type=float, default=0.015)
    parser.add_argument("--max-eef-range", type=float, default=0.040)
    parser.add_argument("--max-wrist-iqr", type=float, default=15.0)
    parser.add_argument("--max-elbow-iqr", type=float, default=15.0)
    parser.add_argument("--outlier-mad-z", type=float, default=2.5)
    parser.add_argument("--min-cell-count-for-outliers", type=int, default=4)

    args = parser.parse_args()

    out_dir = Path(args.out)
    overlay_dir = out_dir / "overlays"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    roi = parse_roi(args.roi)
    assert roi is not None

    dataset_label = args.repo_id if args.repo_id is not None else args.local_dataset

    print(f"[*] Loading dataset: {dataset_label}")
    repo_path = download_or_use_repo(args.repo_id, args.local_dataset, args.cache_dir, include_videos=True)
    print(f"[*] Local dataset path: {repo_path}")

    info = load_dataset_info(repo_path)
    fps = float(info.get("fps", 30))
    print(f"[*] Dataset fps: {fps}")
    print(f"[*] Video path template: {info.get('video_path')}")

    raw = load_lerobot_parquets(repo_path)
    print(f"[*] Raw rows: {len(raw)}")
    print(f"[*] Columns: {list(raw.columns)}")

    ep_meta = load_episode_metadata(repo_path)
    print(f"[*] Loaded episode metadata rows: {len(ep_meta)}")

    model, data = load_fk_model(Path(args.xml))
    frame_df = normalize_dataset_df(raw, model, data)
    frame_df.to_csv(out_dir / "frame_long.csv", index=False)

    ep_summary = summarize_episodes(frame_df)
    ep_meta_by_idx = ep_meta.set_index("episode_index", drop=False)

    frame_indices = parse_int_list(args.detect_frame_indices)

    det_rows = []
    print("[*] Detecting red cube position from chunked videos using ffmpeg...")
    for ep in ep_summary["episode_idx"].astype(int).tolist():
        if ep not in ep_meta_by_idx.index:
            det_rows.append({
                "episode_idx": int(ep),
                "video_path": "",
                "found": False,
                "cx": np.nan,
                "cy": np.nan,
                "area": 0.0,
                "fill_ratio": np.nan,
                "aspect": np.nan,
                "num_found_frames": 0,
                "reason": "episode_missing_from_metadata",
            })
            continue

        row = detect_episode_position(
            repo_path=repo_path,
            ep_row=ep_meta_by_idx.loc[ep],
            camera_key=args.camera_key,
            frame_indices=frame_indices,
            fps=fps,
            min_area=args.red_min_area,
            max_area=args.red_max_area,
            roi=roi,
            overlay_dir=overlay_dir,
            info=info,
        )
        det_rows.append(row)

    det = pd.DataFrame(det_rows).sort_values("episode_idx")
    det.to_csv(out_dir / "cube_detection.csv", index=False)

    found = det[det["found"]].copy()
    if found.empty:
        raise RuntimeError(
            "No valid cube detections. Check ROI, red threshold, ffmpeg AV1 support, and overlays."
        )

    reference_top_frame = load_reference_top_frame(
        repo_path=repo_path,
        ep_meta_by_idx=ep_meta_by_idx,
        episode_indices=found["episode_idx"].astype(int).tolist(),
        camera_key=args.camera_key,
        frame_indices=frame_indices,
        fps=fps,
        info=info,
    )

    win = summarize_window(frame_df, ep_summary, parse_pre_peak_window(args.pre_peak_window))

    metrics = (
        win.merge(ep_summary, on="episode_idx", how="left", suffixes=("", "_episode"))
           .merge(det, on="episode_idx", how="left")
    )

    for c in list(metrics.columns):
        if c.endswith("_episode") and c.replace("_episode", "") in metrics.columns:
            metrics.drop(columns=[c], inplace=True)

    metrics = metrics[metrics["found"] == True].copy()
    metrics = assign_grid_cells(metrics, roi=roi, grid_cols=args.grid_cols, grid_rows=args.grid_rows)

    points = metrics[["cx", "cy"]].to_numpy(dtype=np.float64)
    selected_k, auto_labels, auto_centers, auto_region_candidates = choose_auto_region_count(
        points=points,
        fixed_k=args.auto_region_k,
        min_k=args.auto_region_min_k,
        max_k=args.auto_region_max_k,
        target_episodes_per_region=args.auto_region_target_episodes,
        min_episodes_per_region=args.min_episodes_per_auto_region,
    )
    metrics = assign_auto_regions(metrics, labels=auto_labels, centers=auto_centers)

    metrics.to_csv(out_dir / "episode_metrics.csv", index=False)

    grid = grid_cell_summary(metrics, grid_cols=args.grid_cols, grid_rows=args.grid_rows)
    grid.to_csv(out_dir / "grid_cell_metrics.csv", index=False)

    advice = make_grid_advice(
        grid,
        min_episodes_per_cell=args.min_episodes_per_cell,
        max_eef_std=args.max_eef_std,
        max_eef_range=args.max_eef_range,
        max_wrist_iqr=args.max_wrist_iqr,
        max_elbow_iqr=args.max_elbow_iqr,
    )
    advice.to_csv(out_dir / "grid_cell_advice.csv", index=False)

    auto_summary = auto_region_summary(metrics)
    auto_summary.to_csv(out_dir / "auto_region_metrics.csv", index=False)

    auto_advice = make_auto_region_advice(
        auto_summary,
        min_episodes_per_region=args.min_episodes_per_auto_region,
        max_eef_std=args.max_eef_std,
        max_eef_range=args.max_eef_range,
        max_wrist_iqr=args.max_wrist_iqr,
        max_elbow_iqr=args.max_elbow_iqr,
    )
    auto_advice.to_csv(out_dir / "auto_region_advice.csv", index=False)

    auto_region_candidates.to_csv(out_dir / "auto_region_k_candidates.csv", index=False)

    auto_selection = {
        "selected_k": int(selected_k),
        "fixed_k_requested": int(args.auto_region_k),
        "min_k": int(args.auto_region_min_k),
        "max_k": int(args.auto_region_max_k),
        "target_episodes_per_region": int(args.auto_region_target_episodes),
        "min_episodes_per_auto_region": int(args.min_episodes_per_auto_region),
        "selection_note": (
            "selected_k is chosen by silhouette score with penalties for low-count regions "
            "unless --auto-region-k is set."
        ),
    }
    (out_dir / "auto_region_selection.json").write_text(json.dumps(auto_selection, indent=2), encoding="utf-8")

    outliers = find_outlier_episodes(
        metrics,
        min_cell_count_for_outliers=args.min_cell_count_for_outliers,
        mad_z_threshold=args.outlier_mad_z,
    )
    outliers.to_csv(out_dir / "outlier_episodes.csv", index=False)

    auto_outliers = find_auto_region_outliers(
        metrics,
        min_region_count_for_outliers=args.min_cell_count_for_outliers,
        mad_z_threshold=args.outlier_mad_z,
    )
    auto_outliers.to_csv(out_dir / "auto_region_outlier_episodes.csv", index=False)

    health = dataset_health_summary(
        grid=grid,
        advice=advice,
        metrics=metrics,
        outliers=outliers,
        min_episodes_per_cell=args.min_episodes_per_cell,
    )
    (out_dir / "dataset_health_summary.json").write_text(json.dumps(health, indent=2), encoding="utf-8")

    auto_health = auto_region_health_summary(
        auto_summary=auto_summary,
        auto_advice=auto_advice,
        auto_outliers=auto_outliers,
        selected_k=selected_k,
    )
    (out_dir / "auto_region_health_summary.json").write_text(json.dumps(auto_health, indent=2), encoding="utf-8")

    make_plots(
        plot_dir=plot_dir,
        metrics=metrics,
        grid=grid,
        advice=advice,
        roi=roi,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        background_bgr=reference_top_frame,
        min_episodes_per_cell=args.min_episodes_per_cell,
        auto_region_summary_df=auto_summary,
        auto_region_advice_df=auto_advice,
        min_episodes_per_auto_region=args.min_episodes_per_auto_region,
    )

    write_report(
        out_dir=out_dir,
        dataset_name=str(dataset_label),
        camera_key=args.camera_key,
        roi=roi,
        grid=grid,
        advice=advice,
        outliers=outliers,
        health=health,
        auto_region_selection=auto_selection,
        auto_region_candidates=auto_region_candidates,
        auto_region_summary_df=auto_summary,
        auto_region_advice_df=auto_advice,
        auto_region_outliers=auto_outliers,
        auto_region_health=auto_health,
    )

    print("[DONE] Random-placement grid robustness analysis complete.")
    print(f"[DONE] Output directory: {out_dir.resolve()}")
    print()
    print("Dataset health:")
    print(json.dumps(health, indent=2))
    print()
    print("Auto-region health:")
    print(json.dumps(auto_health, indent=2))
    print()
    print("Auto-region selection:")
    print(json.dumps(auto_selection, indent=2))
    print()
    print("Auto-region advice:")
    auto_show_cols = [
        "auto_region_id", "status", "flags", "needed_demos", "num_episodes",
        "eef_z_max_range", "prelift_wrist_iqr", "prelift_elbow_iqr", "episode_ids",
    ]
    print(auto_advice[auto_show_cols].to_string(index=False))
    print()
    print("Grid-cell advice:")
    show_cols = [
        "grid_id", "status", "flags", "num_episodes",
        "eef_z_max_range", "prelift_wrist_iqr", "prelift_elbow_iqr", "episode_ids",
    ]
    print(advice[show_cols].to_string(index=False))
    print()
    print("Outlier episodes:")
    if outliers.empty:
        print("(none)")
    else:
        print(outliers.to_string(index=False))
    print()
    print("Key outputs:")
    print(f"  - {out_dir / 'diagnosis.md'}")
    print(f"  - {out_dir / 'dataset_health_summary.json'}")
    print(f"  - {out_dir / 'grid_cell_advice.csv'}")
    print(f"  - {out_dir / 'grid_cell_metrics.csv'}")
    print(f"  - {out_dir / 'auto_region_advice.csv'}")
    print(f"  - {out_dir / 'auto_region_metrics.csv'}")
    print(f"  - {out_dir / 'auto_region_k_candidates.csv'}")
    print(f"  - {out_dir / 'auto_region_selection.json'}")
    print(f"  - {out_dir / 'auto_region_health_summary.json'}")
    print(f"  - {out_dir / 'outlier_episodes.csv'}")
    print(f"  - {out_dir / 'auto_region_outlier_episodes.csv'}")
    print(f"  - {plot_dir / 'grid_count_heatmap.png'}")
    print(f"  - {plot_dir / 'grid_advice_severity_heatmap.png'}")
    print(f"  - {plot_dir / 'grid_recording_gap_overlay.png'}")
    print(f"  - {plot_dir / 'auto_region_positions.png'}")
    print(f"  - {plot_dir / 'auto_region_recording_gap_overlay.png'}")
    print(f"  - {overlay_dir}/*.png")


if __name__ == "__main__":
    main()
