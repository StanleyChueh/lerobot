"""
Dataset-conditioned offline steering evaluation for SmolVLA Koch arm.

Visual context (camera images) replays from a pre-recorded LeRobot dataset episode.
Joint state evolves autonomously via MuJoCo FK driven by the policy's predicted actions.

This avoids the sim-to-real image gap: the policy always sees real camera images,
but joint execution is simulated. EEF height is computed at every step via exact FK.

Evaluation conditions:
  none     -- no steering (baseline)
  caa      -- all-layer dense CAA steering (from caa_neurons_finding.py)
  physical -- per-neuron activation steering (from physical_neuron_finding_eef_improved.py)

Usage:
  # Baseline
  python src/lerobot/scripts/sim_eval_steering.py \\
    --policy-path ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \\
    --source-dataset ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \\
    --steering none \\
    --n-rollouts 30 \\
    --task "Put the red cube in the box." \\
    --out-dir sim_eval_baseline

  # CAA high steering
  python src/lerobot/scripts/sim_eval_steering.py \\
    --policy-path ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \\
    --source-dataset ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \\
    --steering caa \\
    --caa-path /home/bruce/CSL/lerobot_nn/caa_dense_all_layers_lm_expert_high_low_60eps_f3.pt \\
    --alpha 2.0 \\
    --n-rollouts 30 \\
    --task "Put the red cube in the box." \\
    --out-dir sim_eval_caa_alpha2

  # Physical neuron high steering
  python src/lerobot/scripts/sim_eval_steering.py \\
    --policy-path ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \\
    --source-dataset ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \\
    --steering physical \\
    --neurons-json /path/to/neurons.json \\
    --steer-direction high \\
    --alpha 4.0 \\
    --n-rollouts 30 \\
    --task "Put the red cube in the box." \\
    --out-dir sim_eval_physical_alpha4

Neurons JSON format (same as hardcoded sets in the recording script):
  {
    "high": {"0": [1293], "1": [1050], "3": [2259], ...},
    "low":  {"3": [962],  "4": [1627], ...}
  }

Output files (in --out-dir):
  rollout_metrics.csv   -- per-rollout EEF height stats (loadable by compare_steering_motion_eef.py)
  eef_height_traces.npz -- raw EEF height trace per rollout (for trajectory plots)
  summary.json          -- aggregate statistics with bootstrap CI and Cohen's d vs baseline
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


# ==============================================================================
# MuJoCo FK calibration
# Matches compare_steering_motion_eef.py and lerobot_replay_in_mujoco.py exactly.
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
    91.7, 15.0, 40.0, 65.0, 0.0, -30.0,
], dtype=np.float64)

CALIB_ORDER = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
CALIB_SCALE = np.asarray([1.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
CALIB_SIGN  = np.asarray([1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float64)

GRIPPER_A = -1.424134513274
GRIPPER_B =  8.690973451327

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def state_deg_to_mujoco_qrad(state_deg: np.ndarray) -> np.ndarray:
    raw_deg = np.asarray(state_deg, dtype=np.float64).reshape(-1)[:6]
    raw_delta = raw_deg[CALIB_ORDER] - CALIB_REST_STATE_DEG[CALIB_ORDER]
    q_deg = CALIB_TARGET_REST_DEG + CALIB_SIGN * CALIB_SCALE * raw_delta
    q_deg[5] = GRIPPER_A * raw_deg[5] + GRIPPER_B
    q_deg[5] = np.clip(q_deg[5], -140.4, 1.8)
    return np.deg2rad(q_deg)


def set_mujoco_qpos(mj_model, mj_data, state_deg: np.ndarray):
    q_rad = state_deg_to_mujoco_qrad(state_deg)
    mujoco.mj_resetData(mj_model, mj_data)
    for i, jname in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        mj_data.qpos[mj_model.jnt_qposadr[jid]] = float(q_rad[i])
    mujoco.mj_forward(mj_model, mj_data)


def get_eef_z(mj_model, mj_data) -> float:
    sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    return float(mj_data.site_xpos[sid][2])


def load_mujoco(xml_path: str):
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


# ==============================================================================
# Steering setup / teardown
# ==============================================================================

def _lm_expert_layers(policy):
    return policy.model.vlm_with_expert.lm_expert.layers


def clear_all_steering(policy):
    """Remove every hook added by this script."""
    for attr in ("_sim_caa_handles", "_sim_phys_handles"):
        if hasattr(policy, attr):
            for h in getattr(policy, attr):
                try:
                    h.remove()
                except Exception:
                    pass
            setattr(policy, attr, [])

    if hasattr(policy, "clear_activation_steering"):
        try:
            policy.clear_activation_steering()
        except Exception:
            pass


def setup_caa_steering(policy, caa_path: str, alpha: float, layer_allowlist=None):
    """Register dense CAA steering hooks — same mechanism as the recording script."""
    clear_all_steering(policy)
    obj = torch.load(caa_path, map_location="cpu", weights_only=False)
    if "vectors" not in obj:
        raise KeyError(f"{caa_path} missing 'vectors' key.")

    layers = _lm_expert_layers(policy)
    handles = []

    for layer_key, vec in obj["vectors"].items():
        layer_idx = int(layer_key)
        if layer_allowlist is not None and layer_idx not in layer_allowlist:
            continue
        if not torch.is_tensor(vec):
            vec = torch.as_tensor(vec)
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        v_cpu = vec.float().cpu()

        if not (0 <= layer_idx < len(layers)):
            continue
        if not hasattr(layers[layer_idx], "mlp"):
            continue

        def _make_hook(v_base, _alpha):
            def hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    h = outputs[0]
                    return (h + _alpha * v_base.to(h.device, h.dtype),) + outputs[1:]
                return outputs + _alpha * v_base.to(outputs.device, outputs.dtype)
            return hook

        handles.append(
            layers[layer_idx].mlp.register_forward_hook(_make_hook(v_cpu, alpha))
        )

    policy._sim_caa_handles = handles
    print(f"[CAA] Registered {len(handles)} hooks, alpha={alpha}")


def setup_physical_neuron_steering(policy, neurons: dict, alpha: float):
    """Register per-neuron activation steering hooks via policy API."""
    clear_all_steering(policy)
    if not hasattr(policy, "set_activation_steering"):
        raise AttributeError(
            "Policy does not have set_activation_steering(). "
            "Ensure modeling_smolvla.py contains the hook-based steering code."
        )
    policy.set_activation_steering(
        steering_neurons=neurons,
        alpha=alpha,
        record_debug=False,
        enable_steering=True,
    )
    policy._sim_phys_handles = []
    print(f"[PHYS] Registered physical neuron steering, alpha={alpha}, neurons={neurons}")


# ==============================================================================
# Dataset utilities
# ==============================================================================

def _to_int(x):
    return int(x.item()) if hasattr(x, "item") else int(x)


def get_episode_frame_ranges(dataset) -> list:
    """Return list of (start_frame, end_frame) tuples per episode."""
    ep_index = None
    for obj in [dataset, dataset.meta]:
        if hasattr(obj, "episode_data_index"):
            ep_index = getattr(obj, "episode_data_index")
            break
    if ep_index is None and hasattr(dataset.meta, "episodes"):
        ep_index = dataset.meta.episodes
    if ep_index is None:
        raise AttributeError("Cannot determine episode frame ranges from dataset.")

    ranges = []
    for ep_idx in range(dataset.num_episodes):
        if hasattr(ep_index, "column_names"):
            cols = set(ep_index.column_names)
            row = ep_index[ep_idx]
            if "dataset_from_index" in cols and "dataset_to_index" in cols:
                ranges.append((_to_int(row["dataset_from_index"]), _to_int(row["dataset_to_index"])))
                continue
            if "from" in cols and "to" in cols:
                ranges.append((_to_int(row["from"]), _to_int(row["to"])))
                continue
            if "length" in cols:
                start = sum(_to_int(ep_index[i]["length"]) for i in range(ep_idx))
                length = _to_int(ep_index[ep_idx]["length"])
                ranges.append((start, start + length))
                continue
        elif isinstance(ep_index, list):
            ep = ep_index[ep_idx]
            for (fk, tk) in [("from", "to"), ("start", "end"), ("dataset_from_index", "dataset_to_index")]:
                if fk in ep and tk in ep:
                    ranges.append((_to_int(ep[fk]), _to_int(ep[tk])))
                    break
        else:
            raise TypeError(f"Unrecognized episode index type: {type(ep_index)}")
    return ranges


def tensor_to_hwc_uint8(value) -> np.ndarray:
    if hasattr(value, "detach"):
        img = value.detach().cpu().numpy()
    else:
        img = np.asarray(value)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def build_obs_frame(dataset_frame: dict, joint_state_deg: np.ndarray, rename_map: dict) -> dict:
    """Build predict_action-compatible observation from dataset frame + current sim joint state."""
    obs = {}
    for k, v in dataset_frame.items():
        if not k.startswith("observation.images."):
            continue
        target_key = rename_map.get(k, k)
        obs[target_key] = tensor_to_hwc_uint8(v)
    obs["observation.state"] = joint_state_deg[:6].astype(np.float32)
    return obs


# ==============================================================================
# Sim rollout
# ==============================================================================

def run_sim_rollout(
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    episode_frames: list,        # list of raw dataset frame dicts for one source episode
    initial_state_deg: np.ndarray,
    mj_model,
    mj_data,
    task: str,
    rename_map: dict,
    steps: int | None = None,
    fps: float = 30.0,
) -> dict:
    """
    Run one sim rollout.

    At each step:
      1. Camera images come from the corresponding dataset frame (real images).
      2. joint state in observation = current MuJoCo-tracked state.
      3. Policy predicts action → treated as next joint target (degrees).
      4. MuJoCo is set to those targets → EEF height computed via FK.
      5. Joint state advances to predicted targets.

    This is closed-loop on joints, open-loop on vision.
    """
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    set_mujoco_qpos(mj_model, mj_data, initial_state_deg)
    current_state_deg = initial_state_deg[:6].astype(np.float64).copy()

    n_steps = len(episode_frames) if steps is None else steps
    n_steps = min(n_steps, len(episode_frames))

    eef_heights = []
    joint_states = []      # state BEFORE action (what policy observed)
    applied_states = []    # state AFTER action (what MuJoCo was set to → for video)
    timestamps = []

    for step_i in range(n_steps):
        frame = episode_frames[step_i]
        obs_frame = build_obs_frame(frame, current_state_deg, rename_map)

        action_values = predict_action(
            observation=obs_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=getattr(policy.config, "use_amp", False),
            task=task,
        )

        # Extract action as 6-dim numpy array of joint degrees
        if torch.is_tensor(action_values):
            act_np = action_values.detach().cpu().numpy().reshape(-1)
        elif isinstance(action_values, dict):
            act_np = np.array(list(action_values.values()), dtype=np.float64).reshape(-1)
        else:
            act_np = np.asarray(action_values, dtype=np.float64).reshape(-1)

        next_state_deg = act_np[:6].astype(np.float64)

        # Perfect-execution assumption: set MuJoCo to predicted joint targets
        set_mujoco_qpos(mj_model, mj_data, next_state_deg)
        eef_z = get_eef_z(mj_model, mj_data)

        eef_heights.append(eef_z)
        joint_states.append(current_state_deg.copy())
        applied_states.append(next_state_deg.copy())
        timestamps.append(step_i / fps)

        current_state_deg = next_state_deg

    return {
        "eef_heights": np.array(eef_heights, dtype=np.float64),
        "joint_states": np.array(joint_states, dtype=np.float64),
        "applied_states": np.array(applied_states, dtype=np.float64),
        "timestamps": np.array(timestamps, dtype=np.float64),
    }


# ==============================================================================
# Per-rollout metrics (matches compare_steering_motion_eef.py column names)
# ==============================================================================

def compute_rollout_metrics(rollout_idx: int, result: dict, source_episode: int) -> dict:
    heights_m = result["eef_heights"]
    timestamps = result["timestamps"]
    n = len(heights_m)
    heights_cm = heights_m * 100.0

    # Transport phase: middle 20–80% of episode (same window as caa_neurons_finding.py)
    lo = int(0.20 * n)
    hi = int(0.80 * n)
    transport = heights_cm[lo:hi]

    def _safe(arr, fn):
        return float(fn(arr)) if len(arr) > 0 else float("nan")

    duration = float(timestamps[-1] - timestamps[0]) if n >= 2 else 0.0

    return {
        # Identifiers
        "rollout_index": rollout_idx,
        "source_episode": source_episode,
        "num_steps": n,
        "duration_s": duration,

        # EEF height — full episode (cm)
        "eef_height_mean_cm": _safe(heights_cm, np.mean),
        "eef_height_max_cm": _safe(heights_cm, np.max),
        "eef_height_min_cm": _safe(heights_cm, np.min),
        "eef_height_range_cm": _safe(heights_cm, lambda h: np.max(h) - np.min(h)),
        "eef_height_final_cm": float(heights_cm[-1]) if n > 0 else float("nan"),

        # Transport phase (cm) — more robust than full-episode max
        "eef_height_transport_peak_cm": _safe(transport, np.max),
        "eef_height_transport_mean_cm": _safe(transport, np.mean),

        # Raw metre values for compatibility with compare_steering_motion_eef.py
        "eef_height_mean": _safe(heights_m, np.mean),
        "eef_height_max": _safe(heights_m, np.max),
        "eef_height_min": _safe(heights_m, np.min),
        "eef_height_range": _safe(heights_m, lambda h: np.max(h) - np.min(h)),
        "eef_height_final": float(heights_m[-1]) if n > 0 else float("nan"),
    }


# ==============================================================================
# Aggregate statistics
# ==============================================================================

def bootstrap_ci(values: np.ndarray, n_boot: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    boot = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return float("nan")
    return float((b.mean() - a.mean()) / pooled)


def aggregate_metrics(rows: list, key: str = "eef_height_transport_peak_cm") -> dict:
    vals = np.array([r[key] for r in rows if key in r], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {}
    ci_lo, ci_hi = bootstrap_ci(vals)
    return {
        "metric": key,
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
    }


# ==============================================================================
# CSV / NPZ saving
# ==============================================================================

def save_csv(path: Path, rows: list):
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ==============================================================================
# Video rendering
# ==============================================================================

def _make_mj_camera():
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 0.65
    cam.azimuth = 315   # front-right view — matches real camera direction (arm moves RIGHT)
    cam.elevation = -10  # near table level, slight downward tilt
    cam.lookat[:] = np.array([0.0, 0.0, 0.16])
    return cam


def _draw_text(img: np.ndarray, lines: list[str], x: int = 12, y0: int = 24, dy: int = 22):
    for i, text in enumerate(lines):
        yy = y0 + i * dy
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def render_rollout_to_video(
    rollout_idx: int,
    applied_states: np.ndarray,      # [T, 6] joint degrees AFTER each action
    eef_heights: np.ndarray,         # [T] EEF z in metres
    episode_frames: list,            # source dataset frames for camera images
    mj_model,
    mj_data,
    output_path: Path,
    rename_map: dict,
    fps: float = 15.0,
    render_w: int = 640,
    render_h: int = 480,
):
    """
    Save an MP4 video showing:
      Left  — camera1 image (what the policy sees)
      Right — MuJoCo render of the arm in its simulated position

    The MuJoCo panel uses the arm's ACTUAL simulated joint angles so you can
    directly see whether the arm rises or stays flat under each steering condition.
    """
    renderer = mujoco.Renderer(mj_model, height=render_h, width=render_w)
    cam = _make_mj_camera()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (render_w * 2, render_h))

    n_steps = len(applied_states)
    peak_h_cm = float(np.max(eef_heights) * 100.0) if n_steps > 0 else 0.0

    for step_i in range(n_steps):
        # ---- Left panel: real camera image that the policy saw ----
        frame = episode_frames[min(step_i, len(episode_frames) - 1)]
        cam1_key = rename_map.get("observation.images.front", "observation.images.camera1")
        # Try renamed key first, then original
        img_tensor = frame.get(cam1_key) or frame.get("observation.images.camera1") or frame.get("observation.images.front")
        if img_tensor is not None:
            left_bgr = cv2.cvtColor(tensor_to_hwc_uint8(img_tensor), cv2.COLOR_RGB2BGR)
            left_bgr = cv2.resize(left_bgr, (render_w, render_h))
        else:
            left_bgr = np.zeros((render_h, render_w, 3), dtype=np.uint8)

        _draw_text(left_bgr, [
            f"Camera (policy input)",
            f"rollout={rollout_idx}  step={step_i}/{n_steps}",
        ])

        # ---- Right panel: MuJoCo render of simulated arm ----
        set_mujoco_qpos(mj_model, mj_data, applied_states[step_i])
        renderer.update_scene(mj_data, camera=cam)
        rgb = renderer.render()
        right_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        eef_z_cm = eef_heights[step_i] * 100.0

        # EEF height bar on the right panel
        bar_x = render_w - 28
        bar_max_h = int(render_h * 0.7)
        bar_bot = render_h - 20
        max_z_cm = 40.0
        bar_fill = int(bar_max_h * min(eef_z_cm / max_z_cm, 1.0))
        cv2.rectangle(right_bgr, (bar_x, bar_bot - bar_max_h), (bar_x + 18, bar_bot), (60, 60, 60), -1)
        cv2.rectangle(right_bgr, (bar_x, bar_bot - bar_fill), (bar_x + 18, bar_bot), (0, 220, 80), -1)
        cv2.putText(right_bgr, "H", (bar_x + 2, bar_bot - bar_max_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        _draw_text(right_bgr, [
            f"MuJoCo sim  EEF z={eef_z_cm:.1f} cm",
            f"peak so far={float(np.max(eef_heights[:step_i+1]) * 100):.1f} cm  (ep peak={peak_h_cm:.1f})",
        ])

        combined = np.concatenate([left_bgr, right_bgr], axis=1)
        writer.write(combined)

    writer.release()
    print(f"  [video] saved: {output_path}  ({n_steps} frames @ {fps:.0f} fps)")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset-conditioned offline steering evaluation for SmolVLA."
    )

    # Policy
    parser.add_argument("--policy-path", type=str,
        default="ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2")

    # Source dataset (visual context provider)
    parser.add_argument("--source-dataset", type=str,
        default="ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2",
        help="LeRobot dataset repo used for camera images. Can be training data or eval recordings.")
    parser.add_argument("--source-episodes", type=int, nargs="+", default=None,
        help="Episode indices to use as visual context. Default: all episodes.")
    parser.add_argument("--source-episode-range", type=int, nargs=2, default=None,
        metavar=("START", "END"),
        help="Use episodes [START, END) from the source dataset.")

    # Rollout config
    parser.add_argument("--n-rollouts", type=int, default=30,
        help="Total sim rollouts. Cycles through source episodes if more rollouts than episodes.")
    parser.add_argument("--steps-per-rollout", type=int, default=None,
        help="Steps per rollout. Defaults to length of source episode.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--task", type=str, default="Put the red cube in the box.")

    # Steering
    parser.add_argument("--steering", type=str, default="none",
        choices=["none", "caa", "physical"],
        help="Steering method to apply.")
    parser.add_argument("--alpha", type=float, default=2.0,
        help="Steering coefficient. Positive = high trajectory direction.")
    parser.add_argument("--caa-path", type=str, default=None,
        help="Path to CAA .pt file (for --steering caa).")
    parser.add_argument("--neurons-json", type=str, default=None,
        help=(
            "JSON file with neuron sets (for --steering physical). "
            'Format: {"high": {"layer": [neurons]}, "low": {"layer": [neurons]}}'
        ))
    parser.add_argument("--steer-direction", type=str, default="high",
        choices=["high", "low"],
        help="Which neuron set to use when --steering physical.")

    # Rename map: training dataset uses front/top/wrist; eval datasets use camera1/2/3
    parser.add_argument("--rename-map", type=str, default=None,
        help=(
            'JSON string for camera key rename. '
            'Default maps front/top/wrist → camera1/2/3 for training datasets. '
            'Pass "{}" (empty dict) if source dataset already uses camera1/2/3.'
        ))

    # MuJoCo
    parser.add_argument("--xml-path", type=str, default=None,
        help="Path to follower.xml. Defaults to the one next to this script.")

    # Video
    parser.add_argument("--save-video", action="store_true",
        help="Render side-by-side MP4 videos (camera view + MuJoCo sim view) for selected rollouts.")
    parser.add_argument("--video-rollouts", type=int, nargs="+", default=None,
        help="Rollout indices to save videos for. Default: first 3 rollouts.")
    parser.add_argument("--video-fps", type=float, default=15.0,
        help="Video frame rate (default 15). Use half source fps since frames play back faster.")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)

    # Output
    parser.add_argument("--out-dir", type=str, default="sim_eval_out")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------------ Device
    if args.device is None:
        device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[*] Device: {device}")

    # --------------------------------------------------------- Rename map
    if args.rename_map is not None:
        rename_map = json.loads(args.rename_map)
    else:
        # Default: training dataset camera names → policy camera names
        rename_map = {
            "observation.images.front": "observation.images.camera1",
            "observation.images.top":   "observation.images.camera2",
            "observation.images.wrist": "observation.images.camera3",
        }
    print(f"[*] Rename map: {rename_map}")

    # ---------------------------------------------------------- MuJoCo FK
    if args.xml_path is not None:
        xml_path = Path(args.xml_path)
    else:
        xml_path = Path(__file__).resolve().parent / "follower.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"follower.xml not found at: {xml_path}")
    mj_model, mj_data = load_mujoco(str(xml_path))
    print(f"[*] Loaded MuJoCo model: {xml_path}")

    # ------------------------------------------------------- Source dataset
    print(f"[*] Loading source dataset: {args.source_dataset}")
    source_ds = LeRobotDataset(args.source_dataset)
    ep_ranges = get_episode_frame_ranges(source_ds)

    if args.source_episodes is not None:
        selected_ep_indices = args.source_episodes
    elif args.source_episode_range is not None:
        start, end = args.source_episode_range
        selected_ep_indices = list(range(start, end))
    else:
        selected_ep_indices = list(range(source_ds.num_episodes))

    selected_ep_indices = [i for i in selected_ep_indices if i < len(ep_ranges)]
    if not selected_ep_indices:
        raise ValueError("No valid source episodes selected.")

    print(f"[*] Using {len(selected_ep_indices)} source episodes: {selected_ep_indices[:10]}{'...' if len(selected_ep_indices) > 10 else ''}")

    # Lazy frame loader with a small LRU-style cache (avoids reloading when cycling).
    # Keeps at most N_CACHE episodes in RAM at once — one episode of 3×640×480 images
    # is ~350 MB, so 3 episodes ≈ 1 GB, which is safe on any GPU workstation.
    N_CACHE = 3
    _frame_cache: dict[int, list] = {}
    _cache_order: list[int] = []

    def load_episode_frames(ep_idx: int) -> list:
        if ep_idx in _frame_cache:
            return _frame_cache[ep_idx]
        # Evict oldest entry if cache is full
        if len(_cache_order) >= N_CACHE:
            evict = _cache_order.pop(0)
            del _frame_cache[evict]
        start_f, end_f = ep_ranges[ep_idx]
        frames = [source_ds[fi] for fi in range(start_f, end_f)]
        _frame_cache[ep_idx] = frames
        _cache_order.append(ep_idx)
        print(f"  [cache] loaded ep {ep_idx}: {len(frames)} frames (cache size: {len(_frame_cache)})")
        return frames

    # ------------------------------------------------------- Policy + preprocessors
    rename_map_for_policy = rename_map

    print(f"[*] Loading policy: {args.policy_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.device = str(device)

    # Patch dataset meta for policy (rename camera features)
    import copy as _copy
    patched_meta = _copy.deepcopy(source_ds.meta)
    for old_k, new_k in rename_map.items():
        if old_k in patched_meta.features:
            patched_meta.features[new_k] = patched_meta.features.pop(old_k)
        if hasattr(patched_meta, "stats") and patched_meta.stats and old_k in patched_meta.stats:
            patched_meta.stats[new_k] = patched_meta.stats.pop(old_k)

    policy = make_policy(policy_cfg, ds_meta=patched_meta)
    policy.eval()
    if hasattr(policy, "to"):
        policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        dataset_stats=patched_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    # ------------------------------------------------------- Steering setup
    print(f"\n[*] Steering: {args.steering} | alpha={args.alpha}")

    if args.steering == "none":
        clear_all_steering(policy)
        print("[*] No steering applied (baseline).")

    elif args.steering == "caa":
        if args.caa_path is None:
            raise ValueError("--caa-path required for --steering caa")
        if not Path(args.caa_path).exists():
            raise FileNotFoundError(f"CAA file not found: {args.caa_path}")
        setup_caa_steering(policy, args.caa_path, alpha=args.alpha)

    elif args.steering == "physical":
        if args.neurons_json is None:
            raise ValueError("--neurons-json required for --steering physical")
        with open(args.neurons_json) as f:
            neurons_raw = json.load(f)
        direction = args.steer_direction
        if direction not in neurons_raw:
            raise ValueError(
                f"Direction '{direction}' not found in {args.neurons_json}. "
                f"Available: {list(neurons_raw.keys())}"
            )
        # Convert string keys to int
        neurons = {int(layer_k): neuron_list for layer_k, neuron_list in neurons_raw[direction].items()}
        setup_physical_neuron_steering(policy, neurons, alpha=args.alpha)

    # ------------------------------------------------------- Run rollouts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_eef_traces = []

    video_rollout_set = set(args.video_rollouts if args.video_rollouts is not None else range(3))
    video_dir = Path(args.out_dir) / "videos"
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Running {args.n_rollouts} sim rollouts...")
    if args.save_video:
        print(f"[*] Videos will be saved for rollouts: {sorted(video_rollout_set)}")

    for rollout_idx in tqdm(range(args.n_rollouts), desc="Rollouts"):
        # Cycle through selected source episodes
        ep_idx = selected_ep_indices[rollout_idx % len(selected_ep_indices)]
        frames = load_episode_frames(ep_idx)

        if not frames:
            print(f"[!] Episode {ep_idx} has no frames, skipping rollout {rollout_idx}.")
            continue

        # Initial joint state: first frame of source episode
        first_state = frames[0].get("observation.state")
        if first_state is None:
            print(f"[!] Episode {ep_idx} frame 0 missing observation.state, skipping.")
            continue

        if torch.is_tensor(first_state):
            initial_state_deg = first_state.detach().cpu().numpy().reshape(-1)[:6].astype(np.float64)
        else:
            initial_state_deg = np.asarray(first_state, dtype=np.float64).reshape(-1)[:6]

        result = run_sim_rollout(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            episode_frames=frames,
            initial_state_deg=initial_state_deg,
            mj_model=mj_model,
            mj_data=mj_data,
            task=args.task,
            rename_map=rename_map_for_policy,
            steps=args.steps_per_rollout,
            fps=args.fps,
        )

        metrics = compute_rollout_metrics(rollout_idx, result, ep_idx)
        all_metrics.append(metrics)
        all_eef_traces.append(result["eef_heights"])

        # Optional video for this rollout
        if args.save_video and rollout_idx in video_rollout_set:
            video_path = video_dir / f"rollout_{rollout_idx:03d}_ep{ep_idx}_{args.steering}.mp4"
            render_rollout_to_video(
                rollout_idx=rollout_idx,
                applied_states=result["applied_states"],
                eef_heights=result["eef_heights"],
                episode_frames=frames,
                mj_model=mj_model,
                mj_data=mj_data,
                output_path=video_path,
                rename_map=rename_map_for_policy,
                fps=args.video_fps,
                render_w=args.video_width,
                render_h=args.video_height,
            )

    # ------------------------------------------------------- Save outputs
    save_csv(out_dir / "rollout_metrics.csv", all_metrics)
    print(f"[✓] Saved: {out_dir / 'rollout_metrics.csv'}")

    # Save raw EEF traces (zero-padded to max length for stacking)
    if all_eef_traces:
        max_len = max(len(t) for t in all_eef_traces)
        trace_arr = np.full((len(all_eef_traces), max_len), np.nan, dtype=np.float64)
        for i, trace in enumerate(all_eef_traces):
            trace_arr[i, :len(trace)] = trace
        np.savez_compressed(
            out_dir / "eef_height_traces.npz",
            traces_m=trace_arr,
            traces_cm=trace_arr * 100.0,
        )
        print(f"[✓] Saved: {out_dir / 'eef_height_traces.npz'}")

    # Aggregate summary
    peak_key = "eef_height_transport_peak_cm"
    mean_key = "eef_height_transport_mean_cm"
    summary = {
        "steering": args.steering,
        "alpha": args.alpha,
        "n_rollouts": len(all_metrics),
        "task": args.task,
        "source_dataset": args.source_dataset,
        "transport_peak_cm": aggregate_metrics(all_metrics, peak_key),
        "transport_mean_cm": aggregate_metrics(all_metrics, mean_key),
        "full_episode_max_cm": aggregate_metrics(all_metrics, "eef_height_max_cm"),
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Saved: {out_dir / 'summary.json'}")

    # Print quick summary
    peak = summary["transport_peak_cm"]
    if peak:
        print(
            f"\n[RESULT] Transport peak EEF height (cm): "
            f"mean={peak.get('mean', float('nan')):.2f} ± {peak.get('std', float('nan')):.2f}, "
            f"median={peak.get('median', float('nan')):.2f}, "
            f"CI95=[{peak.get('ci95_lo', float('nan')):.2f}, {peak.get('ci95_hi', float('nan')):.2f}], "
            f"n={peak.get('n', 0)}"
        )

    print(f"\n[✓] All outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
