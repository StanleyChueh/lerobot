#!/usr/bin/env python3

# SAME OBSERVATION WILL LEAD TO SAME RESULT!!!!!

"""
replay_lift_action_diagnostics.py

Offline replay tests for high-intervention lift instability.

This script conducts three tests:

Test 1:
  Repeatedly predict from the exact same reach/about-to-grasp observation.
  Goal: check whether the policy/intervention is deterministic for the same input.

Test 2:
  Replay chunk-1 reach/about-to-grasp observations from all high-intervention rollouts.
  Goal: check whether visually similar reach observations already produce different lift actions.

Test 3:
  Correlate replayed chunk-1 predicted actions with actual chunk-2 EEF height.
  Goal: identify which action dimensions predict the later lift height.

Expected high debug folder:

debug_runs/20260529_112158_high/
  episode_000000/
    debug_chunk_rawid_0_000_observation_frame.pt
    debug_chunk_rawid_1_001_observation_frame.pt
    debug_chunk_rawid_2_002_observation_frame.pt
    ...
  episode_000001/
  ...

Run example:

python src/lerobot/scripts/analyze_high_low_baseline_replay_reach_to_predict_lift.py \
  --high "$HIGH_RUN" \
  --policy-path "$POLICY" \
  --dataset-repo-id "$DATASET" \
  --xml "$XML" \
  --out analysis_replay_high_lift_prof \
  --intervention-name high_transport \
  --alpha 6.0 \
  --task "$TASK" \
  --reach-chunk 1 \
  --lift-chunk 2 \
  --repeat-exact 100 \
  --repeat-per-observation 10 \
  --focus-dims 1,2,5 \
  --rename-map-json "$RENAME_MAP"

Important:
  Use the same --intervention-name and --alpha that you used when recording the high run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy

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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


# ==============================================================================
# 1. EEF FK calibration copied from your physical_neuron_finding_eef.py
# ==============================================================================

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


def state_to_q_rad(state_vec: np.ndarray) -> np.ndarray:
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)
    if state_vec.size < 6:
        raise ValueError(f"State dimensionality must be >= 6, got {state_vec.shape}")

    raw_deg = state_vec[:6].copy()
    raw_delta = raw_deg[CALIB_ORDER] - CALIB_REST_STATE_DEG[CALIB_ORDER]
    q_deg = CALIB_TARGET_REST_DEG + CALIB_SIGN * CALIB_SCALE * raw_delta
    return np.deg2rad(q_deg)


def load_fk_model(xml_path: Path):
    if mujoco is None:
        raise ImportError(
            "Could not import mujoco. Run this in the same environment where "
            f"physical_neuron_finding_eef.py works. Original error: {MUJOCO_IMPORT_ERROR}"
        )
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def compute_eef_height_from_state(state_vec: np.ndarray, mj_model, mj_data) -> float:
    q = state_to_q_rad(state_vec)
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    mujoco.mj_resetData(mj_model, mj_data)

    for i, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in XML: {joint_name}")

        qpos_adr = mj_model.jnt_qposadr[joint_id]
        mj_data.qpos[qpos_adr] = float(q[i])

    mujoco.mj_forward(mj_model, mj_data)

    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    if site_id < 0:
        raise ValueError("Site not found in XML: end_effector_site")

    return float(mj_data.site_xpos[site_id].copy()[2])


# ==============================================================================
# 2. Steering neuron config copied from your recorder
# ==============================================================================

SEMANTIC_NEURON_SETS: dict[str, dict[int, list[int]]] = {
    "low_transport_paper": {
        1: [1222],
        3: [2003],
        5: [1877, 1904],
        10: [2349],
        13: [1744],
    },
    "high_transport_paper": {
        2: [826],
        3: [369],
        5: [2102],
        7: [1151],
        9: [2554],
        13: [414],
    },
    "high_transport": {
        0: [1293],
        1: [1050],
        3: [2259],
        4: [1183],
        7: [295],
        11: [1115, 1595],
        13: [431],
        14: [736, 805],
    },
    "low_transport": {
        3: [962],
        4: [1627],
        6: [587],
        7: [1007],
        9: [149],
        11: [1066],
        12: [629, 1164],
        14: [423],
        15: [1886],
    },
    "high_transport_clean_dataset": {
        6: [1816],
        9: [1596],
        11: [665, 1273],
        12: [1937],
        13: [489, 500, 1034, 1261],
        15: [1964],
    },
    "low_transport_clean_dataset": {
        3: [1556],
        6: [1558],
        8: [1034, 2114],
        10: [454, 2135],
        11: [188, 988, 1115],
        14: [1836],
    },
    "green": {
        0: [1930, 491, 2532, 1677, 930, 1286, 1429],
        1: [805, 1596],
        2: [2033],
        4: [1854],
        5: [416],
        6: [1767],
        7: [6, 2055],
        8: [1278],
        10: [997],
        14: [156],
        15: [848, 2261],
    },
    "red": {
        0: [1461, 2168, 1728, 1996, 1435],
        2: [702, 672],
        4: [1262],
        6: [1633],
        7: [2415, 1466, 934, 2125, 188],
        8: [508],
        9: [847],
        11: [1022],
        12: [1396],
        14: [1924, 246],
    },
    "fast_transport": {
        7: [884],
        8: [735],
        12: [287],
        14: [1994],
    },
    "slow_transport": {
        0: [435],
        5: [779],
        8: [2269],
        10: [1333],
        11: [2157],
        13: [1456],
    },
    "right": {
        4: [1897, 1400],
        10: [1257],
        13: [1122],
        14: [479, 1650],
    },
    "left": {
        3: [583],
        4: [1936],
        5: [1872],
        6: [2374],
        9: [1941],
        11: [1367],
    },
}


def apply_activation_steering(policy, intervention_name: str, alpha: float, enable_steering: bool = True) -> None:
    """
    Reapply activation steering in the same style as the recorder.

    This intentionally avoids semantic-token printing because replay should be lightweight.
    """
    if intervention_name == "none" or alpha == 0.0 or not enable_steering:
        if hasattr(policy, "clear_activation_steering"):
            policy.clear_activation_steering()
        if intervention_name != "none" and alpha == 0.0 and hasattr(policy, "set_activation_steering"):
            if intervention_name not in SEMANTIC_NEURON_SETS:
                raise KeyError(f"Unknown intervention_name: {intervention_name}")
            policy.set_activation_steering(
                steering_neurons=SEMANTIC_NEURON_SETS[intervention_name],
                alpha=0.0,
                record_debug=True,
                top_k_runtime=10,
                enable_steering=False,
            )
        return

    if intervention_name not in SEMANTIC_NEURON_SETS:
        raise KeyError(
            f"Unknown intervention_name={intervention_name!r}. "
            f"Available: {sorted(SEMANTIC_NEURON_SETS.keys())}"
        )

    if hasattr(policy, "clear_activation_steering"):
        policy.clear_activation_steering()

    if not hasattr(policy, "set_activation_steering"):
        raise AttributeError(
            "Policy does not have set_activation_steering(). "
            "Use the same modeling_smolvla.py implementation as your recorder."
        )

    policy.set_activation_steering(
        steering_neurons=SEMANTIC_NEURON_SETS[intervention_name],
        alpha=alpha,
        record_debug=True,
    )


# ==============================================================================
# 3. General helpers
# ==============================================================================

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def natural_sort_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}

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

    arr = np.asarray(arr).astype(np.float32, copy=False)
    if arr.ndim == 0:
        out[prefix] = float(arr)
        return out

    flat = arr.reshape(-1)
    for i, v in enumerate(flat):
        key = f"{prefix}.{i}" if prefix else str(i)
        out[key] = float(v)

    return out


def vector_from_dict(d: dict[str, float], keys: list[str]) -> np.ndarray:
    return np.asarray([float(d.get(k, 0.0)) for k in keys], dtype=np.float32)

def load_action_to_obs_mapping(path: str | None) -> pd.DataFrame | None:
    if path is None or str(path).strip() == "":
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Action-to-observation mapping CSV not found: {p}")

    mapping = pd.read_csv(p)

    required = {"joint_idx", "a", "b"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            f"Mapping CSV is missing columns {missing}. "
            f"Expected columns include: {sorted(required)}"
        )

    return mapping


def map_action_to_obs_convention(action_vec: np.ndarray, mapping: pd.DataFrame | None) -> np.ndarray:
    """
    Convert raw policy action convention into follower observation_state convention
    before using observation-state FK.

    Important:
      - Use this for predicted/replayed action_values before FK.
      - Do NOT use this for actual observation.state, because observation.state
        is already in follower observation convention.
    """
    action_vec = np.asarray(action_vec, dtype=np.float64).reshape(-1)[:6].copy()

    if mapping is None:
        return action_vec

    out = action_vec.copy()
    for _, row in mapping.iterrows():
        j = int(row["joint_idx"])
        a = float(row["a"])
        b = float(row["b"])
        out[j] = a * out[j] + b

    return out


def compute_eef_height_from_action(
    action_vec: np.ndarray,
    action_to_obs_map: pd.DataFrame | None,
    mj_model,
    mj_data,
) -> tuple[float, np.ndarray]:
    """
    Correct predicted-action FK:
      raw action -> mapped observation convention -> FK.
    """
    obs_like_vec = map_action_to_obs_convention(action_vec, action_to_obs_map)
    z = compute_eef_height_from_state(obs_like_vec, mj_model, mj_data)
    return z, obs_like_vec


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


def parse_rename_map_json(text: str | None) -> dict[str, str]:
    if text is None or str(text).strip() == "":
        return {}
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--rename-map-json must decode to a JSON object")
    return {str(k): str(v) for k, v in obj.items()}


def infer_episode_idx(episode_name: str) -> int:
    m = re.search(r"(\d+)$", episode_name)
    return int(m.group(1)) if m else -1


def find_chunk_file(root: Path, episode_idx: int, chunk_idx: int) -> Path | None:
    episode_dir = root / f"episode_{episode_idx:06d}"
    if not episode_dir.exists():
        return None

    patterns = [
        f"debug_chunk_rawid_{chunk_idx}_{chunk_idx:03d}_observation_frame.pt",
        f"debug_chunk*_{chunk_idx:03d}_observation_frame.pt",
        f"*_{chunk_idx:03d}_observation_frame.pt",
    ]

    for pattern in patterns:
        matches = sorted(episode_dir.glob(pattern), key=lambda p: natural_sort_key(p.name))
        matches = [p for p in matches if "policy_internal" not in p.parts]
        if matches:
            return matches[0]

    return None


def list_episode_indices(root: Path) -> list[int]:
    out = []
    for p in sorted(root.glob("episode_*"), key=lambda x: natural_sort_key(x.name)):
        if p.is_dir():
            out.append(infer_episode_idx(p.name))
    return out


def load_debug_pt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_observation_frame(data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the saved observation_frame that was originally passed to predict_action.
    """
    skip = {"chunk_id", "keys", "action_values", "metadata"}
    obs = {}
    for key, value in data.items():
        if key in skip:
            continue
        if isinstance(key, str) and (
            key.startswith("observation.")
            or key.startswith("observation_")
            or key == "state"
        ):
            obs[key] = value
    if not obs:
        # Fallback: keep numeric/tensor keys except known debug keys.
        for key, value in data.items():
            if key not in skip:
                obs[key] = value
    return obs


def observation_tensors_to_numpy(observation: dict[str, Any]) -> dict[str, Any]:
    """
    predict_action.prepare_observation_for_inference expects numpy arrays,
    because it internally calls torch.from_numpy(...).

    Saved debug .pt files store observation values as torch.Tensor, so replay
    must convert tensors back to numpy before calling predict_action.
    """
    converted: dict[str, Any] = {}

    for key, value in observation.items():
        if torch.is_tensor(value):
            converted[key] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            converted[key] = value
        else:
            try:
                converted[key] = np.asarray(value)
            except Exception:
                converted[key] = value

    return converted

def is_observation_image_key(key: str) -> bool:
    key = str(key).lower()
    return (
        "image" in key
        or "camera" in key
        or key.endswith("front")
        or key.endswith("top")
        or key.endswith("wrist")
    )


def observation_image_to_uint8(value: Any) -> np.ndarray | None:
    try:
        arr = to_numpy(value)
    except Exception:
        return None

    arr = np.asarray(arr)

    # Remove batch/time dims if present
    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3:
        return None

    # CHW -> HWC
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= 4:
        arr = arr[..., :3]

    if arr.shape[-1] != 3:
        return None

    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)

    # normalized [0,1]
    if arr.max() <= 1.5 and arr.min() >= -0.5:
        arr = arr * 255.0
    # other abnormal range -> robust visualize
    elif arr.min() < 0.0 or arr.max() > 255.0:
        lo, hi = np.percentile(arr, [1, 99])
        if hi - lo < 1e-8:
            hi = lo + 1.0
        arr = (arr - lo) / (hi - lo) * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def collect_displayable_observation_images(observation_frame: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    image_items: list[tuple[str, np.ndarray]] = []

    # First pass: prefer keys that look like image/camera fields
    for key, value in observation_frame.items():
        if isinstance(key, str) and is_observation_image_key(key):
            img = observation_image_to_uint8(value)
            if img is not None:
                image_items.append((key, img))

    # Fallback: if nothing matched, try all values
    if not image_items:
        for key, value in observation_frame.items():
            if not isinstance(key, str):
                continue
            img = observation_image_to_uint8(value)
            if img is not None:
                image_items.append((key, img))

    return image_items


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(text)).strip("_")


def save_test1_observation_images(plot_dir: Path, observation_frame: dict[str, Any]) -> list[Path]:
    image_items = collect_displayable_observation_images(observation_frame)
    saved_paths: list[Path] = []

    if not image_items:
        warnings.warn("No displayable observation images found for Test 1.")
        return saved_paths

    # Save each image separately
    for key, img in image_items:
        safe_key = sanitize_filename(key)
        out_path = plot_dir / f"test1_exact_observation_{safe_key}.png"
        plt.imsave(out_path, img)
        saved_paths.append(out_path)

    # Save montage
    n = len(image_items)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (key, img) in zip(axes, image_items):
        ax.imshow(img)
        ax.set_title(key, fontsize=9)
        ax.axis("off")

    fig.suptitle("Test 1 exact observation used for repeated inference")
    fig.tight_layout()
    montage_path = plot_dir / "test1_exact_observation_montage.png"
    fig.savefig(montage_path, dpi=160)
    plt.close(fig)
    saved_paths.append(montage_path)

    return saved_paths


def save_test1_repeat_plot(
    plot_dir: Path,
    test1_trials: pd.DataFrame,
    action_cols_test1: list[str],
    focus_dims: list[int] | None = None,
) -> None:
    cols_to_plot = list(action_cols_test1)

    # If focus_dims exist, prefer plotting those first
    if focus_dims:
        focus_cols = [f"action_values.{d}" for d in focus_dims if f"action_values.{d}" in cols_to_plot]
        if len(focus_cols) > 0:
            cols_to_plot = focus_cols

    if len(cols_to_plot) == 0:
        warnings.warn("No action columns available for Test 1 repeat plot.")
        return

    plt.figure(figsize=(10, 5))
    x = test1_trials["trial"].to_numpy()

    for col in cols_to_plot:
        y = test1_trials[col].to_numpy(dtype=np.float64)
        plt.plot(x, y, marker="o", markersize=3, linewidth=1, label=col)

    plt.xlabel("repeat trial index")
    plt.ylabel("predicted action value")
    plt.title("Test 1: exact same observation repeated")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_dir / "test1_exact_same_observation_action_repeat_plot.png", dpi=160)
    plt.close()


def get_state_vec_from_obs_or_data(obs: dict[str, Any], data: dict[str, Any]) -> np.ndarray:
    for key in ["observation.state", "observation_state", "state"]:
        if key in obs:
            return np.asarray(to_numpy(obs[key]), dtype=np.float64).reshape(-1)
        if key in data:
            return np.asarray(to_numpy(data[key]), dtype=np.float64).reshape(-1)

    for source in [obs, data]:
        for key, value in source.items():
            if isinstance(key, str) and key.endswith("state"):
                return np.asarray(to_numpy(value), dtype=np.float64).reshape(-1)

    raise KeyError("Could not find observation.state in saved chunk.")


def get_task(data: dict[str, Any], fallback: str) -> str:
    metadata = data.get("metadata", {}) or {}
    return str(metadata.get("task", fallback))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_policy_processors(policy, preprocessor, postprocessor) -> None:
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(preprocessor, "reset"):
        preprocessor.reset()
    if hasattr(postprocessor, "reset"):
        postprocessor.reset()
    if hasattr(policy, "eval"):
        policy.eval()
    if hasattr(policy, "model") and hasattr(policy.model, "eval"):
        policy.model.eval()


# ==============================================================================
# 4. Model loading / replay
# ==============================================================================

@dataclass
class ReplayContext:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: Any
    robot_type: str
    use_amp: bool
    intervention_name: str
    alpha: float
    enable_steering: bool
    reset_seed_each_trial: bool
    seed: int


def load_policy_context(args) -> ReplayContext:
    device = get_safe_torch_device(args.device)

    print(f"[*] Loading dataset metadata: {args.dataset_repo_id}")
    if args.dataset_root:
        dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, video_backend=args.video_backend)
    else:
        dataset = LeRobotDataset(args.dataset_repo_id, video_backend=args.video_backend)

    print(f"[*] Loading policy config/model: {args.policy_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.device = args.device

    rename_map = parse_rename_map_json(args.rename_map_json)

    # The dataset metadata may use front/top/wrist, while the policy expects camera1/2/3.
    # make_policy(...) validates visual feature names, so ds_meta must be renamed before make_policy.
    policy_meta = copy.deepcopy(dataset.meta)

    for actual_key, expected_key in rename_map.items():
        if actual_key in policy_meta.features:
            policy_meta.features[expected_key] = policy_meta.features.pop(actual_key)

    policy = make_policy(policy_cfg, ds_meta=policy_meta)
    if hasattr(policy, "to"):
        policy = policy.to(device)
    if hasattr(policy, "eval"):
        policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    apply_activation_steering(
        policy=policy,
        intervention_name=args.intervention_name,
        alpha=args.alpha,
        enable_steering=not args.disable_steering,
    )

    robot_type = args.robot_type
    if robot_type is None:
        robot_type = getattr(dataset.meta, "robot_type", None) or getattr(dataset, "robot_type", None) or "koch_follower"

    return ReplayContext(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        robot_type=robot_type,
        use_amp=bool(policy_cfg.use_amp),
        intervention_name=args.intervention_name,
        alpha=float(args.alpha),
        enable_steering=not args.disable_steering,
        reset_seed_each_trial=bool(args.reset_seed_each_trial),
        seed=int(args.seed),
    )


def predict_once(
    ctx: ReplayContext,
    observation_frame: dict[str, Any],
    task: str,
    trial_idx: int,
) -> dict[str, float]:
    if ctx.reset_seed_each_trial:
        set_global_seed(ctx.seed + trial_idx)

    reset_policy_processors(ctx.policy, ctx.preprocessor, ctx.postprocessor)

    # Reapply steering after reset to mimic your recorder's per-episode setup.
    apply_activation_steering(
        policy=ctx.policy,
        intervention_name=ctx.intervention_name,
        alpha=ctx.alpha,
        enable_steering=ctx.enable_steering,
    )

    observation_frame_np = observation_tensors_to_numpy(observation_frame)

    with torch.no_grad():
        action_values = predict_action(
            observation=observation_frame_np,
            policy=ctx.policy,
            device=ctx.device,
            preprocessor=ctx.preprocessor,
            postprocessor=ctx.postprocessor,
            use_amp=ctx.use_amp,
            task=task,
            robot_type=ctx.robot_type,
        )

    return flatten_numeric(action_values, "action_values")


def replay_repeated(
    ctx: ReplayContext,
    observation_frame: dict[str, Any],
    task: str,
    repeat: int,
    source_path: str,
    episode_idx: int | None = None,
    save_idx: int | None = None,
) -> pd.DataFrame:
    rows = []
    for trial in range(repeat):
        action_dict = predict_once(ctx, observation_frame, task, trial_idx=trial)
        row = {
            "trial": trial,
            "source_path": source_path,
        }
        if episode_idx is not None:
            row["episode_idx"] = episode_idx
        if save_idx is not None:
            row["save_idx"] = save_idx
        row.update(action_dict)
        rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================================
# 5. Experiment logic
# ==============================================================================

def collect_high_episode_records(
    high_root: Path,
    reach_chunk: int,
    lift_chunk: int,
    mj_model,
    mj_data,
) -> pd.DataFrame:
    rows = []
    for ep_idx in list_episode_indices(high_root):
        reach_path = find_chunk_file(high_root, ep_idx, reach_chunk)
        lift_path = find_chunk_file(high_root, ep_idx, lift_chunk)

        if reach_path is None:
            warnings.warn(f"Missing reach chunk {reach_chunk} for episode {ep_idx}")
            continue
        if lift_path is None:
            warnings.warn(f"Missing lift chunk {lift_chunk} for episode {ep_idx}")
            continue

        reach_data = load_debug_pt(reach_path)
        lift_data = load_debug_pt(lift_path)

        reach_obs = extract_observation_frame(reach_data)
        lift_obs = extract_observation_frame(lift_data)

        reach_state = get_state_vec_from_obs_or_data(reach_obs, reach_data)
        lift_state = get_state_vec_from_obs_or_data(lift_obs, lift_data)

        reach_eef_z = compute_eef_height_from_state(reach_state, mj_model, mj_data)
        lift_eef_z = compute_eef_height_from_state(lift_state, mj_model, mj_data)

        saved_reach_action = flatten_numeric(reach_data.get("action_values", None), "saved_action_values")
        saved_lift_action = flatten_numeric(lift_data.get("action_values", None), "saved_lift_action_values")

        row = {
            "episode_idx": ep_idx,
            "episode": f"episode_{ep_idx:06d}",
            "reach_chunk": reach_chunk,
            "lift_chunk": lift_chunk,
            "reach_path": str(reach_path),
            "lift_path": str(lift_path),
            "reach_eef_z": reach_eef_z,
            "lift_eef_z": lift_eef_z,
            "delta_eef_z_lift_minus_reach": lift_eef_z - reach_eef_z,
            "task": get_task(reach_data, ""),
        }
        row.update(saved_reach_action)
        row.update(saved_lift_action)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("episode_idx")
    if len(df) == 0:
        raise RuntimeError("No complete high episodes found for reach/lift chunks.")
    return df


def assign_eef_modes(df: pd.DataFrame, manual_high: list[int] | None, manual_low: list[int] | None) -> pd.DataFrame:
    df = df.copy()

    if manual_high is not None or manual_low is not None:
        high_set = set(manual_high or [])
        low_set = set(manual_low or [])
        labels = []
        for ep in df["episode_idx"]:
            if int(ep) in high_set:
                labels.append("high_eef_mode")
            elif int(ep) in low_set:
                labels.append("low_eef_mode")
            else:
                labels.append("unlabeled")
        df["eef_mode"] = labels
        df["eef_mode_threshold"] = np.nan
        return df

    # Median split by actual lift chunk EEF height.
    threshold = float(df["lift_eef_z"].median())
    df["eef_mode"] = np.where(df["lift_eef_z"] >= threshold, "high_eef_mode", "low_eef_mode")
    df["eef_mode_threshold"] = threshold
    return df


def summarize_repeat_trials(df: pd.DataFrame, action_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in action_cols:
        vals = df[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "action_key": col,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "range": float(np.max(vals) - np.min(vals)),
                "max_abs_deviation_from_mean": float(np.max(np.abs(vals - np.mean(vals)))),
            }
        )
    return pd.DataFrame(rows).sort_values(["std", "range"], ascending=False)


def mean_prediction_for_episode(
    ctx: ReplayContext,
    reach_path: Path,
    repeat: int,
    fallback_task: str,
    episode_idx: int,
    reach_chunk: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    data = load_debug_pt(reach_path)
    obs = extract_observation_frame(data)
    task = get_task(data, fallback_task)
    trials = replay_repeated(
        ctx,
        observation_frame=obs,
        task=task,
        repeat=repeat,
        source_path=str(reach_path),
        episode_idx=episode_idx,
        save_idx=reach_chunk,
    )

    action_cols = sorted([c for c in trials.columns if c.startswith("action_values.")], key=natural_sort_key)
    mean_dict = {col: float(trials[col].mean()) for col in action_cols}
    std_dict = {f"{col}_repeat_std": float(trials[col].std(ddof=0)) for col in action_cols}
    merged = {}
    merged.update(mean_dict)
    merged.update(std_dict)
    return trials, merged


def group_action_stats(df: pd.DataFrame, action_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_rows = []
    for mode, g in df.groupby("eef_mode"):
        for col in action_cols:
            vals = g[col].to_numpy(dtype=np.float64)
            stats_rows.append(
                {
                    "eef_mode": mode,
                    "action_key": col,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "range": float(np.max(vals) - np.min(vals)),
                    "count": int(len(vals)),
                }
            )
    stats = pd.DataFrame(stats_rows)

    diff_rows = []
    high = df[df["eef_mode"] == "high_eef_mode"]
    low = df[df["eef_mode"] == "low_eef_mode"]
    for col in action_cols:
        if len(high) == 0 or len(low) == 0:
            continue
        high_vals = high[col].to_numpy(dtype=np.float64)
        low_vals = low[col].to_numpy(dtype=np.float64)
        pooled = math.sqrt(max(np.var(high_vals) + np.var(low_vals), 1e-12) / 2.0)
        diff_rows.append(
            {
                "action_key": col,
                "high_mode_mean": float(np.mean(high_vals)),
                "low_mode_mean": float(np.mean(low_vals)),
                "high_minus_low": float(np.mean(high_vals) - np.mean(low_vals)),
                "abs_high_minus_low": float(abs(np.mean(high_vals) - np.mean(low_vals))),
                "effect_z": float(abs(np.mean(high_vals) - np.mean(low_vals)) / pooled),
                "high_mode_std": float(np.std(high_vals, ddof=0)),
                "low_mode_std": float(np.std(low_vals, ddof=0)),
            }
        )
    diffs = pd.DataFrame(diff_rows).sort_values(["effect_z", "abs_high_minus_low"], ascending=False)
    return stats, diffs


def compute_correlations(df: pd.DataFrame, action_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
    rows = []
    for target in target_cols:
        y = df[target].to_numpy(dtype=np.float64)
        for col in action_cols:
            x = df[col].to_numpy(dtype=np.float64)
            if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
                corr = np.nan
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
            rows.append(
                {
                    "target": target,
                    "action_key": col,
                    "pearson_corr": corr,
                    "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "abs_corr"], ascending=[True, False])

def map_action_to_obs_convention(action_vec: np.ndarray, mapping: pd.DataFrame | None) -> np.ndarray:
    action_vec = np.asarray(action_vec, dtype=np.float64).reshape(-1)[:6].copy()

    if mapping is None:
        return action_vec

    out = action_vec.copy()
    for _, row in mapping.iterrows():
        j = int(row["joint_idx"])
        a = float(row["a"])
        b = float(row["b"])
        out[j] = a * out[j] + b

    return out


# ==============================================================================
# 6. Plots / report
# ==============================================================================

def save_plots(
    out_dir: Path,
    episode_df: pd.DataFrame,
    action_cols: list[str],
    focus_dims: list[int],
    test1_trials: pd.DataFrame | None = None,
    test1_action_cols: list[str] | None = None,
    test1_observation: dict[str, Any] | None = None,
) -> None:
    plot_dir = out_dir / "plots"
    safe_mkdir(plot_dir)

    # ------------------------------------------------------------------
    # Test 1 plots: exact same observation repeated
    # ------------------------------------------------------------------
    if test1_trials is not None and test1_action_cols is not None:
        save_test1_repeat_plot(
            plot_dir=plot_dir,
            test1_trials=test1_trials,
            action_cols_test1=test1_action_cols,
            focus_dims=focus_dims,
        )

    if test1_observation is not None:
        save_test1_observation_images(
            plot_dir=plot_dir,
            observation_frame=test1_observation,
        )

    # ------------------------------------------------------------------
    # Test 3: Lift EEF by episode and mode
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    for mode, g in episode_df.groupby("eef_mode"):
        plt.scatter(g["episode_idx"], g["lift_eef_z"], label=mode)
    plt.xlabel("High-run episode index")
    plt.ylabel("Actual chunk-2 EEF z")
    plt.title("High intervention: actual lift EEF height by episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "test3_lift_eef_by_episode.png", dpi=160)
    plt.close()

    # ------------------------------------------------------------------
    # Test 3: Focus action dims vs lift EEF
    # ------------------------------------------------------------------
    for dim in focus_dims:
        col = f"action_values.{dim}"
        if col not in episode_df.columns:
            continue
        plt.figure(figsize=(6, 5))
        for mode, g in episode_df.groupby("eef_mode"):
            plt.scatter(g[col], g["lift_eef_z"], label=mode)
        plt.xlabel(f"Replay predicted {col} from reach chunk")
        plt.ylabel("Actual chunk-2 EEF z")
        plt.title(f"Predicted {col} vs actual lift height")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"test3_{col.replace('.', '_')}_vs_lift_eef.png", dpi=160)
        plt.close()

    # ------------------------------------------------------------------
    # Test 2: Action dimension means by mode for focus dims
    # ------------------------------------------------------------------
    if focus_dims:
        rows = []
        for dim in focus_dims:
            col = f"action_values.{dim}"
            if col not in episode_df.columns:
                continue
            for mode, g in episode_df.groupby("eef_mode"):
                rows.append(
                    {
                        "mode": mode,
                        "dim": dim,
                        "mean": g[col].mean(),
                        "std": g[col].std(ddof=0),
                    }
                )
        plot_df = pd.DataFrame(rows)
        if len(plot_df):
            plt.figure(figsize=(10, 5))
            x = np.arange(len(focus_dims))
            width = 0.35
            modes = sorted(plot_df["mode"].unique())
            for i, mode in enumerate(modes):
                sub = plot_df[plot_df["mode"] == mode].set_index("dim").reindex(focus_dims)
                plt.bar(
                    x + (i - (len(modes) - 1) / 2) * width,
                    sub["mean"],
                    width=width,
                    label=mode,
                )
            plt.xticks(x, [str(d) for d in focus_dims])
            plt.xlabel("Action dimension")
            plt.ylabel("Mean replay predicted action")
            plt.title("High-EEF vs low-EEF mode action means")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "test2_mode_action_means_focus_dims.png", dpi=160)
            plt.close()

def write_report(
    out_dir: Path,
    args,
    test1_summary: pd.DataFrame,
    episode_df: pd.DataFrame,
    group_diffs: pd.DataFrame,
    correlations: pd.DataFrame,
    focus_dims: list[int],
) -> None:
    lines = []
    lines.append("# Replay lift action diagnostics")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```")
    lines.append(f"high_root = {args.high}")
    lines.append(f"policy_path = {args.policy_path}")
    lines.append(f"dataset_repo_id = {args.dataset_repo_id}")
    lines.append(f"intervention_name = {args.intervention_name}")
    lines.append(f"alpha = {args.alpha}")
    lines.append(f"reach_chunk = {args.reach_chunk}")
    lines.append(f"lift_chunk = {args.lift_chunk}")
    lines.append(f"repeat_exact = {args.repeat_exact}")
    lines.append(f"repeat_per_observation = {args.repeat_per_observation}")
    lines.append("```")
    lines.append("")

    # Compute these before using them in the interpretation.
    max_std = float(test1_summary["std"].max()) if len(test1_summary) else float("nan")
    max_range = float(test1_summary["range"].max()) if len(test1_summary) else float("nan")

    lines.append("## Test 1: exact same observation repeated")
    lines.append("")
    lines.append("If the same saved reach observation produces different actions across repeats, the policy/intervention path is nondeterministic or stateful.")
    lines.append("")
    lines.append("Top varying action dimensions:")
    lines.append("```")
    lines.append(test1_summary.head(20).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append(f"- Test 1 max action std: `{max_std:.8f}`")
    lines.append(f"- Test 1 max action range: `{max_range:.8f}`")
    if np.isfinite(max_range) and max_range < 1e-5:
        lines.append("- Interpretation: exact same observation gives numerically identical action. Policy replay is deterministic.")
    else:
        lines.append("- Interpretation: exact same observation gives non-identical action. Check stochastic sampling, dropout/train mode, CUDA nondeterminism, action queue, or steering hook state.")
    lines.append("")

    lines.append("Saved Test 1 visualization files:")
    lines.append("- `plots/test1_exact_same_observation_action_repeat_plot.png`")
    lines.append("- `plots/test1_exact_observation_montage.png`")
    lines.append("")
    lines.append("How to read Test 1:")
    lines.append("- `plots/test1_exact_observation_montage.png` shows the exact RGB observation used for repeated inference.")
    lines.append("- `plots/test1_exact_same_observation_action_repeat_plot.png` shows predicted action values across repeated inference calls.")
    lines.append("- If the montage is fixed and the repeated-action plot shows flat horizontal lines, the same saved observation is producing the same predicted action across repeats.")
    lines.append("- This means the replay path is deterministic for the same input under the same intervention setup.")
    lines.append("")

    lines.append("## High EEF mode split")
    lines.append("")
    lines.append("```")
    lines.append(
        episode_df[
            [
                "episode_idx",
                "eef_mode",
                "reach_eef_z",
                "lift_eef_z",
                "delta_eef_z_lift_minus_reach",
            ]
        ].to_string(index=False)
    )
    lines.append("```")
    lines.append("")

    lines.append("## Test 2: high-mode vs low-mode replay from chunk-1 observations")
    lines.append("")
    lines.append("This compares predicted actions from the reach/about-to-grasp observation inside high intervention only.")
    lines.append("")
    lines.append("Most separating action dimensions:")
    lines.append("```")
    lines.append(group_diffs.head(20).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Test 3: chunk-1 predicted action vs actual chunk-2 EEF height")
    lines.append("")
    lines.append("Top action dimensions correlated with actual lift height:")
    lines.append("```")
    lines.append(correlations.head(30).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Practical interpretation")
    lines.append("")
    lines.append("Use these rules:")
    lines.append("")
    lines.append("1. If Test 1 has near-zero variance, exact same input is stable.")
    lines.append("2. If Test 2 separates high-EEF and low-EEF modes, the decisive difference is already present in the chunk-1 observation/state.")
    lines.append("3. If Test 3 shows strong correlation between one predicted action dimension and chunk-2 EEF height, that dimension is a likely lift-amplitude control channel.")
    lines.append("4. If Test 1 is stable but Test 2 differs, the issue is input sensitivity, not random policy output.")
    lines.append("5. If Test 2 does not differ but Test 3 lift heights differ, the issue is likely robot execution/contact/grasp dynamics after the action is issued.")
    lines.append("")
    if focus_dims:
        lines.append(f"Focus action dimensions plotted: `{focus_dims}`")
        lines.append("")

    (out_dir / "diagnosis_replay.md").write_text("\n".join(lines), encoding="utf-8")


# ==============================================================================
# 7. Main
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", required=True, help="Path to high intervention debug folder.")
    parser.add_argument("--policy-path", required=True, help="Policy repo id or local policy path.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo id used for metadata/stats.")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root.")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--xml", required=True, help="Path to follower.xml.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--intervention-name", default="high_transport")
    parser.add_argument("--alpha", type=float, required=True, help="Use the exact high scalar used during recording.")
    parser.add_argument("--disable-steering", action="store_true")
    parser.add_argument("--task", default="Put the red cube in the box.")
    parser.add_argument("--robot-type", default="koch_follower")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rename-map-json", default=None)
    parser.add_argument(
        "--allow-raw-action-fk",
        action="store_true",
        help=(
            "Allow raw action_values to be used directly for FK. "
            "Only use for debugging; not for paper plots."
        ),
    )
    parser.add_argument("--reach-chunk", type=int, default=1)
    parser.add_argument("--lift-chunk", type=int, default=2)
    parser.add_argument("--test1-pt", default=None, help="Specific saved reach chunk .pt for Test 1. If omitted, first high_eef_mode episode is used.")
    parser.add_argument("--repeat-exact", type=int, default=100)
    parser.add_argument("--repeat-per-observation", type=int, default=10)
    parser.add_argument("--manual-high-eef-episodes", default=None, help="Comma/range list, e.g. 1,2,5,6,7")
    parser.add_argument("--manual-low-eef-episodes", default=None, help="Comma/range list, e.g. 0,3,4,8,9")
    parser.add_argument("--focus-dims", default="1,2,5", help="Action dims to emphasize in plots.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-seed-each-trial", action="store_true")
    parser.add_argument(
        "--action-to-obs-map-csv",
        default=None,
        help="CSV containing action -> observation_state affine mapping.",
    )
    args = parser.parse_args()

    action_to_obs_map = load_action_to_obs_mapping(args.action_to_obs_map_csv)

    if action_to_obs_map is None and not args.allow_raw_action_fk:
        raise RuntimeError(
            "Refusing to compute predicted EEF height from raw action_values. "
            "Pass --action-to-obs-map-csv <mapping.csv>, or pass "
            "--allow-raw-action-fk only for debugging."
        )

    # action_to_obs_map = None
    # if args.action_to_obs_map_csv is not None:
    #     action_to_obs_map = pd.read_csv(args.action_to_obs_map_csv)

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    set_global_seed(args.seed)

    xml_path = Path(args.xml)
    if not xml_path.exists():
        xml_path = Path.cwd() / args.xml
    mj_model, mj_data = load_fk_model(xml_path)

    high_root = Path(args.high)

    # Load high-run metadata first, then split into high/lower EEF modes.
    episode_df = collect_high_episode_records(
        high_root=high_root,
        reach_chunk=args.reach_chunk,
        lift_chunk=args.lift_chunk,
        mj_model=mj_model,
        mj_data=mj_data,
    )
    episode_df = assign_eef_modes(
        episode_df,
        manual_high=parse_episode_list(args.manual_high_eef_episodes),
        manual_low=parse_episode_list(args.manual_low_eef_episodes),
    )
    episode_df.to_csv(out_dir / "high_episode_reach_lift_eef.csv", index=False)

    # Load policy after metadata loading; this can take time.
    ctx = load_policy_context(args)

    # Test 1: exact same observation repeated.
    if args.test1_pt is not None:
        test1_path = Path(args.test1_pt)
        test1_ep = None
    else:
        high_mode = episode_df[episode_df["eef_mode"] == "high_eef_mode"].sort_values("episode_idx")
        if len(high_mode) == 0:
            high_mode = episode_df.sort_values("episode_idx")
        test1_ep = int(high_mode.iloc[0]["episode_idx"])
        test1_path = Path(high_mode.iloc[0]["reach_path"])

    test1_data = load_debug_pt(test1_path)
    test1_obs = extract_observation_frame(test1_data)
    test1_task = get_task(test1_data, args.task)

    print(f"[*] Test 1 exact replay source: {test1_path}")
    test1_trials = replay_repeated(
        ctx=ctx,
        observation_frame=test1_obs,
        task=test1_task,
        repeat=args.repeat_exact,
        source_path=str(test1_path),
        episode_idx=test1_ep,
        save_idx=args.reach_chunk,
    )
    test1_trials.to_csv(out_dir / "test1_exact_repeat_actions.csv", index=False)

    action_cols_test1 = sorted([c for c in test1_trials.columns if c.startswith("action_values.")], key=natural_sort_key)
    test1_summary = summarize_repeat_trials(test1_trials, action_cols_test1)
    test1_summary.to_csv(out_dir / "test1_exact_repeat_summary.csv", index=False)

    # Test 2 and 3: replay all high reach observations.
    print("[*] Test 2/3 replaying each high episode reach observation...")
    all_trial_dfs = []
    mean_rows = []

    for row in episode_df.sort_values("episode_idx").itertuples(index=False):
        reach_path = Path(row.reach_path)
        trials, pred = mean_prediction_for_episode(
            ctx=ctx,
            reach_path=reach_path,
            repeat=args.repeat_per_observation,
            fallback_task=args.task,
            episode_idx=int(row.episode_idx),
            reach_chunk=args.reach_chunk,
        )
        trials["eef_mode"] = row.eef_mode
        trials["lift_eef_z"] = row.lift_eef_z
        trials["reach_eef_z"] = row.reach_eef_z
        all_trial_dfs.append(trials)

        mean_row = row._asdict()
        mean_row.update(pred)
        mean_rows.append(mean_row)

    all_trials = pd.concat(all_trial_dfs, ignore_index=True)
    all_trials.to_csv(out_dir / "test2_all_reach_replay_trials.csv", index=False)

    episode_pred_df = pd.DataFrame(mean_rows).sort_values("episode_idx")
    episode_pred_df.to_csv(out_dir / "test2_episode_mean_replay_actions.csv", index=False)

    action_cols = sorted([c for c in episode_pred_df.columns if c.startswith("action_values.") and not c.endswith("_repeat_std")], key=natural_sort_key)

    group_stats, group_diffs = group_action_stats(episode_pred_df, action_cols)
    group_stats.to_csv(out_dir / "test2_group_action_stats.csv", index=False)
    group_diffs.to_csv(out_dir / "test2_high_vs_low_eef_mode_action_diff.csv", index=False)

    target_cols = ["lift_eef_z", "delta_eef_z_lift_minus_reach"]
    correlations = compute_correlations(episode_pred_df, action_cols, target_cols)
    correlations.to_csv(out_dir / "test3_action_vs_lift_eef_correlations.csv", index=False)

    focus_dims = parse_episode_list(args.focus_dims) or []
    save_plots(
        out_dir=out_dir,
        episode_df=episode_pred_df,
        action_cols=action_cols,
        focus_dims=focus_dims,
        test1_trials=test1_trials,
        test1_action_cols=action_cols_test1,
        test1_observation=test1_obs,
    )

    write_report(
        out_dir=out_dir,
        args=args,
        test1_summary=test1_summary,
        episode_df=episode_pred_df,
        group_diffs=group_diffs,
        correlations=correlations,
        focus_dims=focus_dims,
    )

    print("[DONE] Replay lift diagnostics complete.")
    print(f"[DONE] Output directory: {out_dir.resolve()}")
    print()
    print("Mode split:")
    print(episode_pred_df[["episode_idx", "eef_mode", "reach_eef_z", "lift_eef_z"]].to_string(index=False))
    print()
    print("Test 1 top action repeat variance:")
    print(test1_summary.head(10).to_string(index=False))
    print()
    print("Test 2 top high-mode vs low-mode action differences:")
    print(group_diffs.head(10).to_string(index=False))
    print()
    print("Test 3 top action correlations with lift EEF:")
    print(correlations.head(10).to_string(index=False))
    print()
    print("Key outputs:")
    print(f"  - {out_dir / 'diagnosis_replay.md'}")
    print(f"  - {out_dir / 'test1_exact_repeat_summary.csv'}")
    print(f"  - {out_dir / 'test2_episode_mean_replay_actions.csv'}")
    print(f"  - {out_dir / 'test2_high_vs_low_eef_mode_action_diff.csv'}")
    print(f"  - {out_dir / 'test3_action_vs_lift_eef_correlations.csv'}")
    print(f"  - {out_dir / 'plots' / 'test1_exact_observation_montage.png'}")
    print(f"  - {out_dir / 'plots' / 'test1_exact_same_observation_action_repeat_plot.png'}")
    print(f"  - {out_dir / 'plots'}")


if __name__ == "__main__":
    main()