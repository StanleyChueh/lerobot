# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''

python src/lerobot/scripts/lerobot_record_realtime_attention_weight_stanley_steering_test.py  --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras='{
    camera1: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera2: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera3: {type: opencv, index_or_path: 11, width: 640, height: 480, fps: 30, fourcc: MJPG},
  }'   --dataset.single_task="Put the red cube in the box."   --dataset.repo_id=ethanCSL/eval_steering_ours_high_6   --dataset.episode_time_s=500000   --dataset.num_episodes=20   --teleop.type=koch_leader   --teleop.port=/dev/ttyUSB_leader   --teleop.id=my_awesome_leader_arm   --policy.path=ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 --dataset.reset_time_s=5 


'''
import logging
import math
import time
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so100_leader,
    so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Strip ANSI escape sequences and control characters from terminal input
# to prevent keyboard/terminal artifacts (e.g. arrow keys, hotkeys)
# from polluting language prompts passed to the policy.
import re
import cv2
import numpy as np
import rerun as rr
import torch

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")


## Show top token
from lerobot.scripts.lerobot_reord_top_token_steering import (
    extract_semantic_embeddings_from_policy,
    print_steered_neurons_info,
)

@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class FixedActionConfig:
    """
    Experiment A: capture/replay an identical action sequence on the real robot.

    mode:
      "off"     = normal policy/teleop behavior
      "capture" = run policy normally, capture exact sent actions
      "replay"  = ignore policy action, replay saved action sequence
    """
    mode: str = "off"
    path: str = "fixed_action_sequences/reach1_to_lift2.pt"

    # Capture actions from generated chunk start_chunk until before stop_chunk.
    # For your current phase:
    #   chunk 1 = reach/about-to-grasp
    #   chunk 2 = lift/highest
    # So capture chunk 1 actions until chunk 2 starts.
    start_chunk: int = 1
    stop_chunk: int = 2

    # Safety limit in case stop_chunk is never reached.
    max_steps: int = 120

    # In replay mode, pause before executing the sequence.
    wait_before_replay: bool = True

    # Replay speed multiplier.
    # 1.0 = normal fps replay.
    # 3.0 = three times slower.
    # 5.0 = five times slower.
    replay_slowdown: float = 4.0

@dataclass
class ActionTraceConfig:
    """
    Log full per-control-step policy/action/state trajectory.

    Purpose:
      Diagnose why normal high-intervention rollouts produce different lift heights.
      This logs every executed policy step, not only chunk-level action_values.
    """
    enabled: bool = False
    save_name: str = "policy_action_trace.pt"

@dataclass
class InterventionConfig:
    """
    Runtime intervention/label config.

    label:
      "baseline", "high", "low" for analysis metadata.

    name:
      neuron set name, e.g. "high_transport", "low_transport".
      For baseline, use name="high_transport" or "low_transport" with alpha=0.0
      and enable_steering=False.

    alpha:
      steering strength.

    enable_steering:
      False means no actual steering is applied.
    """
    label: str = "high"
    name: str = "high_transport"
    alpha: float = 6.0
    enable_steering: bool = True

@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to controlRobotClient the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = False
    # Resume recording on an existing dataset.
    resume: bool = False
    #Debug freq
    debug_freq: bool = False
    fixed_action: FixedActionConfig = field(default_factory=FixedActionConfig)
    action_trace: ActionTraceConfig = field(default_factory=ActionTraceConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")
        
        allowed_labels = {"baseline", "high", "low"}
        if self.intervention.label not in allowed_labels:
            raise ValueError(
                f"Unknown intervention.label={self.intervention.label}. "
                f"Expected one of {sorted(allowed_labels)}."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""

def normalize_fixed_action_mode(mode) -> str:
    """
    Some config parsers treat 'off' as boolean False.
    Normalize it back to the intended fixed-action mode string.
    """
    if isinstance(mode, bool):
        if mode is False:
            return "off"
        raise ValueError(
            "fixed_action.mode=True is ambiguous. "
            "Use one of: off, capture, replay."
        )

    mode_str = str(mode).lower().strip()

    if mode_str in {"off", "false", "0", "none", "disabled", "disable"}:
        return "off"

    if mode_str in {"capture", "record"}:
        return "capture"

    if mode_str in {"replay", "playback"}:
        return "replay"

    raise ValueError(
        f"Unknown fixed_action.mode: {mode}. "
        "Expected one of: off, capture, replay."
    )

def extract_cross_attention_maps(attn_matrix, token_layout):
    if attn_matrix is None or token_layout is None:
        return [], {}

    # attn_matrix: [Q, K]
    # Q = action/query tokens
    # K = prefix tokens: image/language/state/etc.
    mean_action_attn = attn_matrix.mean(dim=0)   # [K]

    image_maps = []
    contrib = {}

    total_sum = mean_action_attn.sum().item() + 1e-8

    for seg in token_layout:
        start, end = seg["start"], seg["end"]
        seg_attn = mean_action_attn[start:end]

        seg_sum = seg_attn.sum().item()
        seg_mean = seg_attn.mean().item() if seg_attn.numel() > 0 else 0.0
        seg_max = seg_attn.max().item() if seg_attn.numel() > 0 else 0.0
        seg_tokens = int(seg_attn.numel())
        seg_ratio = seg_sum / total_sum

        stats = {
            "sum": seg_sum,
            "mean": seg_mean,
            "max": seg_max,
            "tokens": seg_tokens,
            "ratio": seg_ratio,
        }

        if seg["type"] == "image":
            image_maps.append({
                "image_index": seg["image_index"],
                "heat_1d": seg_attn,
                "stats": stats,
            })

        key = seg["type"]
        if seg["type"] == "image":
            key = f'image_{seg["image_index"]}'

        contrib[key] = stats

    return image_maps, contrib

def process_heatmap(heat_1d, original_size=(480, 640)):
    if heat_1d is None:
        return None

    heat_1d = heat_1d.float().detach().cpu()
    num_tokens = heat_1d.numel()
    if num_tokens == 0:
        return None

    side = int(math.sqrt(num_tokens))
    heat_2d = None

    if side * side == num_tokens:
        heat_2d = heat_1d.reshape(side, side).numpy()
    else:
        for h in range(side, 0, -1):
            if num_tokens % h == 0:
                w = num_tokens // h
                heat_2d = heat_1d.reshape(h, w).numpy()
                break

    if heat_2d is None:
        logging.warning("Unable to reshape %s attention tokens into a 2D heatmap.", num_tokens)
        return None

    heat_resized = cv2.resize(
        heat_2d,
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    v_min, v_max = np.percentile(heat_resized, [0, 98])
    heat_norm = np.clip((heat_resized - v_min) / (v_max - v_min + 1e-6), 0, 1)
    return heat_norm


def to_hwc_uint8(img):
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()

    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image tensor/array, got shape {img.shape}")

    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)

    return np.ascontiguousarray(img)

def make_debug_run_dir(root: str = "debug_runs") -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[DEBUG] Saving debug files to: {run_dir.resolve()}")
    return run_dir


def save_debug_observation_frame(
    observation_frame,
    chunk_id: int,
    debug_dir: Path,
    prefix: str = "debug_chunk",
    action_values=None,
    metadata: dict | None = None,
):
    """
    Save the exact observation_frame that is passed into predict_action().
    PNG images are saved with correct RGB->BGR conversion for cv2.imwrite().
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    torch_save_dict = {
        "chunk_id": chunk_id,
        "keys": list(observation_frame.keys()),
    }

    if action_values is not None:
        torch_save_dict["action_values"] = clone_debug_value(action_values)

    if metadata is not None:
        torch_save_dict["metadata"] = metadata

    for key, value in observation_frame.items():
        safe_key = key.replace(".", "_").replace("/", "_")

        if hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
            torch_save_dict[key] = value.detach().cpu()
        else:
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype, np.number):
                torch_save_dict[key] = torch.as_tensor(arr)

        if key.startswith("observation.images."):
            img = np.asarray(arr)

            # Remove batch dim if present.
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]

            # CHW -> HWC
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))

            if img.dtype != np.uint8:
                img_float = img.astype(np.float32)
                if img_float.max() <= 1.0:
                    img_float = img_float * 255.0
                img = np.clip(img_float, 0, 255).astype(np.uint8)

            # IMPORTANT:
            # observation images are usually RGB, but cv2.imwrite expects BGR.
            if img.ndim == 3 and img.shape[-1] == 3:
                img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_to_save = img

            cv2.imwrite(
                str(debug_dir / f"{prefix}_{chunk_id:03d}_{safe_key}.png"),
                img_to_save,
            )

    torch.save(
        torch_save_dict,
        debug_dir / f"{prefix}_{chunk_id:03d}_observation_frame.pt",
    )


def clone_debug_observation_frame(observation_frame):
    """
    Cache a safe copy of observation_frame in memory.
    This prevents later camera / tensor mutation from changing cached chunks.
    """
    cached = {}

    for key, value in observation_frame.items():
        if hasattr(value, "detach"):
            cached[key] = value.detach().cpu().clone()
        elif isinstance(value, np.ndarray):
            cached[key] = value.copy()
        else:
            try:
                arr = np.asarray(value)
                if np.issubdtype(arr.dtype, np.number):
                    cached[key] = arr.copy()
                else:
                    cached[key] = value
            except Exception:
                cached[key] = value

    return cached

def clone_debug_value(value):
    """
    Safely clone tensors / numpy arrays / nested dicts / lists for debug caching.
    """
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()

    if isinstance(value, np.ndarray):
        return value.copy()

    if isinstance(value, dict):
        return {k: clone_debug_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return type(value)(clone_debug_value(v) for v in value)

    try:
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            return arr.copy()
    except Exception:
        pass

    return value

def extract_observation_state_for_action_trace(obs_processed, observation_frame=None):
    """
    Extract robot state for per-step action trace logging.
    Prefer observation_frame because build_dataset_frame(...) usually creates
    'observation.state'. Fall back to obs_processed or assembled joint positions.
    """
    if isinstance(observation_frame, dict):
        for key in ["observation.state", "state"]:
            if key in observation_frame:
                return key, observation_frame[key]

    if isinstance(obs_processed, dict):
        for key in ["observation.state", "state"]:
            if key in obs_processed:
                return key, obs_processed[key]

        joint_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        if all(k in obs_processed for k in joint_keys):
            state = np.asarray([obs_processed[k] for k in joint_keys], dtype=np.float32)
            return "assembled_joint_pos", state

    return None, None


def save_policy_action_trace(path: str | Path, trace_steps: list[dict[str, Any]], metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "num_steps": len(trace_steps),
        "trace": trace_steps,
        "metadata": metadata,
    }

    torch.save(payload, path)
    print(f"[ACTION_TRACE] Saved {len(trace_steps)} steps to: {path.resolve()}")


def clone_action_value_for_save(value):
    """
    Save action values in a torch-loadable, replay-safe format.
    """
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()

    if isinstance(value, np.ndarray):
        return value.copy()

    if isinstance(value, dict):
        return {k: clone_action_value_for_save(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return type(value)(clone_action_value_for_save(v) for v in value)

    try:
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            if arr.ndim == 0:
                return float(arr)
            return arr.copy()
    except Exception:
        pass

    return value


def extract_observation_state_for_replay_log(obs_processed, observation_frame=None):
    """
    Extract robot state for replay execution logging.

    Prefer observation_frame because build_dataset_frame(...) usually creates
    'observation.state' there. Fall back to obs_processed or manually assembled
    joint position vector.
    """
    if isinstance(observation_frame, dict):
        for key in ["observation.state", "state"]:
            if key in observation_frame:
                return key, observation_frame[key]

    if isinstance(obs_processed, dict):
        for key in ["observation.state", "state"]:
            if key in obs_processed:
                return key, obs_processed[key]

        joint_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        if all(k in obs_processed for k in joint_keys):
            state = np.asarray([obs_processed[k] for k in joint_keys], dtype=np.float32)
            return "assembled_joint_pos", state

    return None, None


def restore_action_value_for_send(value):
    """
    Convert saved torch/numpy/scalar values back to something robot.send_action can accept.
    """
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
        if arr.ndim == 0:
            return float(arr)
        return arr

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        return value.copy()

    if isinstance(value, dict):
        return {k: restore_action_value_for_send(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return type(value)(restore_action_value_for_send(v) for v in value)

    return value


def save_fixed_action_sequence(path: str | Path, actions: list[dict[str, Any]], fps: int, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "fps": fps,
        "num_steps": len(actions),
        "actions": actions,
        "metadata": metadata,
    }

    torch.save(payload, path)
    print(f"[FIXED_ACTION][CAPTURE] Saved {len(actions)} actions to: {path.resolve()}")


def load_fixed_action_sequence(path: str | Path) -> dict:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if "actions" not in payload:
        raise KeyError(f"No 'actions' key found in fixed action file: {path}")

    print(
        f"[FIXED_ACTION][REPLAY] Loaded {len(payload['actions'])} actions "
        f"from: {path.resolve()}"
    )

    return payload


def draw_attention_stats_overlay(
    img,
    cam_stats,
    contrib,
    debug_freq: bool = False,
    control_hz: float | None = None,
    attn_hz: float | None = None,
    attn_is_stale: bool = False,
    obs_ms: float | None = None,
    infer_ms: float | None = None,
    dataset_ms: float | None = None,
    rerun_ms: float | None = None,
):
    out = img.copy()

    def fmt_stats(name, stats):
        if stats is None:
            return f"{name}: none"

        return (
            f"{name}: "
            f"sum={stats.get('sum', 0.0):.4f}, "
            f"mean={stats.get('mean', 0.0):.6f}, "
            f"max={stats.get('max', 0.0):.6f}, "
            f"n={stats.get('tokens', 0)}"
        )

    lines = [
        fmt_stats("this_cam", cam_stats),
        # fmt_stats("image1", contrib.get("image_0")),
        # fmt_stats("image2", contrib.get("image_1")),
        # fmt_stats("image3", contrib.get("image_2")),
        fmt_stats("prompt", contrib.get("language")),
        fmt_stats("state", contrib.get("state")),
    ]

    if debug_freq:
        if control_hz is not None:
            lines.append(f"control_hz: {control_hz:.2f}")
        if attn_hz is not None:
            stale_suffix = " (cached)" if attn_is_stale else ""
            lines.append(f"attn_hz: {attn_hz:.2f}{stale_suffix}")
        if obs_ms is not None:
            lines.append(f"obs_ms: {obs_ms:.1f}")
        if infer_ms is not None:
            lines.append(f"infer_ms: {infer_ms:.1f}")
        if dataset_ms is not None:
            lines.append(f"dataset_ms: {dataset_ms:.1f}")
        if rerun_ms is not None:
            lines.append(f"rerun_ms: {rerun_ms:.1f}")

    x, y = 16, 28
    dy = 26
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness_bg = 4
    thickness_fg = 1

    for i, text in enumerate(lines):
        yy = y + i * dy
        cv2.putText(out, text, (x, yy), font, font_scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
        cv2.putText(out, text, (x, yy), font, font_scale, (255, 255, 255), thickness_fg, cv2.LINE_AA)

    return out

def get_policy_image_keys(policy, obs_processed):
    config_image_features = getattr(policy.config, "image_features", None)
    if isinstance(config_image_features, dict):
        configured_keys = list(config_image_features.keys())
    elif config_image_features is not None:
        configured_keys = list(config_image_features)
    else:
        configured_keys = []

    image_keys = [key for key in configured_keys if key in obs_processed]
    if image_keys:
        return image_keys

    return sorted(key for key in obs_processed if key.startswith("observation.images."))

# Helper for showing top token
def activation_steering_config_for_print(steering_neurons, alpha):
    return {
        layer_idx: {neuron_idx: alpha for neuron_idx in neuron_ids}
        for layer_idx, neuron_ids in steering_neurons.items()
    }


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    task_holder: dict | None = None, 
    display_data: bool = False,
    listener = None,
    debug_freq: bool = False,
    debug_run_dir: Path | None = None,
    pending_debug_observation_frames: list[tuple[int, dict[str, Any], Any]] | None = None,
    fixed_action: FixedActionConfig | None = None,
    fixed_action_replay_payload: dict | None = None,
    pending_policy_action_trace: list[dict[str, Any]] | None = None,
    action_trace: ActionTraceConfig | None = None,
    intervention: InterventionConfig | None = None,
):
    if debug_run_dir is None:
        debug_run_dir = make_debug_run_dir("debug_runs")
    else:
        debug_run_dir = Path(debug_run_dir)
        debug_run_dir.mkdir(parents=True, exist_ok=True)

    if pending_debug_observation_frames is None:
        pending_debug_observation_frames = []
    
    if pending_policy_action_trace is None:
        pending_policy_action_trace = []

    if action_trace is None:
        action_trace = ActionTraceConfig()
    
    if intervention is None:
        intervention = InterventionConfig()

    record_label = intervention.label

    trace_enabled = bool(action_trace.enabled)
    trace_step_idx = 0
    current_raw_chunk_id = None
    current_generated_chunk_index = -1
    
    if fixed_action is None:
        fixed_action = FixedActionConfig()

    fixed_mode = normalize_fixed_action_mode(fixed_action.mode)

    fixed_capture_actions: list[dict[str, Any]] = []
    fixed_capture_active = False
    fixed_capture_finished = False
    generated_chunk_index = -1

    fixed_replay_actions = None
    fixed_replay_step = 0
    fixed_execution_log: list[dict[str, Any]] = []

    if fixed_mode == "replay":
        if fixed_action_replay_payload is None:
            raise ValueError("fixed_action.mode='replay' but fixed_action_replay_payload is None.")

        fixed_replay_actions = fixed_action_replay_payload["actions"]

        print(
            f"[FIXED_ACTION][REPLAY] Ready to replay {len(fixed_replay_actions)} steps "
            f"from {fixed_action.path}"
        )

        if fixed_action.wait_before_replay:
            input(
                "\n[FIXED_ACTION][REPLAY] Put robot/object into the same reach/about-to-grasp "
                "start pose, then press Enter to execute the fixed action sequence..."
            )

    if policy:
        print("record loop, policy is not None:")
    else:
        print("record loop, policy is None:")
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so100_leader.SO100Leader
                        | so101_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        print("policy is not None and preprocessor is not None and postprocessor is not None")
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        policy_internal_debug_dir = Path(debug_run_dir) / "policy_internal"
        policy_internal_debug_dir.mkdir(parents=True, exist_ok=True)
        policy._debug_run_dir = policy_internal_debug_dir

        # --- ⚡ FULL-MODEL VLA STEERING SETUP ⚡ ---
        print("\n--- ⚡ FULL-MODEL VLA STEERING SETUP ⚡ ---")

#########################################################################################

        # --- Paper-alike FFN activation steering setup ---
        # Important:
        #   alpha = 0.0 with NO hook  -> no-steering baseline
        #   alpha = 0.0 with hook     -> activation ablation
        #   alpha != 0.0 with hook    -> activation steering

        intervention_name = intervention.name
        alpha = float(intervention.alpha)
        enable_steering = bool(intervention.enable_steering)

        print(
            f"[INTERVENTION] label={intervention.label} "
            f"name={intervention_name} alpha={alpha} enable_steering={enable_steering}"
        )

        semantic_neuron_sets = {

            # lerobot_reord_top_token.py
            "low_transport_paper": {
                1: [1222],
                3: [2003],
                5: [1877,1904],
                10: [2349],
                13: [1744],
            },
            "high_transport_paper": {
                2: [826],
                3: [369],
                5: [2102],
                7: [1151],
                9:[2554],
                13: [414],
            },

            # # eef Z (python src/lerobot/scripts/physical_neuron_picking_test_Z.py)
            # "high_transport": {
            #     8:[333],
            #     9:[327,163,756],
            #     12:[45,902,54],
            #     13:[93,106],
            #     14:[426],
            # },
            # "low_transport": {
            #     8:[640,607],
            #     9:[396],
            #     10:[87],
            #     11:[43,546],
            #     12:[544,652,781,87],
            # },

            ### Constrative
            # physical_neuron_picking.py
            "high_transport": {
                0: [1293],
                1: [1050],
                3: [2259],
                4: [1183],
                7: [295],
                11: [1115,1595],
                13: [431],
                14: [736,805],
            },
            "low_transport": {
                3: [962],
                4: [1627],
                6: [587],
                7: [1007],
                9: [149],
                11: [1066],
                12:[629,1164],
                14: [423],
                15:[1886],
            },

            ### Constrative(dataset from ethanCSL/svla_koch_pick_n_place_vla_steering_height_experiment_setup)
            "high_transport_clean_dataset": {
                6: [1816],
                9: [1596],
                11: [665,1273],
                12: [1937],
                13: [489,500,1034,1261],
                15: [1964],
            },
            "low_transport_clean_dataset": {
                3: [1556],
                6: [1558],
                8: [1034,2114],
                10: [454,2135],
                11: [188,988,1115],
                14: [1836],
            },


            # physical_neuron_finding.py
            "green": {0: [1930, 491, 2532, 1677, 930, 1286, 1429], 1: [805, 1596], 2: [2033], 4: [1854], 5: [416], 6: [1767], 7: [6, 2055], 8: [1278], 10: [997], 14: [156], 15: [848, 2261]},

            "red" : {0: [1461, 2168, 1728, 1996, 1435], 2: [702, 672], 4: [1262], 6: [1633], 7: [2415, 1466, 934, 2125, 188], 8: [508], 9: [847], 11: [1022], 12: [1396], 14: [1924, 246]},



            #####################
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
            # "green": {
            #     1: [798],
            #     7: [230],
            #     8: [1527],
            #     9: [863],
            # },
            # "red" : {
            #     1: [67, 1986],
            #     4: [688],
            #     8: [216],
            #     10: [2283],
            #     13: [268],
            # },
            "right" : {
                4: [1897, 1400],
                10: [1257],
                13: [1122],
                14: [479,1650]
            },
            "left" : {
                3: [583],
                4: [1936],
                5: [1872],
                6: [2374],
                9: [1941],
                11: [1367],
            }
        }

        if hasattr(policy, "clear_activation_steering"):
            policy.clear_activation_steering()

        if (not enable_steering) or alpha == 0.0 or intervention_name == "none":
            print(
                f"[BASELINE / NO STEERING] label={intervention.label}, "
                f"name={intervention_name}, alpha={alpha}, enable_steering={enable_steering}"
            )

            # Optional: keep debug hook with zero effect if a valid neuron set is provided.
            if (
                intervention_name != "none"
                and intervention_name in semantic_neuron_sets
                and hasattr(policy, "set_activation_steering")
            ):
                selected_neurons = semantic_neuron_sets[intervention_name]
                policy.set_activation_steering(
                    steering_neurons=selected_neurons,
                    alpha=0.0,
                    record_debug=True,
                    top_k_runtime=10,
                    enable_steering=False,
                )

        else:
            if not hasattr(policy, "set_activation_steering"):
                raise AttributeError(
                    "Policy does not have set_activation_steering(). "
                    "Make sure modeling_smolvla.py contains the hook-based steering code."
                )

            if intervention_name not in semantic_neuron_sets:
                raise ValueError(
                    f"Unknown intervention.name={intervention_name}. "
                    f"Available: {sorted(semantic_neuron_sets.keys())}"
                )

            selected_neurons = semantic_neuron_sets[intervention_name]

            _, metadata, _ = extract_semantic_embeddings_from_policy(
                policy,
                top_k_tokens=5,
                device=get_safe_torch_device(policy.config.device),
            )

            print_steered_neurons_info(
                metadata,
                activation_steering_config_for_print(selected_neurons, alpha),
                title=f"[SELECTED BEFORE STEERING] {intervention_name}: Value-Vector Top Tokens",
            )

            policy.set_activation_steering(
                steering_neurons=selected_neurons,
                alpha=alpha,
                record_debug=True,
            )

        print("--------------------------------\n")

#########################################################################################

#########################################################################################
# Below is the CAA-alike method to steer VLA
# Reference: https://arxiv.org/html/2308.10248v4(Activation Addition: Steering Language Models Without Optimization)
# Reference2: https://aclanthology.org/2024.acl-long.828.pdf(Steering Llama 2 via Contrastive Activation Addition)

        # Ting:
        # ⚡ 實時 CAA 轉向設定 ⚡
        # target_layer = 14
        # alpha = -3.0  # 💡 轉向強度調整：正值（例如 +3.0）會引導模型做出 High 的動作；負值（-3.0）引導做出 Low 的動作;0為baseline
        # v_steer_path = Path("steering_vector_L10_caa.pt")

        # # 💡 防重複註冊機制：因為 record_loop 在多個 Episode 之間會被重複呼叫，
        # # 我們用一個自訂屬性 _caa_hook_registered 確保整個 Evaluation 過程只註冊一次 Hook，避免記憶體洩漏與強度疊加。
        # if not getattr(policy, "_caa_hook_registered", False) and v_steer_path.exists():
        #     print(f"\n[⚡ CAA ONLINE STEERING] 偵測到轉向向量，正在注入 Layer {target_layer}...")
            
        #     # 1. 載入轉向向量
        #     v_steer_base = torch.load(v_steer_path) # Shape: [1, 720]
            
        #     # 2. 定義實時相加的 Hook 函式
        #     def caa_steering_hook(module, inputs, outputs):
        #         # outputs 通常是 hidden_states，或者是一個包含 hidden_states 的 tuple
        #         if isinstance(outputs, tuple):
        #             h = outputs[0]
        #             # 透過 PyTorch 廣播機制 (Broadcasting)，[1, 720] 會自動對齊並加到 [B, N, 720] 的每一幀/每一個 Token 上
        #             # 同時動態將設備與精度 (FP16/BF16) 對齊當前 hidden_states
        #             h_steered = h + alpha * v_steer_base.to(device=h.device, dtype=h.dtype)
        #             return (h_steered,) + outputs[1:]
        #         else:
        #             return outputs + alpha * v_steer_base.to(device=outputs.device, dtype=outputs.dtype)

        #     # 3. 定位並註冊到目標層
        #     try:
        #         target_module = policy.model.vlm_with_expert.lm_expert.layers[target_layer]
        #         target_module.register_forward_hook(caa_steering_hook)
        #         policy._caa_hook_registered = True
        #         print(f"[✓] 成功於 Layer {target_layer} 注入 CAA 實時轉向 Hook (alpha={alpha})")
        #     except Exception as e:
        #         print(f"[X] 注入 CAA 轉向失敗，錯誤訊息: {e}")
                
        # elif not v_steer_path.exists():
        #     print(f"[!] 警告: 找不到轉向向量檔案 {v_steer_path}，本次評估將以 Baseline (未轉向) 執行。")
        ############################################################################################
        
        # intervention_name = "height_high"
        # alpha = 3.0 #3.0

        # height_steering_deltas = {
        #     1: {
        #         863:  -0.025490,
        #         960:  +0.007833,
        #         1248: +0.006831,
        #         565:  +0.007804,
        #     }
        #     # 1: {
        #     #     863:  -1.000,
        #     #     960:  +0.307,
        #     #     1248: +0.268,
        #     #     565:  +0.306,
        #     # }
        # }

        # if intervention_name == "height_high":
        #     signed_alpha = +alpha
        # elif intervention_name == "height_low":
        #     signed_alpha = -alpha
        # else:
        #     raise ValueError(f"Unknown intervention_name: {intervention_name}")

        # if hasattr(policy, "clear_activation_steering"):
        #     policy.clear_activation_steering()

        # if signed_alpha == 0.0:
        #     print("[BASELINE] alpha=0.0: no signed activation steering.")
        # else:
        #     if not hasattr(policy, "set_signed_activation_steering"):
        #         raise AttributeError(
        #             "Policy does not have set_signed_activation_steering(). "
        #             "Add the signed steering hook to modeling_smolvla.py first."
        #         )

        #     policy.set_signed_activation_steering(
        #         steering_deltas=height_steering_deltas,
        #         alpha=signed_alpha,
        #         record_debug=True,
        #         enable_steering=True,
        #     )

#########################################################################################

    timestamp = 0
    start_episode_t = time.perf_counter()
    last_control_tick_t = None
    smoothed_control_hz = None
    last_new_attn_t = None
    smoothed_attn_hz = None

    smoothed_obs_ms = None
    smoothed_infer_ms = None
    smoothed_dataset_ms = None
    smoothed_rerun_ms = None
    printed_activation_debug = False
    exited_by_key = False

    while timestamp < control_time_s:

        attn = None
        token_layout = None
        num_img_tokens = None
        act_processed_policy = None
        act_processed_teleop = None
        raw_policy_action_values = None

        # Keyboard 't' to switch task in real-time!
        # USe events["change_task"],events["in_task_input"]
        if events["change_task"]  == True:
            print("enter if confition")
            events["change_task"] = False
            events["in_task_input"] = True
            try:
                print("\n=== TASK SWITCH MODE ===")
                print("Type new task prompt and press Enter:")
                raw = input(">> ")
            finally:
                events["in_task_input"] = False
            
            raw = _ANSI_ESCAPE_RE.sub("", raw)
            raw = _CONTROL_RE.sub("", raw)
            new_task = " ".join(raw.strip().split())

            if new_task.startswith("t"):
                new_task = new_task[1:].lstrip()

            #print("[DEBUG][SANITIZED] task repr:", repr(new_task), "len:", len(new_task))

            if len(new_task) > 0:
                task_holder["text"] = new_task
                print(f"[INFO] Task updated to: {new_task}")
                events["exit_early"] = True
                events["rerecord_episode"] = True
            else:
                print("[WARN] Empty input, task unchanged")

        attn = None
        num_img_tokens = None
        act_processed_policy = None
        act_processed_teleop = None

        start_loop_t = time.perf_counter()

        if last_control_tick_t is not None:
            control_dt_s = max(start_loop_t - last_control_tick_t, 1e-6)
            instant_control_hz = 1.0 / control_dt_s
            if smoothed_control_hz is None:
                smoothed_control_hz = instant_control_hz
            else:
                smoothed_control_hz = 0.8 * smoothed_control_hz + 0.2 * instant_control_hz
        last_control_tick_t = start_loop_t

        fresh_attn_this_step = False

        if events["exit_early"]:
            exited_by_key = True
            events["exit_early"] = False
            break

        # Get robot observation
        obs_t0 = time.perf_counter()
        obs = robot.get_observation()
        obs_ms = (time.perf_counter() - obs_t0) * 1000.0
        if smoothed_obs_ms is None:
            smoothed_obs_ms = obs_ms
        else:
            smoothed_obs_ms = 0.8 * smoothed_obs_ms + 0.2 * obs_ms

        top_cam_key = "top" 
        eval_cam2 = "camera4"
        
        if top_cam_key in obs:

            img = obs[top_cam_key] 
            
            if img.shape[0] == 3:
                img_hwc = np.transpose(img, (1, 2, 0))
            else:
                img_hwc = img # 已經是 HWC 格式

            h, w, c = img_hwc.shape

            # img_hwc[y1:y2, x1:x2]
            # Plz refer to lerobot_record_pictureXY.py to find y1:y2,x1:x2
            # Remember to deactivate lerobot,to base env
            img_cropped = img_hwc[19:217, 285:546]

            # 3. Resize to 640x480
            img_resized_hwc = cv2.resize(img_cropped, (640, 480), interpolation=cv2.INTER_LINEAR)

            # Keep HWC for downstream preprocessors / policy image processor
            obs[top_cam_key] = np.ascontiguousarray(img_resized_hwc)

        elif eval_cam2 in obs:

            img = obs[eval_cam2] 
            
            if img.shape[0] == 3:
                img_hwc = np.transpose(img, (1, 2, 0))
            else:
                img_hwc = img 

            h, w, c = img_hwc.shape

            # x1:x2=19 to 285
            # y1:y2=217 to 546
            img_cropped = img_hwc[19:217, 285:546]
            img_resized_hwc1 = cv2.resize(img_cropped, (640, 480), interpolation=cv2.INTER_LINEAR)

            # Keep HWC for downstream preprocessors / policy image processor
            obs[eval_cam2] = np.ascontiguousarray(img_resized_hwc1)

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from fixed replay, policy, or teleop
        fixed_replay_this_step = False

        if fixed_mode == "replay":
            if fixed_replay_step >= len(fixed_replay_actions):
                print("[FIXED_ACTION][REPLAY] Finished fixed action sequence.")
                exited_by_key = True
                break

            robot_action_to_send = restore_action_value_for_send(
                fixed_replay_actions[fixed_replay_step]
            )

            action_values = robot_action_to_send
            fixed_replay_this_step = True

            print(
                f"[FIXED_ACTION][REPLAY] step "
                f"{fixed_replay_step + 1}/{len(fixed_replay_actions)} "
                f"slowdown={fixed_action.replay_slowdown}"
            )

            fixed_replay_step += 1

        elif policy is not None and preprocessor is not None and postprocessor is not None:
            
            # Check the prompt, to see if there any incorrect char
            # print(
            #     "[DEBUG] task repr:",
            #     repr(task_holder["text"]),
            #     "len:",
            #     len(task_holder["text"]),
            # )
            model = policy.model.vlm_with_expert
            model.record_attn = True
            model.debug_attn = True
            model.attention_mode = "cross_attn"

            if not hasattr(model, "attn_records") or model.attn_records is None:
                model.attn_records = {}
            if not hasattr(model, "_last_vis_attn"):
                model._last_vis_attn = None
            if not hasattr(model, "_last_vis_num_img_tokens"):
                model._last_vis_num_img_tokens = None
            if not hasattr(model, "_last_vis_token_layout"):
                model._last_vis_token_layout = None
            
            #print("attn_records keys before predict:", model.attn_records.keys())

            infer_t0 = time.perf_counter()
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task = task_holder["text"],
                robot_type=robot.robot_type,
            )

            raw_policy_action_values = clone_action_value_for_save(action_values)

            if getattr(policy, "_debug_just_generated_chunk", False):
                chunk_id = getattr(policy, "_debug_last_chunk_id", -1)
                generated_chunk_index += 1

                current_raw_chunk_id = chunk_id
                current_generated_chunk_index = generated_chunk_index

                # -----------------------------
                # Fixed-action capture trigger
                # -----------------------------
                if fixed_mode == "capture":
                    if generated_chunk_index == fixed_action.start_chunk:
                        fixed_capture_active = True
                        fixed_capture_finished = False
                        fixed_capture_actions.clear()
                        print(
                            f"[FIXED_ACTION][CAPTURE] START at generated chunk "
                            f"{generated_chunk_index} raw_id={chunk_id}"
                        )

                    # Stop BEFORE capturing actions from stop_chunk.
                    if (
                        fixed_capture_active
                        and generated_chunk_index >= fixed_action.stop_chunk
                    ):
                        fixed_capture_active = False
                        fixed_capture_finished = True
                        print(
                            f"[FIXED_ACTION][CAPTURE] STOP before generated chunk "
                            f"{generated_chunk_index} raw_id={chunk_id}. "
                            f"Captured steps: {len(fixed_capture_actions)}"
                        )

                pending_debug_observation_frames.append(
                    (
                        chunk_id,
                        clone_debug_observation_frame(observation_frame),
                        clone_debug_value(action_values),
                    )
                )

                print(
                    f"[DEBUG] Cached debug chunk {chunk_id}. "
                    f"save_idx={generated_chunk_index}. "
                    f"Pending chunks: {len(pending_debug_observation_frames)}"
                )

                policy._debug_just_generated_chunk = False

            ## Show activation steering (Before and After)
            if (
                not printed_activation_debug
                and hasattr(policy, "print_activation_steering_debug")
            ):
                #policy.print_activation_steering_debug(latest_only=True)

                if hasattr(policy, "reset_activation_steering_debug_records"):
                    policy.reset_activation_steering_debug_records()

                printed_activation_debug = True
                            

            infer_ms = (time.perf_counter() - infer_t0) * 1000.0
            if smoothed_infer_ms is None:
                smoothed_infer_ms = infer_ms
            else:
                smoothed_infer_ms = 0.8 * smoothed_infer_ms + 0.2 * infer_ms

            feature_spec = dataset.features if dataset is not None else robot.action_features
            act_processed_policy: RobotAction = make_robot_action(action_values, feature_spec)

            model = policy.model.vlm_with_expert
            if display_data and hasattr(model, "attn_records"):
                layer_ids = [k[0] for k in model.attn_records.keys() if k[1] == "expert_cross"]
                #print("cross-attn layer count after predict:", len(layer_ids))

                attn = None
                num_img_tokens = getattr(model, "_debug_num_img_tokens", None)

                if len(layer_ids) > 0:
                    final_layer = max(layer_ids)
                    attn_list = model.attn_records.get((final_layer, "expert_cross"), [])
                    print("final layer attn records:", len(attn_list))

                    if len(attn_list) > 0:
                        attn = attn_list[-1]  # [B, heads, Q, K]
                        token_layout = getattr(model, "_last_prefix_token_layout", None)
                        model._last_vis_attn = attn
                        model._last_vis_token_layout = token_layout
                        fresh_attn_this_step = True
                        now_attn_t = time.perf_counter()
                        if last_new_attn_t is not None:
                            attn_dt_s = max(now_attn_t - last_new_attn_t, 1e-6)
                            instant_attn_hz = 1.0 / attn_dt_s
                            if smoothed_attn_hz is None:
                                smoothed_attn_hz = instant_attn_hz
                            else:
                                smoothed_attn_hz = 0.8 * smoothed_attn_hz + 0.2 * instant_attn_hz
                        last_new_attn_t = now_attn_t
                        if num_img_tokens is not None:
                            model._last_vis_num_img_tokens = num_img_tokens

                model.attn_records = {}

            if attn is None:
                attn = getattr(model, "_last_vis_attn", None)
                num_img_tokens = getattr(model, "_last_vis_num_img_tokens", None)
                token_layout = getattr(model, "_last_vis_token_layout", None)
                # if attn is None:
                #     print("[DEBUG] no cross-attn recorded yet (likely action queue reused cached actions)")
                # else:
                #     print("[DEBUG] no new cross-attn this step; reusing cached attention for visualization")

            if display_data and attn is not None:
                image_obs_keys = get_policy_image_keys(policy, observation_frame)

                if num_img_tokens is None:
                    try:
                        device = get_safe_torch_device(policy.config.device)
                        image_batch = {}
                        for image_key in image_obs_keys:
                            x = torch.as_tensor(observation_frame[image_key], device=device)
                            if x.ndim == 3:
                                x = x.unsqueeze(0)
                            image_batch[image_key] = x

                        images_list, _ = policy.prepare_images(image_batch)
                        img_emb0 = policy.model.vlm_with_expert.embed_image(images_list[0])
                        num_img_tokens = int(img_emb0.shape[1])

                        model._debug_num_img_tokens = num_img_tokens
                        model._last_vis_num_img_tokens = num_img_tokens
                        print("[DEBUG] recovered num_img_tokens:", num_img_tokens)
                    except Exception as e:
                        print("[DEBUG] failed to recover num_img_tokens:", repr(e))

                attn_matrix = attn.mean(dim=1)[0]  # [Q, K]
                image_attn_list, contrib = extract_cross_attention_maps(attn_matrix, token_layout)

                if len(image_attn_list) == 0:
                    print("[DEBUG] no heatmap slices generated, so rr.log(attention/...) was not called")

                for item in image_attn_list:
                    image_index = item["image_index"]
                    heat_1d = item["heat_1d"]
                    cam_stats = item["stats"]

                    if image_index >= len(image_obs_keys):
                        continue

                    image_key = image_obs_keys[image_index]
                    cam_idx = image_index + 1

                    img_hwc = to_hwc_uint8(observation_frame[image_key])
                    mask = process_heatmap(heat_1d, original_size=img_hwc.shape[:2])
                    if mask is None:
                        print(f"[DEBUG] mask is None for {image_key}, skip rr.log")
                        continue

                    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    vis = cv2.addWeighted(img_hwc, 0.6, heatmap_rgb, 0.4, 0)
                    vis = draw_attention_stats_overlay(
                        vis,
                        cam_stats,
                        contrib,
                        debug_freq=debug_freq,
                        control_hz=smoothed_control_hz,
                        attn_hz=smoothed_attn_hz,
                        attn_is_stale=not fresh_attn_this_step,
                        obs_ms=smoothed_obs_ms,
                        infer_ms=smoothed_infer_ms,
                        dataset_ms=smoothed_dataset_ms,
                        rerun_ms=smoothed_rerun_ms,
                    )

                    rr.log(f"attention/cam{cam_idx}", rr.Image(vis))
                    #print(f"[DEBUG] logged attention/cam{cam_idx}")

            feature_spec = dataset.features if dataset is not None else robot.action_features
            act_processed_policy: RobotAction = make_robot_action(action_values, feature_spec)

        # always enter this loop in recording,and only enter it when resetting(left key is pressed) in eval.
        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        # Only trigger when it is bimanual robot
        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Applies a pipeline to the action, default is IdentityProcessor.
        # In fixed replay mode, robot_action_to_send is already the saved post-processor command,
        # so do NOT pass it through robot_action_processor again.
        if not fixed_replay_this_step:
            if policy is not None and act_processed_policy is not None:
                action_values = act_processed_policy
                robot_action_to_send = robot_action_processor((act_processed_policy, obs))
            else:
                action_values = act_processed_teleop
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        # ------------------------------------------------------------
        # Normal policy per-step action trace
        # ------------------------------------------------------------
        if (
            trace_enabled
            and fixed_mode == "off"
            and policy is not None
        ):
            state_key, observation_state = extract_observation_state_for_action_trace(
                obs_processed=obs_processed,
                observation_frame=observation_frame if "observation_frame" in locals() else None,
            )

            pending_policy_action_trace.append(
                {
                    "trace_step": trace_step_idx,
                    "time_s": time.perf_counter(),
                    "chunk_index": current_generated_chunk_index,
                    "raw_chunk_id": current_raw_chunk_id,
                    "state_key": state_key,
                    "observation_state": clone_action_value_for_save(
                        observation_state
                    ) if observation_state is not None else None,

                    # Policy output before robot mapping.
                    "policy_action_values": clone_action_value_for_save(raw_policy_action_values),

                    # Action after make_robot_action(...).
                    "act_processed_policy": clone_action_value_for_save(
                        act_processed_policy
                    ) if act_processed_policy is not None else None,

                    # Command before robot.send_action(...).
                    "robot_action_to_send": clone_action_value_for_save(robot_action_to_send),

                    # Command actually returned by robot backend.
                    "sent_action": clone_action_value_for_save(_sent_action),

                    "task": task_holder["text"] if task_holder is not None else None,
                }
            )

            trace_step_idx += 1

        # ------------------------------------------------------------
        # Experiment A capture/replay logging
        # ------------------------------------------------------------
        if fixed_mode == "capture" and fixed_capture_active:
            # Prefer the action returned by robot.send_action(...), because that may include
            # clipping / conversion actually applied by the robot backend.
            if isinstance(_sent_action, dict) and len(_sent_action) > 0:
                action_to_store = _sent_action
            else:
                action_to_store = robot_action_to_send

            fixed_capture_actions.append(clone_action_value_for_save(action_to_store))

            print(
                f"[FIXED_ACTION][CAPTURE] captured step "
                f"{len(fixed_capture_actions)}/{fixed_action.max_steps}"
            )

            if len(fixed_capture_actions) >= fixed_action.max_steps:
                fixed_capture_active = False
                fixed_capture_finished = True
                print(
                    f"[FIXED_ACTION][CAPTURE] Reached max_steps={fixed_action.max_steps}. "
                    "Stopping capture."
                )

        if fixed_mode == "replay":
            state_key, observation_state = extract_observation_state_for_replay_log(
                obs_processed=obs_processed,
                observation_frame=observation_frame if "observation_frame" in locals() else None,
            )

            fixed_execution_log.append(
                {
                    "step": fixed_replay_step - 1,
                    "time_s": time.perf_counter(),
                    "state_key": state_key,
                    "observation_state": clone_action_value_for_save(
                        observation_state
                    ) if observation_state is not None else None,
                    "robot_action_to_send": clone_action_value_for_save(robot_action_to_send),
                    "sent_action": clone_action_value_for_save(_sent_action),
                }
            )

        # Write to dataset
        if dataset is not None:
            dataset_t0 = time.perf_counter()
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": task_holder["text"]}
            dataset.add_frame(frame)
            dataset_ms = (time.perf_counter() - dataset_t0) * 1000.0
            if smoothed_dataset_ms is None:
                smoothed_dataset_ms = dataset_ms
            else:
                smoothed_dataset_ms = 0.8 * smoothed_dataset_ms + 0.2 * dataset_ms

        if display_data:
            rerun_t0 = time.perf_counter()
            log_rerun_data(observation=obs_processed, action=action_values)
            rerun_ms = (time.perf_counter() - rerun_t0) * 1000.0
            if smoothed_rerun_ms is None:
                smoothed_rerun_ms = rerun_ms
            else:
                smoothed_rerun_ms = 0.8 * smoothed_rerun_ms + 0.2 * rerun_ms

        dt_s = time.perf_counter() - start_loop_t

        if fixed_mode == "replay":
            target_dt_s = (1 / fps) * float(fixed_action.replay_slowdown)
        else:
            target_dt_s = 1 / fps

        precise_sleep(target_dt_s - dt_s)

        timestamp = time.perf_counter() - start_episode_t
    
    if fixed_mode == "capture" and len(fixed_capture_actions) > 0:
        save_fixed_action_sequence(
            path=fixed_action.path,
            actions=fixed_capture_actions,
            fps=fps,
            metadata={
                "mode": "capture",
                "start_chunk": fixed_action.start_chunk,
                "stop_chunk": fixed_action.stop_chunk,
                "max_steps": fixed_action.max_steps,
                "num_steps": len(fixed_capture_actions),
                "debug_run_dir": str(debug_run_dir),
                "note": "Actions captured after robot_action_processor and robot.send_action.",
            },
        )

    if fixed_mode == "replay" and len(fixed_execution_log) > 0:
        replay_log_path = Path(debug_run_dir) / "fixed_action_replay_execution_log.pt"
        torch.save(
            {
                "fixed_action_path": fixed_action.path,
                "num_steps": len(fixed_execution_log),
                "execution_log": fixed_execution_log,
            },
            replay_log_path,
        )
        print(f"[FIXED_ACTION][REPLAY] Saved execution log to: {replay_log_path.resolve()}")
        print()

    return exited_by_key


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:

    # Check the prompt, to see if there any incorrect char
    current_task = {"text": cfg.dataset.single_task}
    debug_run_dir = make_debug_run_dir("debug_runs")

    # print(
    #     "[DEBUG][CLI] task repr:",
    #     repr(cfg.dataset.single_task),
    #     "len:",
    #     len(cfg.dataset.single_task),
    # )

    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Create empty dataset or load existing saved episodes
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )

        # Load pretrained policy
        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor = None
        postprocessor = None
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        fixed_action_replay_payload = None
    
        if normalize_fixed_action_mode(cfg.fixed_action.mode) == "replay":
            fixed_action_replay_payload = load_fixed_action_sequence(cfg.fixed_action.path)

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                
                # Add reset in the front to keep same behaviour for right,left esc,and t key
                if policy is not None and preprocessor is not None and postprocessor is not None:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()

                pending_debug_observation_frames: list[tuple[int, dict[str, Any], Any]] = []
                pending_policy_action_trace: list[dict[str, Any]] = []

                episode_debug_dir = debug_run_dir / f"episode_{dataset.num_episodes:06d}"
                episode_debug_dir.mkdir(parents=True, exist_ok=True)

                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                episode_exited_by_key = record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    task_holder=current_task,
                    display_data=cfg.display_data,
                    debug_freq=cfg.debug_freq,
                    listener=listener,
                    debug_run_dir=episode_debug_dir,
                    pending_debug_observation_frames=pending_debug_observation_frames,
                    fixed_action=cfg.fixed_action,
                    fixed_action_replay_payload=fixed_action_replay_payload,
                    pending_policy_action_trace=pending_policy_action_trace,
                    action_trace=cfg.action_trace,
                    intervention=cfg.intervention,
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    if policy:
                        print("before record loop,policy is not None")
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        task_holder=current_task, 
                        display_data=cfg.display_data,
                        debug_freq=cfg.debug_freq,
                        listener=listener,
                        debug_run_dir=episode_debug_dir,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    if policy:
                        print("record,policy is not None")

                    # Do not save debug chunks for a discarded / re-recorded rollout.
                    pending_debug_observation_frames.clear()
                    continue

                # Save debug chunks only when the recording episode ended by key press,
                # and the episode is not discarded by left key / task switch / stop.
                if (
                    episode_exited_by_key
                    and not events["stop_recording"]
                    and len(pending_debug_observation_frames) > 0
                ):
                    episode_debug_dir = debug_run_dir / f"episode_{dataset.num_episodes:06d}"

                    print(
                        f"[DEBUG] Right key accepted rollout. "
                        f"Saving {len(pending_debug_observation_frames)} debug chunks to: "
                        f"{episode_debug_dir.resolve()}"
                    )

                    for save_idx, (chunk_id, cached_observation_frame, cached_action_values) in enumerate(
                        pending_debug_observation_frames
                    ):
                        save_debug_observation_frame(
                            observation_frame=cached_observation_frame,
                            chunk_id=save_idx,
                            debug_dir=episode_debug_dir,
                            prefix=f"debug_chunk_rawid_{chunk_id}",
                            action_values=cached_action_values,
                            metadata={
                                "raw_chunk_id": chunk_id,
                                "save_idx": save_idx,
                                "task": current_task["text"],
                                "intervention": cfg.intervention.label,
                            },
                        )

                    pending_debug_observation_frames.clear()
                else:
                    print(
                        f"[DEBUG] Not saving debug chunks. "
                        f"episode_exited_by_key={episode_exited_by_key}, "
                        f"stop_recording={events['stop_recording']}, "
                        f"pending_chunks={len(pending_debug_observation_frames)}"
                    )

                    pending_debug_observation_frames.clear()


                # Save per-step action trace only for accepted recording episodes.
                if (
                    episode_exited_by_key
                    and not events["stop_recording"]
                    and cfg.action_trace.enabled
                    and len(pending_policy_action_trace) > 0
                ):
                    action_trace_path = episode_debug_dir / cfg.action_trace.save_name

                    save_policy_action_trace(
                        path=action_trace_path,
                        trace_steps=pending_policy_action_trace,
                        metadata={
                            "task": current_task["text"],
                            "intervention": cfg.intervention.label,
                            "episode_debug_dir": str(episode_debug_dir),
                            "num_steps": len(pending_policy_action_trace),
                            "note": (
                                "Per-control-step trace from normal policy rollout. "
                                "Includes policy_action_values, act_processed_policy, "
                                "robot_action_to_send, sent_action, observation_state, chunk_index."
                            ),
                        },
                    )

                    pending_policy_action_trace.clear()
                else:
                    pending_policy_action_trace.clear()

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if dataset is not None and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()