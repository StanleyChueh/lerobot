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

import logging
import math
import time
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
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

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

def extract_cross_attention_maps(attn_matrix, token_layout):
    if attn_matrix is None or token_layout is None:
        return [], {}

    mean_action_attn = attn_matrix.mean(dim=0)   # [K]
    image_maps = []
    contrib = {}

    total = mean_action_attn.sum().item() + 1e-8

    for seg in token_layout:
        start, end = seg["start"], seg["end"]
        seg_attn = mean_action_attn[start:end]
        seg_sum = seg_attn.sum().item()
        ratio = seg_sum / total

        if seg["type"] == "image":
            image_maps.append({
                "image_index": seg["image_index"],
                "heat_1d": seg_attn,
                "ratio": ratio,
            })

        key = seg["type"]
        if seg["type"] == "image":
            key = f'image_{seg["image_index"]}'
        contrib[key] = contrib.get(key, 0.0) + ratio

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

def draw_ratio_overlay(img, cam_ratio, contrib):
    out = img.copy()

    lines = [
        f"this_cam: {cam_ratio * 100:.1f}%",
        f"image1: {contrib.get('image_0', 0.0) * 100:.1f}%",
        f"image2: {contrib.get('image_1', 0.0) * 100:.1f}%",
        f"prompt: {contrib.get('language', 0.0) * 100:.1f}%",
        f"state: {contrib.get('state', 0.0) * 100:.1f}%",
    ]

    x, y = 16, 28
    dy = 26
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness_bg = 4
    thickness_fg = 2

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
    listener = None
):
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

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:

        attn = None
        token_layout = None
        num_img_tokens = None
        act_processed_policy = None
        act_processed_teleop = None

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

            print("[DEBUG][SANITIZED] task repr:", repr(new_task), "len:", len(new_task))

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

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

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

        # Get action from either policy or teleop
        if policy is not None and preprocessor is not None and postprocessor is not None:
            
            # Check the prompt, to see if there any incorrect char
            print(
                "[DEBUG] task repr:",
                repr(task_holder["text"]),
                "len:",
                len(task_holder["text"]),
            )
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

            print("attn_records keys before predict:", model.attn_records.keys())

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

            feature_spec = dataset.features if dataset is not None else robot.action_features
            act_processed_policy: RobotAction = make_robot_action(action_values, feature_spec)

            model = policy.model.vlm_with_expert
            if display_data and hasattr(model, "attn_records"):
                layer_ids = [k[0] for k in model.attn_records.keys() if k[1] == "expert_cross"]
                print("cross-attn layer count after predict:", len(layer_ids))

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
                        if num_img_tokens is not None:
                            model._last_vis_num_img_tokens = num_img_tokens

                model.attn_records = {}

            if attn is None:
                attn = getattr(model, "_last_vis_attn", None)
                num_img_tokens = getattr(model, "_last_vis_num_img_tokens", None)
                token_layout = getattr(model, "_last_vis_token_layout", None)
                if attn is None:
                    print("[DEBUG] no cross-attn recorded yet (likely action queue reused cached actions)")
                else:
                    print("[DEBUG] no new cross-attn this step; reusing cached attention for visualization")

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

            print("[DEBUG] image_obs_keys:", image_obs_keys)
            print("[DEBUG] num_img_tokens:", num_img_tokens)
            print("[DEBUG] len(image_attn_list):", len(image_attn_list))

            if len(image_attn_list) == 0:
                print("[DEBUG] no heatmap slices generated, so rr.log(attention/...) was not called")

            for item in image_attn_list:
                image_index = item["image_index"]
                heat_1d = item["heat_1d"]
                ratio = item["ratio"]

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
                vis = draw_ratio_overlay(vis, ratio, contrib)

                rr.log(f"attention/cam{cam_idx}", rr.Image(vis))
                print(f"[DEBUG] logged attention/cam{cam_idx}")

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

        # Applies a pipeline to the action, default is IdentityProcessor
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

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": task_holder["text"]}
            dataset.add_frame(frame)

        if display_data:

            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:

    # Check the prompt, to see if there any incorrect char
    current_task = {"text": cfg.dataset.single_task}

    print(
        "[DEBUG][CLI] task repr:",
        repr(cfg.dataset.single_task),
        "len:",
        len(cfg.dataset.single_task),
    )

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

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                
                # Add reset in the front to keep same behaviour for right,left esc,and t key
                if policy is not None and preprocessor is not None and postprocessor is not None:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()

                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
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
                    listener=listener,
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
                        listener=listener,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    if policy:
                        print("record,policy is not None")
                    continue

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

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
