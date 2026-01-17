#!/usr/bin/env python3
"""
GR00T (Eagle backbone) attention visualization for LeRobot episodes.

What it does:
- Loads a GR00T policy from a LeRobot checkpoint.
- Runs a forward pass to compute actions (or just backbone) per frame.
- Captures Eagle self-attention (last layer by default).
- Aggregates attention from selected TEXT token(s) -> VISION token span.
- Reshapes vision attention to a patch grid and overlays on the front RGB image.
- Writes an MP4.

Notes:
- Default assumes 14x14=196 vision patch tokens for 224x224 vision input.
  GrootConfig uses image_size=(224,224) by default. Adjust --num_vision_tokens if needed.
"""

import os
import math
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from transformers import AutoTokenizer

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description="Visualize GR00T (Eagle) self-attention over vision tokens")
parser.add_argument("--repo_id", type=str, default="lerobot/svla_so100_pickplace", help="HuggingFace dataset repo ID")
parser.add_argument("--ckpt", type=str, required=True, help="Path to LeRobot GR00T checkpoint directory")
parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
parser.add_argument("--prompt", type=str, required=True, help="Task prompt")
parser.add_argument("--token", type=str, default=None, help="Specific word/token substring to visualize (case-insensitive)")
parser.add_argument("--layer", type=int, default=-1, help="Which Eagle transformer layer to use (default: -1 last)")
parser.add_argument("--head", type=int, default=-1, help="Which head to use; -1 means average over heads")
parser.add_argument("--num_vision_tokens", type=int, default=196, help="How many vision tokens to treat as image patches (default 196=14x14)")
parser.add_argument("--vision_offset", type=int, default=0, help="Start offset of vision tokens in the sequence (default 0)")
parser.add_argument("--out", type=str, default="groot_attention_vis.mp4", help="Output mp4 path")
parser.add_argument("--fps", type=int, default=30, help="Video FPS")
parser.add_argument("--use_backbone_only", action="store_true",
                    help="If set, runs only the Eagle backbone forward (no diffusion/action head).")
args = parser.parse_args()


# -------------------------
# Token selection helper (text side)
# -------------------------
def get_text_token_indices(tokenizer, prompt: str, specific_word: str | None):
    """
    Returns indices (in tokenized prompt) to aggregate over.
    NOTE: This is *prompt-local* token indices, not necessarily full multimodal sequence indices.
          We will map prompt tokens into the full sequence using a heuristic (see below).
    """
    toks = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = toks["input_ids"][0]
    if specific_word is None:
        return list(range(len(ids))), toks

    words = tokenizer.convert_ids_to_tokens(ids)
    indices = []
    for i, w in enumerate(words):
        clean = w.replace("Ġ", "").replace("▁", "").lower()
        if specific_word.lower() in clean:
            indices.append(i)
    if not indices:
        raise ValueError(f"Token '{specific_word}' not found in prompt tokens: {words}")
    return indices, toks


# -------------------------
# Vision attention -> heatmap helper
# -------------------------
def vision_attn_to_heatmap(attn_1d: torch.Tensor, grid_hw: tuple[int, int], out_hw: tuple[int, int]):
    """
    attn_1d: [num_vision_tokens]
    grid_hw: (gh, gw) such that gh*gw == num_vision_tokens
    out_hw: (H, W) of original image
    """
    gh, gw = grid_hw
    heat_2d = attn_1d.reshape(gh, gw)[None, None].float()  # [1,1,gh,gw]
    heat_up = F.interpolate(heat_2d, size=out_hw, mode="bilinear", align_corners=False)[0, 0]
    heat_np = heat_up.detach().cpu().numpy()
    return heat_np


def apply_heatmap_overlay(rgb_img: np.ndarray, heat_map_2d: np.ndarray):
    # percentile clip to suppress outliers
    v_min, v_max = np.percentile(heat_map_2d, [0, 98])
    heat_clipped = np.clip(heat_map_2d, v_min, v_max)
    if v_max - v_min > 1e-6:
        heat_norm = (heat_clipped - v_min) / (v_max - v_min)
    else:
        heat_norm = heat_clipped * 0.0

    heat_color = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr, 0.6, heat_color, 0.4, 0)


# -------------------------
# Load dataset + policy
# -------------------------
print("[INFO] Loading dataset...")
dataset = LeRobotDataset(args.repo_id, root=None, batch_encoding_size=1)

total_episodes = len(dataset.meta.episodes)
if args.episode < 0 or args.episode >= total_episodes:
    raise ValueError(f"Episode index {args.episode} out of range (0..{total_episodes-1})")

ep_meta = dataset.meta.episodes[args.episode]
ep_len = ep_meta["length"]

start_idx = sum(dataset.meta.episodes[i]["length"] for i in range(args.episode))
end_idx = start_idx + ep_len

print(f"[INFO] Episode {args.episode}: global idx range [{start_idx}, {end_idx}) length={ep_len}")

print("[INFO] Loading policy config...")
policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)

print("[INFO] Building policy...")
policy = make_policy(policy_cfg, ds_meta=dataset.meta)
device = get_safe_torch_device(policy.config.device)
policy.reset()
policy = policy.to(device)
policy.eval()

preproc, _ = make_groot_pre_post_processors(
    policy.config,
    dataset_stats=getattr(dataset, "stats", None),
)


# -------------------------
# Monkey-patch Eagle backbone to store attentions
# -------------------------
# GR00T wrapper exposes the underlying Isaac-GR00T model as _groot_model (see GrootPolicy) :contentReference[oaicite:1]{index=1}
# and the Eagle backbone is in _groot_model.backbone (see GR00TN15) :contentReference[oaicite:2]{index=2}.
groot_model = getattr(policy, "_groot_model", None)
if groot_model is None:
    raise RuntimeError("This policy does not look like GrootPolicy (missing _groot_model). "
                       "Make sure you loaded a GR00T checkpoint (config subclass 'groot').")

backbone = groot_model.backbone
eagle_model = backbone.eagle_model

if hasattr(policy._groot_model.backbone.eagle_model, "set_attn_implementation"):
    policy._groot_model.backbone.eagle_model.set_attn_implementation("eager")
else:
    # Fallback for older HF versions
    policy._groot_model.backbone.eagle_model.config.attn_implementation = "eager"

# Patch forward_eagle to enable output_attentions and store them.
# modeling_groot.EagleBackbone.forward_eagle calls self.eagle_model(... output_hidden_states=True ...) :contentReference[oaicite:3]{index=3}

# Patch the correct owner of forward_eagle (usually groot_model, not backbone)
# Patch EagleBackbone.forward_eagle (THIS is what gets called)
if not hasattr(backbone, "forward_eagle"):
    raise RuntimeError("Expected backbone.forward_eagle to exist, but it does not.")

orig_forward_eagle = backbone.forward_eagle

def forward_eagle_with_attn(vl_input):
    eagle_prefix = "eagle_"
    eagle_input = {
        k.removeprefix(eagle_prefix): v
        for k, v in vl_input.items()
        if k.startswith(eagle_prefix)
    }

    eagle_input.pop("image_sizes", None)
    backbone.last_eagle_input_ids = eagle_input.get("input_ids", None)

    device = next(backbone.parameters()).device

    # 🔴 THIS IS THE CRITICAL FIX
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        eagle_out = backbone.eagle_model(
            **eagle_input,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    backbone.last_attentions = eagle_out.attentions

    eagle_features = eagle_out.hidden_states[backbone.select_layer]
    eagle_features = backbone.eagle_linear(eagle_features)

    return eagle_features, eagle_input["attention_mask"]


backbone.forward_eagle = forward_eagle_with_attn

# -------------------------
# Tokenizer for prompt token selection
# -------------------------
# Eagle tokenizer is available under the eagle_model config; safest is AutoTokenizer via the processor,
# but in this integration the tokenizer is embedded in Eagle assets.
# We can use eagle_model's internal tokenizer if exposed; if not, fall back to transformers AutoTokenizer.

tokenizer = AutoTokenizer.from_pretrained(
    getattr(policy_cfg, "tokenizer_assets_repo", "lerobot/eagle2hg-processor-groot-n1p5"),
    trust_remote_code=True,
    fix_mistral_regex=True,
)

text_tok_indices, prompt_tokens = get_text_token_indices(tokenizer, args.prompt, args.token)
prompt_ids = prompt_tokens["input_ids"][0].tolist()


# -------------------------
# Heuristic: map prompt token indices into full multimodal sequence indices
# -------------------------
def map_prompt_indices_to_sequence(full_input_ids: torch.Tensor, prompt_ids_list: list[int], prompt_token_indices: list[int]):
    """
    Heuristic: find the prompt_ids as a contiguous subsequence inside full_input_ids, then offset indices.
    If not found, fall back to using the *last* len(prompt_ids) tokens as text.
    """
    full = full_input_ids[0].tolist()
    n = len(prompt_ids_list)
    start = -1
    for i in range(0, len(full) - n + 1):
        if full[i:i+n] == prompt_ids_list:
            start = i
            break
    if start < 0:
        # fallback: assume prompt sits at the end
        start = max(0, len(full) - n)
    return [start + i for i in prompt_token_indices], start


# -------------------------
# Video writer
# -------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = None

print("[INFO] Start processing frames...")
for fidx, global_idx in enumerate(range(start_idx, end_idx)):
    item = dataset[global_idx]

    # Build a batch using the same keys your policy expects.
    # In LeRobot, policy wrappers typically accept "observation.*" tensors + "task".
    # GrootPolicy will be used through the LeRobot pipeline that builds eagle_* fields internally.
    batch = {}

    # Copy observation.* tensors
    for k, v in item.items():
        if k.startswith("observation."):
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor):
                v = v.unsqueeze(0).to(device)
            batch[k] = v

    batch["task"] = args.prompt

    # Run forward to populate backbone.last_attentions
    with torch.no_grad():
        if args.use_backbone_only:
            raise NotImplementedError("--use_backbone_only is not supported in this script; use the default path.")

        processed = preproc(batch)

        # Ensure tensors are on the same device as the model
        for k, v in processed.items():
            if isinstance(v, torch.Tensor):
                processed[k] = v.to(device)

        assert "eagle_pixel_values" in processed
        assert "eagle_input_ids" in processed

        device = next(policy.parameters()).device

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _ = policy._groot_model.get_action(processed)

    if not hasattr(backbone, "last_attentions") or backbone.last_attentions is None:
        raise RuntimeError(
            "No attentions captured. This usually means Eagle did not receive output_attentions=True "
            "or the forward path did not run Eagle. Try running without --use_backbone_only."
        )

    # Get attention from selected layer
    layer_attn = backbone.last_attentions[args.layer]
    if args.head >= 0:
        attn = layer_attn[0, args.head]  # [T, T]
    else:
        attn = layer_attn[0].mean(0)     # [T, T]

    # Map prompt token indices into the full sequence
    full_input_ids = backbone.last_eagle_input_ids

    if full_input_ids is None:
        raise RuntimeError("Could not access Eagle input_ids; cannot locate prompt tokens in the sequence.")

    seq_text_indices, text_start = map_prompt_indices_to_sequence(full_input_ids, prompt_ids, text_tok_indices)

    # Aggregate rows (query = selected text tokens), then slice cols (keys = vision tokens)
    attn_rows = attn[seq_text_indices].mean(0)  # [T]
    v0 = args.vision_offset
    v1 = v0 + args.num_vision_tokens
    if v1 > attn_rows.numel():
        raise ValueError(
            f"vision span [{v0},{v1}) exceeds sequence length {attn_rows.numel()}. "
            "Adjust --num_vision_tokens and/or --vision_offset."
        )

    vision_attn = attn_rows[v0:v1]  # [num_vision_tokens]

    # Reshape to patch grid
    grid = int(math.sqrt(args.num_vision_tokens))
    if grid * grid != args.num_vision_tokens:
        raise ValueError(
            f"--num_vision_tokens must be a perfect square for 2D heatmap; got {args.num_vision_tokens}."
        )

    # Get front RGB for visualization
    rgb_front = item["observation.images.front"].permute(1, 2, 0).cpu().numpy()
    if rgb_front.max() <= 1.5:
        rgb_front = (rgb_front * 255).astype(np.uint8)
    else:
        rgb_front = rgb_front.astype(np.uint8)

    H, W = rgb_front.shape[:2]
    heat = vision_attn_to_heatmap(vision_attn, (grid, grid), (H, W))
    vis = apply_heatmap_overlay(rgb_front, heat)

    # Title overlay
    token_label = args.token if args.token is not None else "AllPromptTokens"
    cv2.putText(
        vis,
        f"GR00T Eagle self-attn L{args.layer} H{args.head} | {token_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if video_writer is None:
        video_writer = cv2.VideoWriter(args.out, fourcc, args.fps, (vis.shape[1], vis.shape[0]))
        print(f"[INFO] Video initialized: {vis.shape[1]}x{vis.shape[0]} -> {args.out}")

    video_writer.write(vis)

    if fidx % 30 == 0:
        print(f"[INFO] Frame {fidx}/{ep_len} (global_idx={global_idx})")

if video_writer is not None:
    video_writer.release()

print(f"[DONE] Saved to: {os.path.abspath(args.out)}")
