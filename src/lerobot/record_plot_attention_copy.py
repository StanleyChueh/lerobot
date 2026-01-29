import torch
import math
import matplotlib.pyplot as plt
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import cv2
import os
import argparse
import torch.nn.functional as F

# ======================================================
# 1. Global Cache and Hook Setup (New additions)
# ======================================================
captured_attn = {}

def cross_attn_hook(module, input, output):
    """
    Captured during policy.select_action() calls.
    Saves the Cross-Attention weights: [Batch, Heads, Action_Tokens, Visual_Tokens]
    """
    if isinstance(output, tuple) and len(output) > 1:
        # output[1] contains the attention weights
        captured_attn["last_cross_attn"] = output[1].detach()

def setup_expert_hooks(policy):
    """
    Targets the specific layer in VLAFlowMatching that computes Cross-Attention.
    """
    try:
        # Standard path for SmolVLA/VLAFlowMatching expert blocks
        expert_module = policy.model.vlm_with_expert.expert
        # We hook the very last transformer layer's cross-attention
        target_layer = expert_module.layers[-1].cross_attn
        target_layer.register_forward_hook(cross_attn_hook)
        print("[INFO] Hook registered successfully on Action Expert Cross-Attention.")
    except Exception as e:
        print(f"[ERROR] Could not register hook: {e}")

# ======================================================
# 2. Initialization & Argument Parsing
# ======================================================
parser = argparse.ArgumentParser(description="Visualize Trajectory Attention for SmolVLA")
parser.add_argument("--repo_id", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--token", type=str, default=None)
parser.add_argument("--use_state", action="store_true")
args = parser.parse_args()

# Load Policy
policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
dataset = LeRobotDataset(args.repo_id, root=None, batch_encoding_size=1)
policy = make_policy(policy_cfg, ds_meta=dataset.meta)
device = get_safe_torch_device(policy.config.device)
policy.reset()
policy.eval()

# Register the Hook before starting
setup_expert_hooks(policy)

# ======================================================
# 3. Helper Functions
# ======================================================
def extract_ca_attention(policy):
    """
    Extracts the captured Cross-Attention weights.
    Returns 1D weights for Front and Top cameras.
    """
    if "last_cross_attn" not in captured_attn:
        raise RuntimeError("No Cross-Attention captured. Check setup_expert_hooks.")

    # [Heads, Action_Tokens, Visual_Tokens]
    ca_matrix = captured_attn["last_cross_attn"][0] 
    
    # Average over heads and future action steps
    attn_1d = ca_matrix.mean(dim=(0, 1)) 

    # Split for dual cams
    num_img_tokens = policy.model.vlm_with_expert.last_attn["num_image_tokens"]
    heat_front = attn_1d[:num_img_tokens]
    heat_top = attn_1d[num_img_tokens : 2 * num_img_tokens]
    
    return heat_front, heat_top

def process_heatmap(heat_1d, original_image_size=(480, 640), model_input_size=(512, 512)):
    grid_size = int(math.sqrt(heat_1d.numel()))
    heat_2d = heat_1d.reshape(grid_size, grid_size)
    heat_tensor = torch.tensor(heat_2d).unsqueeze(0).unsqueeze(0).float()
    heat_512 = F.interpolate(heat_tensor, size=model_input_size, mode='bilinear', align_corners=False)
    
    orig_h, orig_w = original_image_size
    tgt_h, tgt_w = model_input_size
    ratio = max(orig_w / tgt_w, orig_h / tgt_h)
    resized_h, resized_w = int(orig_h / ratio), int(orig_w / ratio)
    pad_w = max(0, int(tgt_w - resized_w))
    pad_h = max(0, int(tgt_h - resized_h))
    
    heat_valid = heat_512[0, 0, pad_h : pad_h+resized_h, pad_w : pad_w+resized_w]
    heat_final = F.interpolate(heat_valid.unsqueeze(0).unsqueeze(0), size=original_image_size, mode='bilinear')
    return heat_final[0, 0].numpy()

# ======================================================
# 4. Main Loop
# ======================================================
target_episode_idx = args.episode
ep_meta = dataset.meta.episodes[target_episode_idx]
start_idx = sum(dataset.meta.episodes[i]['length'] for i in range(target_episode_idx))
end_idx = start_idx + ep_meta['length']

prompt = args.prompt
output_video_path = f"trajectory_attention_ep{target_episode_idx}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None

for global_idx in range(start_idx, end_idx):
    item = dataset[global_idx]
    observation_batch = {
        k: (torch.from_numpy(v).unsqueeze(0).to(device) if isinstance(v, np.ndarray) else v.unsqueeze(0).to(device))
        for k, v in item.items() if k.startswith("observation.")
    }
    observation_batch["task"] = prompt

    # --- [IMPORTANT] Run full prediction to trigger Expert Cross-Attention ---
    with torch.no_grad():
        actions = policy.select_action(observation_batch)

    # --- Extraction ---
    heat_front_1d, heat_top_1d = extract_ca_attention(policy)
    heat_2d_front = process_heatmap(heat_front_1d)
    heat_2d_top = process_heatmap(heat_top_1d)

    # --- Visualization ---
    rgb_front = (item["observation.images.front"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rgb_top = (item["observation.images.top"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def apply_heatmap_overlay(rgb_img, heat_map_2d):
        v_min, v_max = np.percentile(heat_map_2d, [0, 98])
        heat_norm = np.clip((heat_map_2d - v_min) / (v_max - v_min + 1e-6), 0, 1)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_JET)
        return cv2.addWeighted(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)

    vis_front = apply_heatmap_overlay(rgb_front, heat_2d_front)
    vis_top = apply_heatmap_overlay(rgb_top, heat_2d_top)
    overlay = np.hstack((vis_front, vis_top))

    if video_writer is None:
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (overlay.shape[1], overlay.shape[0]))
    
    video_writer.write(overlay)
    print(f"Frame {global_idx - start_idx}/{ep_meta['length']}", end="\r")

if video_writer:
    video_writer.release()
    print(f"\n[DONE] Video saved to {output_video_path}")