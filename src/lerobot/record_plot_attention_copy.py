'''
[ Image tokens | Language tokens | (State tokens) ]
            ↓
      Transformer Encoder
            ↓
     Self-attention matrices

[ Prefix KV cache ]
            ↓
[ Action tokens ] → Cross-attention → Denoising → Actions

Usage:

python src/lerobot/record_plot_attention.py  --repo_id "ethanCSL/Ting_grip_box_svla"   --ckpt "/home/bruce/CSL/lerobot_nn/outputs/train/svla_so100_pickplace_paper/pretrained_model"   --episode 0   --prompt "Pick up the cube and place it in the box"   --use_state
'''

import torch
import math
import matplotlib.pyplot as plt
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import cv2
import os
import argparse
import torch.nn.functional as F

# 1. Initialization & Argument Parsing
parser = argparse.ArgumentParser(description="Visualize Attention Maps for SmolVLA")
parser.add_argument("--repo_id", type=str, default="lerobot/svla_so100_pickplace", help="HuggingFace Dataset Repo ID")
parser.add_argument("--ckpt", type=str, default="/home/bruce/CSL/lerobot_nn/outputs/train/svla_so100_pickplace_paper", help="Path to model checkpoint")
parser.add_argument("--episode", type=int, default=10, help="Episode index to visualize")
parser.add_argument("--prompt", type=str, default="grip the green block and put it into box", help="Task prompt")
parser.add_argument("--token", type=str, default=None, help="(Optional) Specific word to visualize.")
parser.add_argument("--use_state", action="store_true", help="Condition attention on joint states")

args = parser.parse_args()

DATASET_REPO_ID = args.repo_id
CKPT_PATH = args.ckpt

# Load Config
policy_cfg = PreTrainedConfig.from_pretrained(CKPT_PATH)

print("[INFO] Loading Dataset...")
dataset = LeRobotDataset(
    DATASET_REPO_ID,
    root=None,
    batch_encoding_size=1,
)

# Load Policy
policy = make_policy(policy_cfg, ds_meta=dataset.meta)
device = get_safe_torch_device(policy.config.device)
policy.reset()
policy.model.vlm_with_expert.debug_attn = True

# 2. Episode Selection
target_episode_idx = args.episode

total_episodes = len(dataset.meta.episodes)
if target_episode_idx < 0 or target_episode_idx >= total_episodes:
    raise ValueError(f"Episode index {target_episode_idx} is out of range")

ep_meta = dataset.meta.episodes[target_episode_idx]
ep_length = ep_meta['length']

start_idx = 0
for i in range(target_episode_idx):
    start_idx += dataset.meta.episodes[i]['length']
end_idx = start_idx + ep_length

print(f"\n[INFO] Processing Episode: {target_episode_idx}")
print(f"       Range: {start_idx} to {end_idx}")

# =========================
# 3. Helper Functions 
# =========================

def get_token_indices(processor, prompt, specific_word=None):
    tokens = processor.tokenizer(prompt, return_tensors="pt")
    ids = tokens["input_ids"][0]
    
    if specific_word is None:
        return list(range(len(ids)))

    words = processor.tokenizer.convert_ids_to_tokens(ids)
    indices = []
    for i, w in enumerate(words):
        # Clean up token strings (remove special chars like Ġ) for matching
        clean_w = w.replace('Ġ', '').lower()
        if specific_word.lower() in clean_w:
            indices.append(i)
    
    if not indices:
        raise ValueError(f"Token '{specific_word}' not found in prompt tokens: {words}")
    return indices

def extract_trajectory_attention(policy):
    """
    從 Action Expert 中提取交叉注意力權重。
    這代表了 '動作決策' 對 '視覺特徵' 的依賴程度。
    """
    # 1. 獲取 Expert 的交叉注意力 (通常在最後一次反饋中記錄)
    # 注意：具體屬性名取決於 SmolVLA 的實現，通常是在 Expert Transformer 的層中
    # 這裡假設 policy 已經運行了 select_action 並且緩存了注意力
    if not hasattr(policy.model.action_expert, "last_cross_attn"):
        raise RuntimeError("請確保 policy 已經運行了完整的 select_action 並開啟了 debug_attn")

    # shape: [Batch, Heads, Action_Tokens, Visual_Tokens]
    cross_attn = policy.model.action_expert.last_cross_attn
    
    # 2. 對 Heads 和 Action Steps (預測的未來軌跡步數) 取平均
    # 我們想知道這一段軌跡整體在看哪裡
    avg_attn = cross_attn[0].mean(dim=(0, 1)) # shape: [Visual_Tokens]

    # 3. 獲取視覺 Token 的數量
    num_img_tokens = policy.model.vlm_with_expert.last_attn["num_image_tokens"]
    
    # 拆分前後攝像頭
    heat_front = avg_attn[:num_img_tokens]
    heat_top = avg_attn[num_img_tokens : 2 * num_img_tokens]
    
    return heat_front, heat_top

# =========================
# 修改後的 Main Loop
# =========================

for global_idx in range(start_idx, end_idx):
    item = dataset[global_idx]
    
    # 構造 Batch (與之前相同)
    observation_batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in item.items() if k.startswith("observation.")}
    observation_batch["task"] = prompt

    # --- [關鍵修改] 執行真正的動作預測 ---
    with torch.no_grad():
        # 這會觸發 Action Expert 並生成未來軌跡
        # 確保你的 policy 或 model 內部實現了對 Cross-Attention 的記錄
        actions = policy.select_action(observation_batch)

    # --- 提取動作決策的注意力 ---
    heat_front_1d, heat_top_1d = extract_trajectory_attention(policy)
    
    # 後續的 process_heatmap 和 apply_heatmap_overlay 邏輯不變
    heat_2d_front = process_heatmap(heat_front_1d)
    heat_2d_top = process_heatmap(heat_top_1d)
    
    # ... (其餘繪圖與保存影片邏輯相同)

def process_heatmap(heat_1d, original_image_size=(480, 640), model_input_size=(512, 512)):
    """
    Correctly reshapes 1D attention tokens back to 2D image space 
    accounting for SmolVLA's square padding logic.
    """
    # 1. Reshape to Square Grid (e.g. 32x32)
    grid_size = int(math.sqrt(heat_1d.numel()))
    if grid_size * grid_size != heat_1d.numel():
        # Fallback if not perfect square (rare)
        grid_size = int(math.sqrt(heat_1d.numel()))
        heat_1d = heat_1d[:grid_size*grid_size]
        
    heat_2d = heat_1d.reshape(grid_size, grid_size)
    
    # 2. Prepare for interpolation (Batch, Channel, H, W)
    heat_tensor = torch.tensor(heat_2d).unsqueeze(0).unsqueeze(0).float()
    
    # 3. Upscale to Model Input Size (512x512)
    heat_512 = F.interpolate(heat_tensor, size=model_input_size, mode='bilinear', align_corners=False)
    
    # 4. Calculate Padding (Reverse the padding logic from training)
    orig_h, orig_w = original_image_size
    tgt_h, tgt_w = model_input_size
    
    ratio = max(orig_w / tgt_w, orig_h / tgt_h)
    resized_h = int(orig_h / ratio)
    resized_w = int(orig_w / ratio)
    
    # Padding is applied to Left and Top in F.pad usually, but we check specific config
    # Standard logic: Pad to bottom-right or center. 
    # Based on modeling_smolvla: F.pad(resized_img, (pad_width, 0, pad_height, 0))
    # This means padding is on LEFT and TOP.
    pad_w = max(0, int(tgt_w - resized_w))
    pad_h = max(0, int(tgt_h - resized_h))
    
    # 5. Crop out the valid region (exclude padding)
    heat_valid = heat_512[0, 0, pad_h : pad_h+resized_h, pad_w : pad_w+resized_w]
    
    # 6. Resize to original image size
    heat_final = F.interpolate(heat_valid.unsqueeze(0).unsqueeze(0), size=original_image_size, mode='bilinear')
    
    return heat_final[0, 0].numpy()

def compute_deterministic_attention(policy, batch, device):
    """
    Runs ONLY the Image+Text encoding pass (Prefix) to get deterministic attention.
    Skips the diffusion/action generation loop.
    """
    policy.eval()
    
    # 1. Prepare Inputs using Policy's internal helpers
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
            
    # Normalize images and prepare masks
    batch = policy.normalize_inputs(batch)
    images, img_masks = policy.prepare_images(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)
    state = policy.prepare_state(batch)
    
    # 2. Embed the Prefix (Image + Text + State)
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    
    # 3. Create Attention Masks
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    
    # 4. Run the Transformer Forward Pass (JUST THE PREFIX)
    with torch.no_grad():
        policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None], # None means no Expert/Action tokens
            
            # [CRITICAL FIX] 
            # We must set these to True to trigger 'forward_attn_layer', 
            # which safely handles [prefix_embs, None].
            # If False, it triggers 'forward_cross_attn_layer' which crashes on None.
            use_cache=True,
            fill_kv_cache=True, 
        )
        
    return

# =========================
# 4. Main Loop (FIXED)
# =========================

prompt = args.prompt
processor = policy.model.vlm_with_expert.processor
target_indices = get_token_indices(processor, prompt, args.token)

mode_suffix = "with_state" if args.use_state else "no_state"
output_video_path = f"attention_vis_{mode_suffix}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30 
video_writer = None

print(f"[INFO] Start processing Frames...")
current_frame = 0

for global_idx in range(start_idx, end_idx):
    item = dataset[global_idx]

    # --- [FIX START] Correctly Build the Batch ---
    observation_batch = {}
    
    # 1. Copy data from the Dataset Item
    for k, v in item.items():
        # We only care about observation keys (images, state)
        if k.startswith("observation."):
            val = v
            
            # Ensure it is a Torch Tensor (Dataset might return Numpy)
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            
            # Add Batch Dimension: (C, H, W) -> (1, C, H, W)
            # and move to GPU
            if isinstance(val, torch.Tensor):
                val = val.unsqueeze(0).to(device)
            
            observation_batch[k] = val

    # 2. Add the Task (Prompt)
    # The attention function needs the text prompt to compute text embeddings
    observation_batch["task"] = prompt
    # --- [FIX END] ---

    # Predict (Using the new Deterministic Function)
    compute_deterministic_attention(policy, observation_batch, device)

    # --- ATTENTION EXTRACTION ---
    heat_front_1d, heat_top_1d = extract_aggregated_attention(policy, target_indices)
    
    heat_2d_front = process_heatmap(heat_front_1d)
    heat_2d_top = process_heatmap(heat_top_1d)

    # --- VISUALIZATION ---
    # Retrieve original RGB from dataset item (CPU side) for visualization
    rgb_front = item["observation.images.front"].permute(1, 2, 0).cpu().numpy()
    if "observation.images.top" in item:
        rgb_top = item["observation.images.top"].permute(1, 2, 0).cpu().numpy()
    else:
        rgb_top = np.zeros_like(rgb_front)

    # Convert to uint8 [0, 255]
    if rgb_front.max() <= 1.5:
        rgb_front = (rgb_front * 255).astype(np.uint8)
        rgb_top = (rgb_top * 255).astype(np.uint8)
    else:
        rgb_front = rgb_front.astype(np.uint8)
        rgb_top = rgb_top.astype(np.uint8)

    def apply_heatmap_overlay(rgb_img, heat_map_2d):
        # [FIX] Use Percentile-based normalization to ignore "Register Artifacts"
        # This clips the top 2% of brightest pixels so outliers don't hide the real data
        v_min, v_max = np.percentile(heat_map_2d, [0, 98]) 
        
        # Clip values to this range
        heat_clipped = np.clip(heat_map_2d, v_min, v_max)
        
        # Normalize to 0-1
        if v_max - v_min > 1e-6:
            heat_norm = (heat_clipped - v_min) / (v_max - v_min)
        else:
            heat_norm = heat_clipped - v_min
            
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_JET)
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(rgb_bgr, 0.6, heatmap_color, 0.4, 0)

    vis_front = apply_heatmap_overlay(rgb_front, heat_2d_front)
    vis_top = apply_heatmap_overlay(rgb_top, heat_2d_top)

    overlay = np.hstack((vis_front, vis_top))
    h, w, _ = vis_front.shape
    cv2.line(overlay, (w, 0), (w, h), (255, 255, 255), 2)
    cv2.putText(overlay, f"Front ({args.token or 'All'})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, "Top", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if video_writer is None:
        H_out, W_out = overlay.shape[:2]
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W_out, H_out))
        print(f"[INFO] Video initialized: {W_out}x{H_out}")
    
    if current_frame % 30 == 0:
        print(f"Processing Frame {current_frame}/{ep_length}")

    video_writer.write(overlay)
    current_frame += 1

if video_writer:
    video_writer.release()
    print(f"\n[DONE] Saved to {os.path.abspath(output_video_path)}")