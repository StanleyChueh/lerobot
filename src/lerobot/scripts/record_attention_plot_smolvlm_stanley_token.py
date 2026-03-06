'''
[ Image tokens | Language tokens | (State tokens) ]
            ↓
      Transformer Encoder
            ↓
     Self-attention matrices

[ Prefix KV cache ]
            ↓
[ Action tokens ] → Cross-attention → Denoising → Actions

This is the standalone VLM self attention visualization test, as record_plot_attention.py will use trained model,and smolvla will fine-tune vision encoder

python src/lerobot/scripts/record_attention_plot_smolvlm_stanley_token.py     --repo_id "ethanCSL/svla_koch_sorting_n_stacking"          --episode 0     --prompt "Put the red cube in the right box,green cube in the left box." --use_state

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
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.configs.types import NormalizationMode
import torch.nn.functional as F

# 1. Initialization & Argument Parsing
parser = argparse.ArgumentParser(description="Visualize Attention Maps for SmolVLA")
parser.add_argument("--repo_id", type=str, default="lerobot/svla_so100_pickplace", help="HuggingFace Dataset Repo ID")
parser.add_argument("--episode", type=int, default=10, help="Episode index to visualize")
parser.add_argument("--prompt", type=str, default="grip the green block and put it into box", help="Task prompt")
parser.add_argument("--token", type=str, default=None, help="(Optional) Specific word to visualize.")
parser.add_argument("--use_state", action="store_true", help="Condition attention on joint states")
parser.add_argument("--video_backend", type=str, default="pyav")

args = parser.parse_args()

DATASET_REPO_ID = args.repo_id

# Load Config
policy_cfg = SmolVLAConfig(
    vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    load_vlm_weights=True,
    freeze_vision_encoder=True,
    train_expert_only=True,
    train_state_proj=False,
    attention_mode="self_attn",
    device="cuda",
    empty_cameras=0,
    num_vlm_layers=16,
)

policy_cfg.normalization_mapping["STATE"] = NormalizationMode.IDENTITY

policy_cfg.input_features.update({
    "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    "observation.images.top":   PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
})

print("[INFO] Loading Dataset...")
dataset = LeRobotDataset(
    DATASET_REPO_ID,
    root=None,
    batch_encoding_size=1,
    video_backend=args.video_backend,
)

# Load Policy
policy = SmolVLAPolicy(
    config=policy_cfg,
    dataset_stats=None,
)
device = torch.device(policy_cfg.device)

policy = policy.to(device)
policy.reset()
policy.model.vlm_with_expert.debug_attn = True

print("[DEBUG] attention_mode:", policy.model.vlm_with_expert.attention_mode)

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
    if isinstance(prompt, str) and not prompt.endswith("\n"):
        prompt = prompt + "\n"
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

def extract_full_attention(policy):
    attn = getattr(policy.model.vlm_with_expert, "last_attn_weights", None)
    if attn is None:
        raise RuntimeError("No attention captured.")

    attn_matrix = attn[0].mean(0)  # [Q, K]

    num_img_tokens = policy.model.vlm_with_expert._debug_num_img_tokens
    num_images = policy.model.vlm_with_expert._debug_num_images
    total_img_tokens = num_img_tokens * num_images

    return attn_matrix, total_img_tokens

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
    heat_tensor = heat_2d.unsqueeze(0).unsqueeze(0).float()
    
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
    
    return heat_final[0, 0].detach().cpu().numpy()

def draw_prompt_weights_aligned(img, prompt, weights):
    h, w, _ = img.shape
    text_bar_height = 100
    # 建立白色底框
    text_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(prompt, font, font_scale, thickness)
    
    start_x = 20
    start_y = 40
    
    # 繪製 Prompt 文字
    cv2.putText(text_bar, f"{prompt}", (start_x, start_y), 
                font, font_scale, (50, 50, 50), thickness)

    # 繪製權重條 (Language Attention Weights)
    if len(weights) > 0:
        w_array = np.array(weights)
        # 歸一化權重以利視覺化
        norm_w = (w_array - w_array.min()) / (w_array.max() - w_array.min() + 1e-8)
        
        bar_width = text_w + 100 
        bar_y = start_y + 15
        
        # 將 1D 權重轉為彩色條
        small_bar = (norm_w.reshape(1, -1) * 255).astype(np.uint8)
        color_bar = cv2.applyColorMap(cv2.resize(small_bar, (bar_width, 15)), cv2.COLORMAP_JET)
        
        text_bar[bar_y:bar_y+15, start_x:start_x+bar_width] = color_bar
    
    return np.vstack((img, text_bar))

def analyze_attention_refined(attn_matrix, num_img_tokens, num_lang_tokens, num_state_tokens):
    """
    分析 Self-Attention 矩陣中，所有 Query 對於各個 Token 區段的平均關注比例。
    """
    mean_attn = attn_matrix.mean(dim=0).float().cpu()
    
    # 定義區間 (與主迴圈一致)
    img_end = 2 * num_img_tokens
    lang_end = img_end + num_lang_tokens
    state_end = lang_end + num_state_tokens
    
    vision_attn = mean_attn[:img_end].sum().item()
    text_attn = mean_attn[img_end:lang_end].sum().item()
    state_attn = mean_attn[lang_end:state_end].sum().item()
    
    total_tokens = mean_attn.shape[0]
    others_attn = mean_attn[state_end:].sum().item() if total_tokens > state_end else 0
    
    total = vision_attn + text_attn + state_attn + others_attn + 1e-8
    
    return {
        "Vision": vision_attn / total,
        "Text": text_attn / total,
        "State": state_attn / total,
        "Padding": others_attn / total,
        "valid_sum": vision_attn + text_attn + state_attn # 用於縮放影像權重
    }

def apply_heatmap_overlay(rgb_img, heat_map_2d):
    """
    修改後的 Overlay：不再進行內部的百分位數歸一化，
    而是直接將傳入的權重轉為顏色，這樣權重變低時顏色會變暗。
    """
    # 這裡的 heat_map_2d 已經是經過 valid_total_attn 縮放後的比例
    # 為了讓視覺效果明顯，我們設定一個合理的基準最大值 (例如 0.05 代表強關注)
    v_max = 0.05 
    heat_norm = np.clip(heat_map_2d / v_max, 0, 1)
            
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_JET)
    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(rgb_bgr, 0.6, heatmap_color, 0.4, 0)

def compute_deterministic_attention(policy, batch, device):
    """
    Runs ONLY the Image+Text(+State) encoding pass (Prefix) to get deterministic self-attention.
    Skips the diffusion/action generation loop.

    Compatibility notes:
      - SmolVLAPolicy has no prepare_language(); language tokens must already be in the batch.
      - We tokenize via the VLM processor tokenizer and write:
            OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
      - We cache image token counts to avoid "text/state leakage" when slicing heatmaps later.
    """
    policy.eval()

    # Move tensors to device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # Ensure prompt ends with newline (matches SmolVLANewLineProcessor)
    task = batch.get("task", None)
    if isinstance(task, str) and not task.endswith("\n"):
        task = task + "\n"
        batch["task"] = task

    # Tokenize language
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tok = tokenizer(
        task,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
    )
    batch[OBS_LANGUAGE_TOKENS] = tok["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].to(device).bool()

    # Images + masks (SmolVLAPolicy expects images already in batch)
    images, img_masks = policy.prepare_images(batch)

    # Cache image token count for later slicing (computed once per call)
    with torch.no_grad():
        num_img_tokens = int(policy.model.vlm_with_expert.embed_image(images[0]).shape[1])
    policy.model.vlm_with_expert._debug_num_img_tokens = num_img_tokens
    policy.model.vlm_with_expert._debug_num_images = len(images)

    # State (optional)
    if OBS_STATE in batch:
        state = policy.prepare_state(batch)

        # ====== 強制放大 state（測試用） ======
        state = state * 1000.0
    else:
        bsize = batch[OBS_LANGUAGE_TOKENS].shape[0]
        state = torch.zeros((bsize, policy.config.max_state_dim),
                            device=device, dtype=torch.float32)

    # Embed prefix
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state
    )

    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    with torch.no_grad():
        policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
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

# ===== Create output folder =====
block_dir = "attn_block_every30"
os.makedirs(block_dir, exist_ok=True)

for global_idx in range(start_idx, end_idx):
    item = dataset[global_idx]

    observation_batch = {}

    for k, v in item.items():
        if k in ("observation.images.front", "observation.images.top"):
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            observation_batch[k] = v.unsqueeze(0).to(device)

        elif k == "observation.state" and args.use_state:
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            observation_batch[OBS_STATE] = v.unsqueeze(0).to(device)

    observation_batch["task"] = prompt

    # print("Batch keys:", observation_batch.keys())
    # print("Expected image features:", policy.config.image_features)

    # Predict (Using the new Deterministic Function)
    compute_deterministic_attention(policy, observation_batch, device)

    # --- ATTENTION EXTRACTION ---
    attn_matrix, total_img_tokens = extract_full_attention(policy)

    # ===== Token boundary calculation =====
    num_lang_tokens = observation_batch[OBS_LANGUAGE_TOKENS].shape[1]
    num_state_tokens = observation_batch[OBS_STATE].shape[1] if OBS_STATE in observation_batch else 0

    num_img_tokens_single = policy.model.vlm_with_expert._debug_num_img_tokens
    img_start = 0
    img_end = total_img_tokens
    lang_start = img_end
    lang_end = lang_start + num_lang_tokens
    state_start = lang_end
    state_end = state_start + num_state_tokens

    # ===== Generate block-level attention every 30 frames =====
    if current_frame % 30 == 0 and num_state_tokens > 0:

        attn_np = attn_matrix.detach().cpu().numpy()

        def block_mean(r0, r1, c0, c1):
            if r1 <= r0 or c1 <= c0:
                return 0.0
            return attn_np[r0:r1, c0:c1].mean()

        block_mat = np.array([
            [
                block_mean(img_start, img_end, img_start, img_end),
                block_mean(img_start, img_end, lang_start, lang_end),
                block_mean(img_start, img_end, state_start, state_end)
            ],
            [
                block_mean(lang_start, lang_end, img_start, img_end),
                block_mean(lang_start, lang_end, lang_start, lang_end),
                block_mean(lang_start, lang_end, state_start, state_end)
            ],
            [
                block_mean(state_start, state_end, img_start, img_end),
                block_mean(state_start, state_end, lang_start, lang_end),
                block_mean(state_start, state_end, state_start, state_end)
            ]
        ])

        plt.figure(figsize=(6,6))
        plt.imshow(block_mat, cmap="viridis", vmin=0.0, vmax=0.1)
        plt.colorbar(label="Attention Weight")

        plt.xticks([0,1,2], ["Key:Image","Key:Language","Key:State"])
        plt.yticks([0,1,2], ["Query:Image","Query:Language","Query:State"])

        plt.xlabel("Key (What is being attended to)")
        plt.ylabel("Query (Who is attending)")
        plt.title(f"Block-Level Self-Attention (Frame {current_frame})")

        plt.tight_layout()

        save_path = os.path.join(block_dir,
                                f"block_attention_frame_{current_frame}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"[INFO] Saved block attention at frame {current_frame}")

    # ===== Print modality interaction statistics =====
    if num_state_tokens > 0:

        # prompt as query, img and prompt as key,value
        lang_to_img = attn_matrix[lang_start:lang_end, img_start:img_end]
        lang_to_state = attn_matrix[lang_start:lang_end, state_start:state_end]

        # state action as query, img or prompt as key,value
        state_to_img = attn_matrix[state_start:state_end, img_start:img_end] 
        state_to_lang = attn_matrix[state_start:state_end, lang_start:lang_end]

        # print("Language → Image mean:", lang_to_img.mean().item())
        # print("Language → State mean:", lang_to_state.mean().item())
        # print("State → Image mean:", state_to_img.mean().item())
        # print("State → Language mean:", state_to_lang.mean().item())

    # # ===== Slice language→image attention for heatmap =====
    # lang_query = attn_matrix[lang_start:lang_end].mean(0)

    # num_img_tokens_single = policy.model.vlm_with_expert._debug_num_img_tokens

    # heat_front_1d = lang_query[img_start : img_start + num_img_tokens_single]
    # heat_top_1d = lang_query[img_start + num_img_tokens_single : img_end]


    # heat_2d_front = process_heatmap(heat_front_1d)
    # heat_2d_top = process_heatmap(heat_top_1d)

    # --- 新增：分析與列印權重比例 ---
    attn_results = analyze_attention_refined(
        attn_matrix, 
        num_img_tokens_single, # 單台相機的 token 數
        num_lang_tokens, 
        num_state_tokens
    )
    valid_total_attn = attn_results['valid_sum'] + 1e-8
    print(
        f"Frame {current_frame}: "
        f"Vision:{attn_results['Vision']:.1%} | "
        f"Text:{attn_results['Text']:.1%} | "
        f"State:{attn_results['State']:.1%} | "
        f"Padding:{attn_results['Padding']:.1%}"
    )

    # # --- VISUALIZATION ---
    # # Retrieve original RGB from dataset item (CPU side) for visualization
    # rgb_front = item["observation.images.front"].permute(1, 2, 0).cpu().numpy()
    # if "observation.images.top" in item:
    #     rgb_top = item["observation.images.top"].permute(1, 2, 0).cpu().numpy()
    # else:
    #     rgb_top = np.zeros_like(rgb_front)

    # # Convert to uint8 [0, 255]
    # if rgb_front.max() <= 1.5:
    #     rgb_front = (rgb_front * 255).astype(np.uint8)
    #     rgb_top = (rgb_top * 255).astype(np.uint8)
    # else:
    #     rgb_front = rgb_front.astype(np.uint8)
    #     rgb_top = rgb_top.astype(np.uint8)

    # --- 2. 提取影像權重並根據有效總和重新縮放 ---
    # 這裡使用 Language Query (或 Action Query) 對影像的關注
    # 為了對齊您的邏輯，我們對 Query 維度取平均
    mean_attn_vector = attn_matrix.mean(dim=0) 

    heat_front_1d = mean_attn_vector[img_start : img_start + num_img_tokens_single]
    heat_top_1d = mean_attn_vector[img_start + num_img_tokens_single : img_start + 2*num_img_tokens_single]
    
    # 轉為 2D Heatmap
    heat_2d_front = process_heatmap(heat_front_1d)
    heat_2d_top = process_heatmap(heat_top_1d)

    # --- 3. 提取文字權重清單用於下方繪圖 ---
    text_weights_list = (mean_attn_vector[lang_start:lang_end] / valid_total_attn).cpu().tolist()

    # --- 4. 視覺化生成 ---
    # (影像讀取與 uint8 轉換邏輯不變...)
    rgb_front = item["observation.images.front"].permute(1, 2, 0).cpu().numpy()
    rgb_top = item["observation.images.top"].permute(1, 2, 0).cpu().numpy() if "observation.images.top" in item else np.zeros_like(rgb_front)
    
    # 確保是 uint8
    rgb_front = (rgb_front * 255).astype(np.uint8) if rgb_front.max() <= 1.5 else rgb_front.astype(np.uint8)
    rgb_top = (rgb_top * 255).astype(np.uint8) if rgb_top.max() <= 1.5 else rgb_top.astype(np.uint8)

    # 套用 Heatmap
    vis_front = apply_heatmap_overlay(rgb_front, heat_2d_front)
    vis_top = apply_heatmap_overlay(rgb_top, heat_2d_top)
    # def apply_heatmap_overlay(rgb_img, heat_map_2d):
    #     # [FIX] Use Percentile-based normalization to ignore "Register Artifacts"
    #     # This clips the top 2% of brightest pixels so outliers don't hide the real data
    #     v_min, v_max = np.percentile(heat_map_2d, [0, 98]) 
        
    #     # Clip values to this range
    #     heat_clipped = np.clip(heat_map_2d, v_min, v_max)
        
    #     # Normalize to 0-1
    #     if v_max - v_min > 1e-6:
    #         heat_norm = (heat_clipped - v_min) / (v_max - v_min)
    #     else:
    #         heat_norm = heat_clipped - v_min
            
    #     heatmap_color = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_JET)
    #     rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    #     return cv2.addWeighted(rgb_bgr, 0.6, heatmap_color, 0.4, 0)

    text_weights_list = attn_matrix[:, lang_start:lang_end].mean(dim=0).cpu().tolist()

    vis_front = apply_heatmap_overlay(rgb_front, heat_2d_front)
    vis_top = apply_heatmap_overlay(rgb_top, heat_2d_top)

    combined = np.hstack((vis_front, vis_top))

    final_frame = draw_prompt_weights_aligned(
        combined, 
        prompt, 
        text_weights_list
    )

    overlay = np.hstack((vis_front, vis_top))
    h, w, _ = vis_front.shape
    cv2.line(overlay, (w, 0), (w, h), (255, 255, 255), 2)
    cv2.putText(overlay, f"Front ({args.token or 'All'})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, "Top", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if video_writer is None:
        fh, fw, _ = final_frame.shape
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (fw, fh))
        print(f"[INFO] Video initialized: {fw}x{fh}")

    video_writer.write(final_frame)
    
    if current_frame % 30 == 0:
        print(f"Processing Frame {current_frame}/{ep_length}")

    current_frame += 1

if video_writer:
    video_writer.release()
    print(f"\n[DONE] Saved to {os.path.abspath(output_video_path)}")