'''
Attention Visualization Script — SmolVLM2-2B, 1 frame/second sampling
======================================================================

Key differences from record_attention_plot_smolvlm_stanley.py:
  - Uses SmolVLM2-2B-Instruct (num_vlm_layers=24) for stronger prompt sensitivity
  - Only processes 1 frame per second (skips the rest) → ~30× faster
  - Outputs per-second PNG heatmaps, NOT a continuous video
  - Block-level cross-modality PNGs (Image / Language / State) saved alongside

Usage:
    python src/lerobot/scripts/attn_vis_2b_per_second.py \
        --repo_id "ethanCSL/svla_koch_sorting_n_stacking" \
        --episode 0 \
        --prompt "Put the red cube in the right box, green cube in the left box." \
        --use_state

Output folders (created in the current working directory):
    attn_vis_2b_per_second_<with|no>_state/   ← heatmap PNGs (front + top side-by-side)
    attn_block_2b_per_second/                  ← 3×3 block-attention PNGs (only with --use_state)
'''

import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse
import torch.nn.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

# ──────────────────────────────────────────────────────────────────────────────
# 1. Argument Parsing
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visualize SmolVLM2-2B attention at 1 fps")
parser.add_argument("--repo_id",        type=str,  default="lerobot/svla_so100_pickplace")
parser.add_argument("--episode",        type=int,  default=0)
parser.add_argument("--prompt",         type=str,  default="grip the green block and put it into box")
parser.add_argument("--token",          type=str,  default=None,
                    help="Optional single word to highlight (attends only to that token)")
parser.add_argument("--use_state",      action="store_true",
                    help="Include joint-state tokens in the prefix (also saves block-attention plots)")
parser.add_argument("--video_backend",  type=str,  default="pyav")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Policy Config  — SmolVLM2-2B-Instruct
# ──────────────────────────────────────────────────────────────────────────────
# The 500M variant is too small to produce prompt-sensitive attention.
# SmolVLM2-2.2B uses SmolLM2-1.7B as its LLM backbone (24 transformer layers)
# and shows clearly different attention patterns for different prompts.
policy_cfg = SmolVLAConfig(
    vlm_model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    load_vlm_weights=True,
    freeze_vision_encoder=True,
    train_expert_only=True,
    train_state_proj=False,
    attention_mode="self_attn",
    device="cuda",
    empty_cameras=0,
    num_vlm_layers=24,
)
policy_cfg.normalization_mapping["STATE"] = NormalizationMode.IDENTITY
policy_cfg.input_features.update({
    "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    "observation.images.top":   PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
})

# ──────────────────────────────────────────────────────────────────────────────
# 3. Load Dataset
# ──────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading dataset:", args.repo_id)
dataset = LeRobotDataset(
    args.repo_id,
    root=None,
    batch_encoding_size=1,
    video_backend=args.video_backend,
)
dataset_fps = dataset.fps
print(f"[INFO] Dataset FPS = {dataset_fps}  →  will process 1 frame every {dataset_fps} frames (1 Hz)")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Load Policy
# ──────────────────────────────────────────────────────────────────────────────
policy = SmolVLAPolicy(config=policy_cfg, dataset_stats=None)
device = torch.device(policy_cfg.device)
policy = policy.to(device)
policy.reset()
policy.model.vlm_with_expert.debug_attn = True
print("[DEBUG] attention_mode:", policy.model.vlm_with_expert.attention_mode)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Episode range
# ──────────────────────────────────────────────────────────────────────────────
total_episodes = len(dataset.meta.episodes)
if not (0 <= args.episode < total_episodes):
    raise ValueError(f"Episode {args.episode} out of range [0, {total_episodes})")

ep_meta    = dataset.meta.episodes[args.episode]
ep_length  = ep_meta["length"]
start_idx  = sum(dataset.meta.episodes[i]["length"] for i in range(args.episode))
end_idx    = start_idx + ep_length

print(f"\n[INFO] Episode {args.episode}  |  frames {start_idx} – {end_idx}  |  length {ep_length}")
print(f"[INFO] Will process ≈ {ep_length // dataset_fps} frames (one per second)\n")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_token_indices(processor, prompt, specific_word=None):
    if isinstance(prompt, str) and not prompt.endswith("\n"):
        prompt = prompt + "\n"
    ids = processor.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    if specific_word is None:
        return list(range(len(ids)))
    words = processor.tokenizer.convert_ids_to_tokens(ids)
    indices = [i for i, w in enumerate(words) if specific_word.lower() in w.replace("Ġ", "").lower()]
    if not indices:
        raise ValueError(f"Token '{specific_word}' not found among: {words}")
    return indices


def extract_full_attention(policy):
    attn = getattr(policy.model.vlm_with_expert, "last_attn_weights", None)
    if attn is None:
        raise RuntimeError("No attention captured — is debug_attn=True?")
    attn_matrix      = attn[0].mean(0)                                     # [Q, K]
    num_img_tokens   = policy.model.vlm_with_expert._debug_num_img_tokens
    num_images       = policy.model.vlm_with_expert._debug_num_images
    total_img_tokens = num_img_tokens * num_images
    return attn_matrix, total_img_tokens


def process_heatmap(heat_1d, original_image_size=(480, 640), model_input_size=(512, 512)):
    """
    Reshapes 1D image attention tokens → 2D heatmap aligned to the original image,
    correctly undoing SmolVLA's square-padding preprocessing.
    """
    grid_size = int(math.sqrt(heat_1d.numel()))
    heat_1d   = heat_1d[: grid_size * grid_size]          # trim if not perfect square
    heat_2d   = heat_1d.reshape(grid_size, grid_size)

    heat_tensor = torch.tensor(heat_2d).unsqueeze(0).unsqueeze(0).float()
    heat_512    = F.interpolate(heat_tensor, size=model_input_size, mode="bilinear", align_corners=False)

    orig_h, orig_w = original_image_size
    tgt_h,  tgt_w  = model_input_size
    ratio      = max(orig_w / tgt_w, orig_h / tgt_h)
    resized_h  = int(orig_h / ratio)
    resized_w  = int(orig_w / ratio)
    pad_w      = max(0, tgt_w - resized_w)
    pad_h      = max(0, tgt_h - resized_h)

    # SmolVLA pads left+top: F.pad(img, (pad_w, 0, pad_h, 0))
    heat_valid = heat_512[0, 0, pad_h : pad_h + resized_h, pad_w : pad_w + resized_w]
    heat_final = F.interpolate(heat_valid.unsqueeze(0).unsqueeze(0), size=original_image_size, mode="bilinear")
    return heat_final[0, 0].numpy()


def apply_heatmap_overlay(rgb_img, heat_map_2d):
    """Blend a JET heatmap onto an RGB image using percentile normalisation."""
    v_min, v_max = np.percentile(heat_map_2d, [0, 98])
    heat_clipped = np.clip(heat_map_2d, v_min, v_max)
    heat_norm    = (heat_clipped - v_min) / (v_max - v_min + 1e-8)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_JET)
    rgb_bgr       = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(rgb_bgr, 0.6, heatmap_color, 0.4, 0)


def compute_deterministic_attention(policy, batch, device):
    """
    Runs only the image+text(+state) prefix encoding to get self-attention weights.
    Does NOT run the diffusion / action-generation loop.
    """
    policy.eval()

    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    task = batch.get("task", "")
    if isinstance(task, str) and not task.endswith("\n"):
        task = task + "\n"
        batch["task"] = task

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tok = tokenizer(
        task,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
    )
    batch[OBS_LANGUAGE_TOKENS]           = tok["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK]   = tok["attention_mask"].to(device).bool()

    images, img_masks = policy.prepare_images(batch)

    with torch.no_grad():
        num_img_tokens = int(policy.model.vlm_with_expert.embed_image(images[0]).shape[1])
    policy.model.vlm_with_expert._debug_num_img_tokens = num_img_tokens
    policy.model.vlm_with_expert._debug_num_images     = len(images)

    if OBS_STATE in batch:
        state = policy.prepare_state(batch)
    else:
        bsize = batch[OBS_LANGUAGE_TOKENS].shape[0]
        state = torch.zeros((bsize, policy.config.max_state_dim), device=device, dtype=torch.float32)

    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        state=state,
    )

    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    prefix_att_2d_masks  = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids  = torch.cumsum(prefix_pad_masks, dim=1) - 1

    with torch.no_grad():
        policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )


def save_block_attention(attn_np, img_start, img_end, lang_start, lang_end,
                         state_start, state_end, second_idx, out_folder):
    """Save the 3×3 cross-modality block-attention heatmap as a PNG."""
    def bm(r0, r1, c0, c1):
        return attn_np[r0:r1, c0:c1].mean() if r1 > r0 and c1 > c0 else 0.0

    block_mat = np.array([
        [bm(img_start,   img_end,   img_start,   img_end),
         bm(img_start,   img_end,   lang_start,  lang_end),
         bm(img_start,   img_end,   state_start, state_end)],
        [bm(lang_start,  lang_end,  img_start,   img_end),
         bm(lang_start,  lang_end,  lang_start,  lang_end),
         bm(lang_start,  lang_end,  state_start, state_end)],
        [bm(state_start, state_end, img_start,   img_end),
         bm(state_start, state_end, lang_start,  lang_end),
         bm(state_start, state_end, state_start, state_end)],
    ])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(block_mat, cmap="viridis", vmin=0.0, vmax=0.1)
    fig.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_xticks([0, 1, 2]);  ax.set_xticklabels(["Key:Image", "Key:Language", "Key:State"])
    ax.set_yticks([0, 1, 2]);  ax.set_yticklabels(["Query:Image", "Query:Language", "Query:State"])
    ax.set_xlabel("Key (what is attended to)")
    ax.set_ylabel("Query (who is attending)")
    ax.set_title(f"Block-Level Self-Attention (second {second_idx})")
    fig.tight_layout()

    path = os.path.join(out_folder, f"block_attn_sec_{second_idx:04d}.png")
    fig.savefig(path)
    plt.close(fig)
    return path

# ──────────────────────────────────────────────────────────────────────────────
# 7. Output Folders
# ──────────────────────────────────────────────────────────────────────────────
mode_suffix = "with_state" if args.use_state else "no_state"
heatmap_dir = f"attn_vis_2b_per_second_{mode_suffix}"
block_dir   = "attn_block_2b_per_second"
os.makedirs(heatmap_dir, exist_ok=True)
if args.use_state:
    os.makedirs(block_dir, exist_ok=True)

# Pre-compute which token indices to average over
processor      = policy.model.vlm_with_expert.processor
target_indices = get_token_indices(processor, args.prompt, args.token)

# ──────────────────────────────────────────────────────────────────────────────
# 8. Main Loop — 1 frame per second
# ──────────────────────────────────────────────────────────────────────────────
current_frame = 0
saved_count   = 0

for global_idx in range(start_idx, end_idx):

    # ── Skip frames that are not on a 1-second boundary ──────────────────────
    if current_frame % dataset_fps != 0:
        current_frame += 1
        continue

    second_idx = current_frame // dataset_fps
    item = dataset[global_idx]

    # ── Build batch ───────────────────────────────────────────────────────────
    batch = {}
    for k, v in item.items():
        if k in ("observation.images.front", "observation.images.top"):
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            batch[k] = v.unsqueeze(0).to(device)
        elif k == "observation.state" and args.use_state:
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            batch[OBS_STATE] = v.unsqueeze(0).to(device)
    batch["task"] = args.prompt

    # ── Forward pass (prefix only) ────────────────────────────────────────────
    compute_deterministic_attention(policy, batch, device)

    # ── Extract attention ─────────────────────────────────────────────────────
    attn_matrix, total_img_tokens = extract_full_attention(policy)

    num_lang_tokens  = batch[OBS_LANGUAGE_TOKENS].shape[1]
    num_state_tokens = batch[OBS_STATE].shape[1] if OBS_STATE in batch else 0

    img_start   = 0
    img_end     = total_img_tokens
    lang_start  = img_end
    lang_end    = lang_start + num_lang_tokens
    state_start = lang_end
    state_end   = state_start + num_state_tokens

    # ── Block-level cross-modality plot (only when --use_state) ──────────────
    if args.use_state and num_state_tokens > 0:
        path = save_block_attention(
            attn_matrix.detach().cpu().numpy(),
            img_start, img_end, lang_start, lang_end, state_start, state_end,
            second_idx, block_dir,
        )
        print(f"  [block] {path}")

    # ── Language→Image heatmap ────────────────────────────────────────────────
    # Average attention over the selected language query tokens
    lang_query           = attn_matrix[lang_start:lang_end].mean(0)
    num_img_tokens_single = policy.model.vlm_with_expert._debug_num_img_tokens

    heat_front_1d = lang_query[img_start               : img_start + num_img_tokens_single]
    heat_top_1d   = lang_query[img_start + num_img_tokens_single : img_end]

    heat_2d_front = process_heatmap(heat_front_1d)
    heat_2d_top   = process_heatmap(heat_top_1d)

    # ── Retrieve RGB images ───────────────────────────────────────────────────
    rgb_front = item["observation.images.front"].permute(1, 2, 0).cpu().numpy()
    rgb_top   = item["observation.images.top"].permute(1, 2, 0).cpu().numpy() \
                if "observation.images.top" in item \
                else np.zeros_like(rgb_front)

    if rgb_front.max() <= 1.5:
        rgb_front = (rgb_front * 255).astype(np.uint8)
        rgb_top   = (rgb_top   * 255).astype(np.uint8)
    else:
        rgb_front = rgb_front.astype(np.uint8)
        rgb_top   = rgb_top.astype(np.uint8)

    # ── Compose side-by-side overlay ─────────────────────────────────────────
    vis_front = apply_heatmap_overlay(rgb_front, heat_2d_front)
    vis_top   = apply_heatmap_overlay(rgb_top,   heat_2d_top)

    overlay = np.hstack((vis_front, vis_top))
    h, w, _ = vis_front.shape
    cv2.line(overlay, (w, 0), (w, h), (255, 255, 255), 2)

    label = args.token or "All tokens"
    cv2.putText(overlay, f"Front ({label})", (10,    30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, f"Top   ({label})", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, f"sec {second_idx:04d} | frame {current_frame}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(overlay, f'"{args.prompt[:60]}"', (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)

    png_path = os.path.join(heatmap_dir, f"attn_sec_{second_idx:04d}.png")
    cv2.imwrite(png_path, overlay)
    saved_count += 1
    print(f"[INFO] sec {second_idx:04d} (frame {current_frame:05d}) → {png_path}")

    current_frame += 1

print(f"\n[DONE] Saved {saved_count} heatmap PNGs → {os.path.abspath(heatmap_dir)}")
if args.use_state:
    print(f"[DONE] Block-attention PNGs        → {os.path.abspath(block_dir)}")
