'''
This is the t-SNE visualization script for the SmolVLA attention analysis in the paper.
(Reference:Don't blind your VLA:https://arxiv.org/pdf/2510.25616)
It extracts the hidden representations of specific language tokens (e.g. "red", "green") 
from both the baseline SmolVLM and the fine-tuned SmolVLA, then visualizes how these token representations cluster in 2D space across different layers.

Usage:
python src/lerobot/scripts/record_attention_plot_smolvlm_smolvla_tsne.py     --repo_id "ethanCSL/svla_koch_sorting_n_stacking"          --episode 0     --prompt "put the red cube in the right box,and green cube in the left box."  --ckpt ethanCSL/svla_koch_sorting_n_stacking --use_state

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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Initialization & Argument Parsing
parser = argparse.ArgumentParser(description="Visualize Attention Maps for SmolVLA")
parser.add_argument("--repo_id", type=str, default="lerobot/svla_so100_pickplace", help="HuggingFace Dataset Repo ID")
parser.add_argument("--episode", type=int, default=10, help="Episode index to visualize")
parser.add_argument("--prompt", type=str, default="grip the green block and put it into box", help="Task prompt")
parser.add_argument("--token", type=str, default=None, help="(Optional) Specific word to visualize.")
parser.add_argument("--use_state", action="store_true", help="Condition attention on joint states")
parser.add_argument("--video_backend", type=str, default="pyav")
parser.add_argument("--ckpt", type=str, default=None,
                    help="Trained SmolVLA checkpoint repo or path")

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

device = torch.device(policy_cfg.device)

policies = {}

# ---------- SmolVLM baseline ----------
vlm_policy = SmolVLAPolicy(
    config=policy_cfg,
    dataset_stats=None,
)
vlm_policy = vlm_policy.to(device)
vlm_policy.reset()
vlm_policy.model.vlm_with_expert.debug_attn = True
policies["SmolVLM"] = vlm_policy

# ---------- Trained SmolVLA ----------
if args.ckpt is None:
    raise ValueError("You must provide --ckpt for SmolVLA comparison")

vla_policy = SmolVLAPolicy.from_pretrained(args.ckpt)
# Override image features to match the dataset camera keys
vla_policy.config.input_features.update({
    "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    "observation.images.top":   PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
})
# Remove checkpoint-specific camera keys that don't exist in this dataset
for key in ["observation.images.camera1", "observation.images.camera2",
            "observation.images.camera3", "observation.images.empty_camera_0"]:
    vla_policy.config.input_features.pop(key, None)
vla_policy = vla_policy.to(device)
vla_policy.reset()
vla_policy.model.vlm_with_expert.debug_attn = True
policies["SmolVLA"] = vla_policy
policies["SmolVLA_Blind"]       = vla_policy  # same weights — ALL cameras zeroed
policies["SmolVLA_Blind_Top"]   = vla_policy  # same weights — only TOP camera zeroed
policies["SmolVLA_Blind_Front"] = vla_policy  # same weights — only FRONT camera zeroed

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

    attn_matrix = attn[0].mean(0)

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

PROMPT = args.prompt
print("USING PROMPT:", PROMPT)

processor = policies["SmolVLM"].model.vlm_with_expert.processor

red_idx   = get_token_indices(processor, PROMPT, "red")
green_idx = get_token_indices(processor, PROMPT, "green")

print(f"[INFO] Start processing Frames...")
LAYER_IDS = [0, 7, 15]
all_features = {
    "SmolVLM":           {lid: [] for lid in LAYER_IDS},
    "SmolVLA":           {lid: [] for lid in LAYER_IDS},
    "SmolVLA_Blind":     {lid: [] for lid in LAYER_IDS},
    "SmolVLA_Blind_Top": {lid: [] for lid in LAYER_IDS},
    "SmolVLA_Blind_Front": {lid: [] for lid in LAYER_IDS},
}
all_labels = []
current_frame = 0

# ===== Create output folder =====
# Saves next to the script file regardless of where the command is run from.
block_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_block_every30")
os.makedirs(block_dir, exist_ok=True)

for model_name, policy in policies.items():

    print(f"\n[INFO] Running {model_name}")
    current_frame = 0

    for global_idx in range(start_idx, end_idx):

        item = dataset[global_idx]
        observation_batch = {}

        for k, v in item.items():
            if k in ("observation.images.front", "observation.images.top"):
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                if model_name == "SmolVLA_Blind":
                    v = torch.zeros_like(v)          # all cameras zeroed
                elif model_name == "SmolVLA_Blind_Top" and k == "observation.images.top":
                    v = torch.zeros_like(v)          # only top camera zeroed
                elif model_name == "SmolVLA_Blind_Front" and k == "observation.images.front":
                    v = torch.zeros_like(v)          # only front camera zeroed
                observation_batch[k] = v.unsqueeze(0).to(device)

            elif k == "observation.state" and args.use_state:
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                observation_batch[OBS_STATE] = v.unsqueeze(0).to(device)

        observation_batch["task"] = PROMPT

        compute_deterministic_attention(policy, observation_batch, device)

        attn_matrix, total_img_tokens = extract_full_attention(policy)

        img_start = 0
        img_end = total_img_tokens
        lang_start = img_end

        # Instead of weighted_pool, do this:
        for layer_id in LAYER_IDS:
            hidden = policy.model.vlm_with_expert.hidden_per_layer[layer_id]
            # Take the hidden state AT the language token position, not a pooled image vector
            red_rep   = hidden[0, lang_start + red_idx[0],   :].detach().float().cpu()
            green_rep = hidden[0, lang_start + green_idx[0], :].detach().float().cpu()

            all_features[model_name][layer_id].append(red_rep)
            all_features[model_name][layer_id].append(green_rep)

        # 只在 baseline 加 label 一次
        if model_name == "SmolVLM":
            all_labels.extend(["red", "green"])

        current_frame += 1

# ==================================
# t-SNE visualization (2x3 compare)
# ==================================
print("Feature sizes:")
for m in all_features:
    for l in LAYER_IDS:
        print(m, l, len(all_features[m][l]))

# ── Legend patch ──────────────────────────────────────────────────────────────
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color="red",   label="'red' token"),
    mpatches.Patch(color="green", label="'green' token"),
]

fig, axes = plt.subplots(5, 3, figsize=(15, 20))
# Row labels
row_titles = [
    "SmolVLM (base, with vision)",
    "SmolVLA (fine-tuned, with vision)",
    "SmolVLA Blind (all cameras zeroed)",
    "SmolVLA Blind Top (top camera zeroed)",
    "SmolVLA Blind Front (front camera zeroed)",
]
for row, title in enumerate(row_titles):
    axes[row, 0].set_ylabel(title, fontsize=9, labelpad=6)

for col, layer_id in enumerate(LAYER_IDS):

    # Extract features (cast bfloat16 → float32; numpy doesn't support bf16)
    features_vlm         = torch.stack(all_features["SmolVLM"][layer_id]).float().numpy()
    features_vla         = torch.stack(all_features["SmolVLA"][layer_id]).float().numpy()
    features_blind       = torch.stack(all_features["SmolVLA_Blind"][layer_id]).float().numpy()
    features_blind_top   = torch.stack(all_features["SmolVLA_Blind_Top"][layer_id]).float().numpy()
    features_blind_front = torch.stack(all_features["SmolVLA_Blind_Front"][layer_id]).float().numpy()

    # Joint t-SNE space for all five so geometry is comparable across rows
    features_all = np.concatenate(
        [features_vlm, features_vla, features_blind, features_blind_top, features_blind_front], axis=0
    )

    n_samples = len(features_all)
    perplexity = min(30, n_samples // 5 - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        init="pca",
        random_state=42
    )

    emb_all = tsne.fit_transform(features_all)

    n_vlm         = len(features_vlm)
    n_vla         = len(features_vla)
    n_blind       = len(features_blind)
    n_blind_top   = len(features_blind_top)
    emb_vlm         = emb_all[:n_vlm]
    emb_vla         = emb_all[n_vlm : n_vlm + n_vla]
    emb_blind       = emb_all[n_vlm + n_vla : n_vlm + n_vla + n_blind]
    emb_blind_top   = emb_all[n_vlm + n_vla + n_blind : n_vlm + n_vla + n_blind + n_blind_top]
    emb_blind_front = emb_all[n_vlm + n_vla + n_blind + n_blind_top :]

    colors = ["red", "green"] * (len(features_vlm) // 2)

    # Row 0: SmolVLM — baseline VLM, real images
    axes[0, col].scatter(emb_vlm[:, 0],         emb_vlm[:, 1],         c=colors, s=8)
    axes[0, col].set_title(f"SmolVLM — Layer {layer_id + 1}")

    # Row 1: SmolVLA — fine-tuned, real images
    axes[1, col].scatter(emb_vla[:, 0],         emb_vla[:, 1],         c=colors, s=8)
    axes[1, col].set_title(f"SmolVLA — Layer {layer_id + 1}")

    # Row 2: SmolVLA Blind — all cameras zeroed
    axes[2, col].scatter(emb_blind[:, 0],       emb_blind[:, 1],       c=colors, s=8)
    axes[2, col].set_title(f"SmolVLA Blind (all) — Layer {layer_id + 1}")

    # Row 3: SmolVLA Blind Top — only top camera zeroed
    axes[3, col].scatter(emb_blind_top[:, 0],   emb_blind_top[:, 1],   c=colors, s=8)
    axes[3, col].set_title(f"SmolVLA Blind Top — Layer {layer_id + 1}")

    # Row 4: SmolVLA Blind Front — only front camera zeroed
    axes[4, col].scatter(emb_blind_front[:, 0], emb_blind_front[:, 1], c=colors, s=8)
    axes[4, col].set_title(f"SmolVLA Blind Front — Layer {layer_id + 1}")

fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle(
    "t-SNE of language-token representations\n"
    "Row 0: SmolVLM base  |  Row 1: SmolVLA full vision  |  Row 2: all cameras blind\n"
    "Row 3: top camera blind  |  Row 4: front camera blind",
    fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(block_dir, "tsne_compare_smolvlm_vs_smolvla_blind.png"), bbox_inches="tight")
plt.show()