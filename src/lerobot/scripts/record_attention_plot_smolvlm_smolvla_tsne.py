'''
This is the t-SNE visualization script replicating Figure 5 of "Don't Blind Your VLA".
(Reference: https://arxiv.org/pdf/2510.25616)

Methodology (matching the paper):
  - Run across MULTIPLE episodes to get diverse visual contexts per token class.
  - Probe specific object/color tokens (e.g. "red", "green") across all frames.
  - Each (frame, token) pair becomes one point in t-SNE, colored by token class.
  - Two rows: SmolVLM base vs SmolVLA fine-tuned — if fine-tuning collapses
    representations (like OpenVLA), clusters will overlap in SmolVLA row.

Methodology (matching the paper):
  - Run across MULTIPLE episodes from MULTIPLE tasks to get diverse visual contexts.
  - Probe specific object/color tokens across all frames; each word is one color class.
  - Two rows: SmolVLM base vs SmolVLA fine-tuned.
    Well-separated clusters = preserved grounding (Qwen/Prismatic style).
    Overlapping clusters    = representation collapse (OpenVLA style).

  IDEAL SETUP (this dataset):
    Episode   0-99:  sorting task  - "red"=right-box, "green"=left-box
    Episode 100-199: stacking task - "green" on top of "red"
    Probing "red"/"green" across BOTH tasks is the strongest test:
    if grounding is preserved, all "red" tokens cluster together regardless of task.

Usage - multi-task replication (RECOMMENDED):
python src/lerobot/scripts/record_attention_plot_smolvlm_smolvla_tsne.py \
    --repo_id "ethanCSL/svla_koch_sorting_n_stacking" \
    --ckpt    "ethanCSL/svla_koch_sorting_n_stacking_vision_encoder_unfrozen_train_expert_only_false" \
    --tasks   '[{"episodes": "0,2,4,6,8,10", "prompt": "put the red cube in the right box,and green cube in the left box."}, {"episodes": "100,102,104,106,108,110", "prompt": "put the green cube on top of red cube"}]' \
    --tokens  "red,green" \
    --frame_stride 3

Usage - single-task fallback:
python src/lerobot/scripts/record_attention_plot_smolvlm_smolvla_tsne.py \
    --repo_id  "ethanCSL/svla_koch_sorting_n_stacking" \
    --ckpt     "ethanCSL/svla_koch_sorting_n_stacking_vision_encoder_unfrozen_train_expert_only_false" \
    --episodes "0,1,2,3,4,5" \
    --prompt   "put the red cube in the right box,and green cube in the left box." \
    --tokens   "red,green" \
    --frame_stride 3
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
parser = argparse.ArgumentParser(description="Replicate Don't-Blind-Your-VLA t-SNE (Figure 5)")
parser.add_argument("--repo_id",       type=str, default="lerobot/svla_so100_pickplace")
# Multi-task mode (RECOMMENDED for paper replication)
parser.add_argument("--tasks",         type=str, default=None,
                    help='JSON list: [{"episodes": "0,1,2", "prompt": "..."}, ...]. '
                         'Each task has its own episodes AND prompt. Overrides --episodes/--prompt.')
# Single-task fallback
parser.add_argument("--episodes",      type=str, default="0",
                    help="Comma-separated episode indices (single-task fallback).")
parser.add_argument("--prompt",        type=str, default="grip the green block and put it into box",
                    help="Task prompt (single-task fallback).")
# Shared
parser.add_argument("--tokens",        type=str, default="red,green",
                    help="Comma-separated token words to probe (one color class each in t-SNE).")
parser.add_argument("--frame_stride",  type=int, default=3,
                    help="Sample every Nth frame. Lower = more points, slower.")
parser.add_argument("--use_state",     action="store_true")
parser.add_argument("--video_backend", type=str, default="pyav")
parser.add_argument("--ckpt",          type=str, default=None,
                    help="Unfrozen SmolVLA checkpoint (train_expert_only=False, freeze_vision_encoder=False).")

args = parser.parse_args()

import json as _json

TOKEN_WORDS = [t.strip() for t in args.tokens.split(",") if t.strip()]

# Build task_configs: list of {"episodes": [int, ...], "prompt": str}
if args.tasks is not None:
    task_configs = [
        {
            "episodes": [int(e.strip()) for e in tc["episodes"].split(",") if e.strip()],
            "prompt":   tc["prompt"],
        }
        for tc in _json.loads(args.tasks)
    ]
else:
    task_configs = [{
        "episodes": [int(e.strip()) for e in args.episodes.split(",") if e.strip()],
        "prompt":   args.prompt,
    }]

all_task_episode_indices = [ep for tc in task_configs for ep in tc["episodes"]]

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
# Force self_attn for prefix encoding (checkpoint may have been trained with cross_attn;
# hidden_per_layer is only populated during the VLM self-attention forward pass)
vla_policy.model.vlm_with_expert.attention_mode = "self_attn"
vla_policy.model.vlm_with_expert.debug_attn = True
policies["SmolVLA"] = vla_policy
# Note: blind variants removed for clean 2-row paper replication (Fig. 5 style).
# SmolVLM = pre-fine-tune baseline; SmolVLA = post-fine-tune subject.

# 2. Build episode frame ranges
total_episodes = len(dataset.meta.episodes)
for ep_idx in all_task_episode_indices:
    if ep_idx < 0 or ep_idx >= total_episodes:
        raise ValueError(f"Episode index {ep_idx} out of range [0, {total_episodes-1}]")

_cum = 0
_ep_starts = []
for ep in dataset.meta.episodes:
    _ep_starts.append(_cum)
    _cum += ep['length']

# task_episode_ranges[i] = list of (ep_idx, start, end, prompt) for task i
task_episode_ranges = []
for tc in task_configs:
    ranges = []
    for ep_idx in tc["episodes"]:
        s = _ep_starts[ep_idx]
        e = s + dataset.meta.episodes[ep_idx]['length']
        ranges.append((ep_idx, s, e, tc["prompt"]))
    task_episode_ranges.append(ranges)

all_episode_ranges = [r for trs in task_episode_ranges for r in trs]
total_frames = sum(e - s for _, s, e, _ in all_episode_ranges)

print(f"\n[INFO] Tasks configured: {len(task_configs)}")
for i, tc in enumerate(task_configs):
    print(f"  Task {i}: episodes={tc['episodes']}")
    print(f"           prompt='{tc['prompt']}'")
print(f"[INFO] Token words: {TOKEN_WORDS}")
print(f"[INFO] Frame stride: {args.frame_stride}")
print(f"[INFO] Total frames: {total_frames}  |  ~{total_frames // args.frame_stride} points per token per model")
print(f"[INFO] Approx points per token per model: {total_frames // args.frame_stride}")

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
# 4. Main Loop
# =========================

processor = policies["SmolVLM"].model.vlm_with_expert.processor

# Build per-prompt token index maps (positions differ between prompts)
def build_token_idx_map(prompt):
    idx_map = {}
    for word in TOKEN_WORDS:
        idxs = get_token_indices(processor, prompt, word)
        idx_map[word] = idxs[0]
    return idx_map

prompt_token_maps = {}
for tc in task_configs:
    p = tc["prompt"]
    if p not in prompt_token_maps:
        prompt_token_maps[p] = build_token_idx_map(p)
        print(f"[INFO] Prompt: '{p}'")
        for word, pos in prompt_token_maps[p].items():
            print(f"         token '{word}' -> position {pos}")

LAYER_IDS = [0, 7, 15]

all_features = {
    model_name: {lid: [] for lid in LAYER_IDS}
    for model_name in policies
}
all_labels = {model_name: [] for model_name in policies}

block_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_tsne_paper")
os.makedirs(block_dir, exist_ok=True)

for model_name, policy in policies.items():
    print(f"\n[INFO] Running {model_name} ({len(all_episode_ranges)} episodes, {len(task_configs)} tasks)...")
    frame_count = 0

    for task_ranges in task_episode_ranges:
        for ep_idx, ep_start, ep_end, ep_prompt in task_ranges:
            tok_map = prompt_token_maps[ep_prompt]
            ep_frame = 0

            for global_idx in range(ep_start, ep_end):
                if ep_frame % args.frame_stride != 0:
                    ep_frame += 1
                    continue

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

                # Use this episode's specific task prompt
                observation_batch["task"] = ep_prompt

                compute_deterministic_attention(policy, observation_batch, device)

                _, total_img_tokens = extract_full_attention(policy)
                lang_start = total_img_tokens

                for layer_id in LAYER_IDS:
                    hidden = policy.model.vlm_with_expert.hidden_per_layer[layer_id]
                    for word, tok_pos in tok_map.items():
                        rep = hidden[0, lang_start + tok_pos, :].detach().float().cpu()
                        all_features[model_name][layer_id].append(rep)

                # Label = token word (color class), regardless of which task
                all_labels[model_name].extend(list(tok_map.keys()))

                frame_count += 1
                ep_frame += 1

    print(f"[INFO] {model_name}: {frame_count} frames x {len(TOKEN_WORDS)} tokens = {frame_count * len(TOKEN_WORDS)} points per layer")

# ==============================================================
# t-SNE visualization — Figure 5 style (paper replication)
# 2 rows (SmolVLM base | SmolVLA fine-tuned) x 3 columns (layers)
# Color = token class (red/green, or whatever --tokens specifies)
# ==============================================================
print("\nFeature sizes:")
for m in all_features:
    for l in LAYER_IDS:
        print(f"  {m}  layer={l}  points={len(all_features[m][l])}")

import matplotlib.patches as mpatches

# Assign a distinct color to each token word (matches paper's cup/bottle/knife colors)
_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
token_color_map = {word: _PALETTE[i % len(_PALETTE)] for i, word in enumerate(TOKEN_WORDS)}
legend_handles = [
    mpatches.Patch(color=token_color_map[w], label=f"'{w}' token")
    for w in TOKEN_WORDS
]

ROW_MODELS = ["SmolVLM", "SmolVLA"]
ROW_LABELS = [
    "SmolVLM (base, pre-fine-tune)",
    "SmolVLA (fine-tuned, unfrozen backbone)",
]

fig, axes = plt.subplots(len(ROW_MODELS), len(LAYER_IDS), figsize=(5 * len(LAYER_IDS), 4 * len(ROW_MODELS)))
if len(ROW_MODELS) == 1:
    axes = axes[np.newaxis, :]  # ensure 2D indexing

for row, model_name in enumerate(ROW_MODELS):
    axes[row, 0].set_ylabel(ROW_LABELS[row], fontsize=10, labelpad=8)

for col, layer_id in enumerate(LAYER_IDS):

    # ── Stack features from both models jointly into one t-SNE embedding
    # so the coordinate spaces are comparable across rows (same as paper)
    per_model_feats = [
        torch.stack(all_features[m][layer_id]).float().numpy()
        for m in ROW_MODELS
    ]
    features_all = np.concatenate(per_model_feats, axis=0)
    labels_all   = []
    for m in ROW_MODELS:
        labels_all.extend(all_labels[m])

    n_samples  = len(features_all)
    perplexity = min(30, max(5, n_samples // (len(ROW_MODELS) * len(TOKEN_WORDS)) - 1))
    print(f"[t-SNE] layer={layer_id}  n={n_samples}  perplexity={perplexity}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
        max_iter=1000,
    )
    emb_all = tsne.fit_transform(features_all)

    # Split back per model
    offset = 0
    for row, model_name in enumerate(ROW_MODELS):
        n = len(per_model_feats[row])
        emb  = emb_all[offset : offset + n]
        labs = labels_all[offset : offset + n]
        offset += n

        # Scatter each token class separately so colors are correct
        for word in TOKEN_WORDS:
            mask = np.array([l == word for l in labs])
            axes[row, col].scatter(
                emb[mask, 0], emb[mask, 1],
                c=token_color_map[word],
                s=6, alpha=0.7, linewidths=0,
            )

        axes[row, col].set_title(f"{model_name} — Layer {layer_id + 1}", fontsize=10)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
state_start
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(TOKEN_WORDS),
    fontsize=11,
    bbox_to_anchor=(0.5, -0.04),
)
task_summary = "  |  ".join(
    f"Task {i} ep{tc['episodes'][0]}-{tc['episodes'][-1]}: '{tc['prompt'][:35]}...'"
    for i, tc in enumerate(task_configs)
)
plt.suptitle(
    f"t-SNE of language-token hidden representations\n"
    f"{task_summary}\n"
    f"Tokens: {TOKEN_WORDS}  Stride: {args.frame_stride}  "
    f"| Well-separated = preserved grounding | Overlap = collapse (OpenVLA-style)",
    fontsize=9, y=1.02,
)
plt.tight_layout()
out_path = os.path.join(block_dir, "tsne_dontblind_replication.png")
plt.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"\n[DONE] Saved to {out_path}")
plt.show()