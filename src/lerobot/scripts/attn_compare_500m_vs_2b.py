'''
Side-by-side Attention Comparison: SmolVLM2-500M vs SmolVLM2-2.2B
==================================================================

Loads BOTH models and runs them on the same frame (1 per second).
Outputs a 4-panel PNG per second:

  [ 500M: Front | 500M: Top | 2.2B: Front | 2.2B: Top ]

This lets you directly compare how much each model's attention
pattern actually changes with different prompts.

GPU memory needed: ~1 GB (500M) + ~5 GB (2.2B) ≈ 6 GB total.
If you run out of VRAM, add --offload to move 500M to CPU between calls.

Usage:
    python src/lerobot/scripts/attn_compare_500m_vs_2b.py \
        --repo_id "ethanCSL/svla_koch_sorting_n_stacking" \
        --episode 0 \
        --prompt "Put the red cube in the right box, green cube in the left box." \
        --use_state

Output:
    attn_compare_500m_vs_2b/compare_sec_NNNN.png
'''

import torch
import math
import numpy as np
import cv2
import os
import argparse
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

# ──────────────────────────────────────────────────────────────────────────────
# 1. Args
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Compare 500M vs 2.2B SmolVLM2 attention at 1 fps")
parser.add_argument("--repo_id",       type=str, default="lerobot/svla_so100_pickplace")
parser.add_argument("--episode",       type=int, default=0)
parser.add_argument("--prompt",        type=str, default="grip the green block and put it into box")
parser.add_argument("--token",         type=str, default=None,
                    help="Optional single word to focus on")
parser.add_argument("--use_state",     action="store_true")
parser.add_argument("--video_backend", type=str, default="pyav")
parser.add_argument("--offload",       action="store_true",
                    help="Keep 500M on CPU between calls to save VRAM (slower)")
parser.add_argument("--camera",        type=str, default="both",
                    choices=["front", "top", "both"],
                    help="Which camera(s) to feed into the model and visualize. "
                         "'front' or 'top' feeds only that single camera, eliminating "
                         "cross-camera attention leakage. 'both' = original behaviour.")
parser.add_argument("--vlm_direct",    action="store_true",
                    help="Add a panel showing SmolVLM2 loaded DIRECTLY (no SmolVLAPolicy "
                         "wrapper), averaging attention over middle layers (40-75%% depth). "
                         "This is the 'VLM only' baseline from the Don't Blind Your VLA "
                         "paper — before any action fine-tuning is applied.")
parser.add_argument("--vlm_direct_model", type=str,
                    default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                    help="Which SmolVLM2 model to use for the direct VLM baseline")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ordered list of active cameras and their dataset keys
CAMERA_KEYS = {"front": "observation.images.front",
               "top":   "observation.images.top"}
ACTIVE_CAMERAS = ["front", "top"] if args.camera == "both" else [args.camera]
print(f"[INFO] Active camera(s): {ACTIVE_CAMERAS}")

# Sequential offloading: one model on GPU at a time.
# Auto-enabled when --vlm_direct is set (3 models can't fit simultaneously).
_force_offload = args.offload or args.vlm_direct
if _force_offload:
    print("[INFO] Offload mode: models will be moved to GPU only during their forward pass")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Shared helper to build a SmolVLAConfig + policy
# ──────────────────────────────────────────────────────────────────────────────
def build_policy(vlm_model_name: str, num_vlm_layers: int) -> SmolVLAPolicy:
    cfg = SmolVLAConfig(
        vlm_model_name=vlm_model_name,
        load_vlm_weights=True,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=False,
        attention_mode="self_attn",
        device=str(device),
        empty_cameras=0,
        num_vlm_layers=num_vlm_layers,
    )
    cfg.normalization_mapping["STATE"] = NormalizationMode.IDENTITY
    # Only register active cameras — feeding an unused camera creates attention
    # over blank/zero tokens which produces misleading hotspots on empty areas
    cfg.input_features.update({
        CAMERA_KEYS[cam]: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        for cam in ACTIVE_CAMERAS
    })
    policy = SmolVLAPolicy(config=cfg, dataset_stats=None)
    if not _force_offload:
        policy = policy.to(device)  # keep on CPU if offloading
    policy.reset()
    policy.model.vlm_with_expert.debug_attn = True
    return policy

# ──────────────────────────────────────────────────────────────────────────────
# 2b. SmolVLM2 direct extractor  ("VLM only" baseline per Don't Blind Your VLA)
# ──────────────────────────────────────────────────────────────────────────────
# The paper shows that a PRETRAINED VLM (not action-fine-tuned) produces
# sharp, object-aligned attention — especially in its MIDDLE layers (~40-75%
# depth), where vision-language fusion is most active (Figure 4, Section 5.1).
#
# The SmolVLAPolicy wrapper only captures the LAST layer and runs the model
# through a custom prefix-embedding path.  This class loads SmolVLM2 directly
# via transformers, runs a normal VLM forward pass with output_attentions=True,
# and averages over the middle layers — giving the cleanest "VLM only" result.
class SmolVLMDirectExtractor:
    def __init__(self, model_name: str, device, load_on_cpu: bool = False):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        print(f"[INFO] Loading {model_name} (direct VLM, no SmolVLAPolicy) ...")
        self.device = device
        load_device = "cpu" if load_on_cpu else str(device)
        # eager attention required for output_attentions=True
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=load_device,
            attn_implementation="eager",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        n = len(self.model.model.text_model.layers)
        # Middle layers: 40-75% depth (paper: strongest grounding here)
        lo = max(0, int(n * 0.40))
        hi = min(n, int(n * 0.75))
        self.layer_range = (lo, hi)
        # image_token_id: try model config first, fall back to tokenizer
        self.image_token_id = getattr(self.model.config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        print(f"  {n} layers total → using layers {lo}-{hi-1} "
              f"(middle 40-75%),  image_token_id={self.image_token_id}")

    @torch.no_grad()
    def get_heatmap(self, rgb_np: np.ndarray, prompt: str):
        """
        Returns (heat_1d, grid_hw) — same contract as QwenVLAttentionExtractor.
        SmolVLM2 pads images to 512x512 before patching so the grid is square.
        """
        pil_img = PILImage.fromarray(rgb_np)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text   = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            images=[pil_img], text=[text],
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_attentions=True)

        # ── locate image vs text positions ───────────────────────────────
        # SmolVLM2 uses a single <image> placeholder per image in input_ids,
        # but EXPANDS it into many visual tokens internally before the
        # transformer layers.  The attention matrices are therefore larger
        # than input_ids — e.g. input_ids has 50 tokens but attention is
        # 800×800 because one <image> expanded to ~750 visual tokens.
        #
        # Strategy:
        #  1. Find how many visual tokens exist by comparing the attention
        #     sequence length to input_ids length.
        #  2. SmolVLM2 always places image tokens FIRST in the sequence.
        #  3. Everything after the visual block = text tokens.
        input_ids  = inputs["input_ids"][0]                    # [n_ids]
        n_ids      = input_ids.shape[0]
        # actual sequence length seen by the transformer
        attn_seq   = outputs.attentions[0].shape[-1]           # e.g. 800
        n_img_toks = attn_seq - n_ids                          # visual expansion

        if n_img_toks <= 0:
            # fallback: no expansion detected, find <image> placeholders directly
            img_mask   = (input_ids == self.image_token_id)
            img_pos    = img_mask.nonzero(as_tuple=True)[0]
            text_pos   = (~img_mask).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                raise RuntimeError(
                    f"No image tokens (id={self.image_token_id}) found and "
                    f"attn_seq==n_ids ({attn_seq}). Check model/processor.")
        else:
            # visual tokens come first, text tokens follow
            img_pos  = torch.arange(n_img_toks, device=self.device)
            text_pos = torch.arange(n_img_toks, attn_seq, device=self.device)

        print(f"  [dbg] input_ids={n_ids}, attn_seq={attn_seq}, "
              f"img_toks={len(img_pos)}, text_toks={len(text_pos)}")

        # ── average attention over middle layers and heads ────────────────
        lo, hi = self.layer_range
        mid_attns = [outputs.attentions[i][0].float().mean(0)   # [Q, K]
                     for i in range(lo, hi)]
        avg_attn  = torch.stack(mid_attns).mean(0)              # [seq, seq]

        # language queries attending to image keys
        lang_to_img = avg_attn[text_pos[:, None], img_pos[None, :]]  # [n_text, n_img]
        heat_1d     = lang_to_img.mean(0).cpu()                 # [n_img]

        # SmolVLM2 always pads to 512×512 → square token grid
        gs      = int(math.sqrt(len(img_pos)))
        grid_hw = (gs, gs)

        return heat_1d, grid_hw


# ──────────────────────────────────────────────────────────────────────────────
# 3. Load both models
# ──────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Loading SmolVLM2-500M ...")
policy_500m = build_policy("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", num_vlm_layers=16)

print("\n[INFO] Loading SmolVLM2-2.2B ...")
policy_2b   = build_policy("HuggingFaceTB/SmolVLM2-2.2B-Instruct",      num_vlm_layers=24)

# (models are already on CPU if _force_offload — build_policy skipped .to(device))

# Load the direct VLM extractor (optional — only when --vlm_direct is passed)
vlm_direct_extractor = None
if args.vlm_direct:
    vlm_direct_extractor = SmolVLMDirectExtractor(
        args.vlm_direct_model, device=device, load_on_cpu=_force_offload)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Load dataset
# ──────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Loading dataset:", args.repo_id)
dataset = LeRobotDataset(
    args.repo_id,
    root=None,
    batch_encoding_size=1,
    video_backend=args.video_backend,
)
dataset_fps = dataset.fps
print(f"[INFO] Dataset FPS = {dataset_fps}  →  1 frame per second")

total_episodes = len(dataset.meta.episodes)
if not (0 <= args.episode < total_episodes):
    raise ValueError(f"Episode {args.episode} out of range [0, {total_episodes})")

ep_meta   = dataset.meta.episodes[args.episode]
ep_length = ep_meta["length"]
start_idx = sum(dataset.meta.episodes[i]["length"] for i in range(args.episode))
end_idx   = start_idx + ep_length
print(f"[INFO] Episode {args.episode}  |  frames {start_idx}–{end_idx}  |  ~{ep_length // dataset_fps} seconds\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Helper functions (shared)
# ──────────────────────────────────────────────────────────────────────────────

def compute_attention(policy, batch, dev):
    """Prefix-only forward pass; captures self-attention weights."""
    policy.eval()
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(dev)

    task = batch.get("task", "")
    if isinstance(task, str) and not task.endswith("\n"):
        batch["task"] = task + "\n"

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tok = tokenizer(
        batch["task"],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
    )
    batch[OBS_LANGUAGE_TOKENS]         = tok["input_ids"].to(dev)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].to(dev).bool()

    images, img_masks = policy.prepare_images(batch)

    with torch.no_grad():
        num_img_tokens = int(policy.model.vlm_with_expert.embed_image(images[0]).shape[1])
    policy.model.vlm_with_expert._debug_num_img_tokens = num_img_tokens
    policy.model.vlm_with_expert._debug_num_images     = len(images)

    if OBS_STATE in batch:
        state = policy.prepare_state(batch)
    else:
        bsize = batch[OBS_LANGUAGE_TOKENS].shape[0]
        state = torch.zeros((bsize, policy.config.max_state_dim), device=dev, dtype=torch.float32)

    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        state=state,
    )

    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    att_2d  = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    with torch.no_grad():
        policy.model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )


def extract_lang_to_img_heatmaps(policy, batch):
    """Extract per-camera language→image attention heatmaps.
    Returns {camera_name: heat_1d_tensor} for each camera in ACTIVE_CAMERAS."""
    attn = getattr(policy.model.vlm_with_expert, "last_attn_weights", None)
    if attn is None:
        raise RuntimeError("No attention captured.")

    attn_matrix      = attn[0].mean(0)                                          # [Q, K]
    num_img_tokens   = policy.model.vlm_with_expert._debug_num_img_tokens
    num_lang_tokens  = batch[OBS_LANGUAGE_TOKENS].shape[1]
    num_images       = policy.model.vlm_with_expert._debug_num_images
    img_start        = 0
    img_end          = num_img_tokens * num_images
    lang_start       = img_end
    lang_end         = lang_start + num_lang_tokens

    lang_query = attn_matrix[lang_start:lang_end].mean(0)                       # [K]

    # Slice each camera's token block in the order they appear in ACTIVE_CAMERAS
    return {
        cam: lang_query[img_start + i * num_img_tokens :
                        img_start + (i + 1) * num_img_tokens]
        for i, cam in enumerate(ACTIVE_CAMERAS)
    }


def process_heatmap(heat_1d, original_image_size=(480, 640), model_input_size=(512, 512)):
    grid_size  = int(math.sqrt(heat_1d.numel()))
    heat_1d    = heat_1d[: grid_size * grid_size]
    heat_2d    = heat_1d.reshape(grid_size, grid_size)
    heat_t     = torch.tensor(heat_2d).unsqueeze(0).unsqueeze(0).float()
    heat_512   = F.interpolate(heat_t, size=model_input_size, mode="bilinear", align_corners=False)

    orig_h, orig_w = original_image_size
    tgt_h,  tgt_w  = model_input_size
    ratio      = max(orig_w / tgt_w, orig_h / tgt_h)
    resized_h  = int(orig_h / ratio)
    resized_w  = int(orig_w / ratio)
    pad_w      = max(0, tgt_w - resized_w)
    pad_h      = max(0, tgt_h - resized_h)

    heat_valid = heat_512[0, 0, pad_h : pad_h + resized_h, pad_w : pad_w + resized_w]
    heat_final = F.interpolate(heat_valid.unsqueeze(0).unsqueeze(0), size=original_image_size, mode="bilinear")
    return heat_final[0, 0].numpy()


def overlay(rgb_img, heat_2d):
    v_min, v_max  = np.percentile(heat_2d, [0, 98])
    heat_clipped  = np.clip(heat_2d, v_min, v_max)
    heat_norm     = (heat_clipped - v_min) / (v_max - v_min + 1e-8)
    heat_color    = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_JET)
    rgb_bgr       = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(rgb_bgr, 0.6, heat_color, 0.4, 0)


def label_panel(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 4)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,   2)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 6. Output folder
# ──────────────────────────────────────────────────────────────────────────────
_tag = "vlm_vs_vla" if args.vlm_direct else "500m_vs_2b"
out_dir = f"attn_compare_{_tag}_{args.camera}"
os.makedirs(out_dir, exist_ok=True)
print(f"[INFO] Saving comparison PNGs to: {os.path.abspath(out_dir)}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Main Loop
# ──────────────────────────────────────────────────────────────────────────────
current_frame = 0
saved_count   = 0

for global_idx in range(start_idx, end_idx):

    # ── 1 Hz sampling ────────────────────────────────────────────────────────
    if current_frame % dataset_fps != 0:
        current_frame += 1
        continue

    second_idx = current_frame // dataset_fps
    item = dataset[global_idx]

    # ── Build shared batch — only the selected camera(s) ─────────────────────
    active_keys = {CAMERA_KEYS[c] for c in ACTIVE_CAMERAS}
    def make_batch():
        b = {}
        for k, v in item.items():
            if k in active_keys:
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                b[k] = v.unsqueeze(0)
            elif k == "observation.state" and args.use_state:
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                b[OBS_STATE] = v.unsqueeze(0)
        b["task"] = args.prompt
        return b

    # ── RGB images (only active cameras) ────────────────────────────────────
    rgb_imgs = {}
    for cam in ACTIVE_CAMERAS:
        t = item[CAMERA_KEYS[cam]].permute(1, 2, 0).cpu().numpy()
        rgb_imgs[cam] = (t * 255).astype(np.uint8) if t.max() <= 1.5 else t.astype(np.uint8)

    # ── Run 500M ────────────────────────────────────────────────────────────
    if _force_offload:
        policy_500m.to(device)

    batch_500m = make_batch()
    compute_attention(policy_500m, batch_500m, device)
    heats_500m = extract_lang_to_img_heatmaps(policy_500m, batch_500m)

    if _force_offload:
        policy_500m.cpu()
        torch.cuda.empty_cache()

    # ── Run 2.2B ────────────────────────────────────────────────────────────
    if _force_offload:
        policy_2b.to(device)

    batch_2b = make_batch()
    compute_attention(policy_2b, batch_2b, device)
    heats_2b = extract_lang_to_img_heatmaps(policy_2b, batch_2b)

    if _force_offload:
        policy_2b.cpu()
        torch.cuda.empty_cache()

    # ── Run VLM-direct (base SmolVLM2, middle layers, no SmolVLAPolicy) ─────
    heats_vlm_direct = {}
    if vlm_direct_extractor is not None:
        if _force_offload:
            vlm_direct_extractor.model.to(device)
        for cam in ACTIVE_CAMERAS:
            heat_1d, grid_hw = vlm_direct_extractor.get_heatmap(rgb_imgs[cam], args.prompt)
            orig_h, orig_w = rgb_imgs[cam].shape[:2]
            # SmolVLM2 grid is square, so process_heatmap handles it correctly
            heats_vlm_direct[cam] = process_heatmap(
                heat_1d[:grid_hw[0] * grid_hw[1]].float())
        if _force_offload:
            vlm_direct_extractor.model.cpu()
            torch.cuda.empty_cache()

    # ── Build panels: [ 500M cam(s)... | 2.2B cam(s)... | VLM-direct cam(s)... ] ──
    label  = args.token or "all tokens"
    panels = []
    for cam in ACTIVE_CAMERAS:
        vis = overlay(rgb_imgs[cam], process_heatmap(heats_500m[cam]))
        panels.append(label_panel(vis, f"500M-VLA {cam} ({label})", color=(200, 255, 200)))
    for cam in ACTIVE_CAMERAS:
        vis = overlay(rgb_imgs[cam], process_heatmap(heats_2b[cam]))
        panels.append(label_panel(vis, f"2.2B-VLA {cam} ({label})", color=(255, 200, 200)))
    if heats_vlm_direct:
        for cam in ACTIVE_CAMERAS:
            vis = overlay(rgb_imgs[cam], heats_vlm_direct[cam])
            panels.append(label_panel(vis, f"2.2B-VLM-only {cam} (mid-layers)",
                                      color=(200, 200, 255)))

    top_row = np.hstack(panels)
    h, w, _ = panels[0].shape
    for i in range(1, len(panels)):
        cv2.line(top_row, (w * i, 0), (w * i, h), (255, 255, 255), 2)

    bar  = np.zeros((40, top_row.shape[1], 3), dtype=np.uint8)
    info = f'sec {second_idx:04d} | frame {current_frame:05d} | camera={args.camera} | "{args.prompt[:70]}"'
    cv2.putText(bar, info, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    final = np.vstack([top_row, bar])

    # ── Save ─────────────────────────────────────────────────────────────────
    png_path = os.path.join(out_dir, f"compare_sec_{second_idx:04d}.png")
    cv2.imwrite(png_path, final)
    saved_count += 1
    print(f"[INFO] sec {second_idx:04d} (frame {current_frame:05d}) → {png_path}")

    current_frame += 1

print(f"\n[DONE] Saved {saved_count} comparison PNGs → {os.path.abspath(out_dir)}")
print(
    "\nHow to interpret:\n"
    "  - If 500M and 2.2B look identical for very different prompts → 500M is NOT prompt-sensitive.\n"
    "  - If 2.2B attention clearly shifts with different prompts    → 2.2B IS better for this task.\n"
    "  - Look for: hotspots moving between objects when you change which object is mentioned.\n"
)
