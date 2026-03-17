'''
Task-Switch Analysis: WHY Can SmolVLA Handle Multi-Task Switching?
===================================================================
Model: ethanCSL/svla_koch_sorting_n_stacking  (fine-tuned SmolVLA)
Dataset:  ethanCSL/svla_koch_sorting_n_stacking
  ep   0 –  99 : SORTING  → "put the red cube in the right box, and green cube in the left box."
  ep 100 – 199 : STACKING → "put the green cube on top of red cube."

Five analyses, each saved as one figure in  task_switch_analysis/ :

  A ─ Per-Token Attention Heatmaps   (Don't Blind Your VLA style)
      VLM self-attention (layers 0, 7, 15), head-averaged:
        each language probe-token  →  image patches  →  per-camera overlay.
      Shows whether spatial / object grounding survives fine-tuning.
      • "red"/"green" should ground to the correct coloured cube.
      • "right"/"left"/"on" should ground to the correct spatial region.

  B ─ Prompt Sensitivity / Action Divergence
      Same 5 frames × 4 prompts: correct | wrong-task | opposite-color | empty.
      Predicted actions use a fixed noise seed for reproducibility.
      Bar chart: angular divergence from correct-prompt prediction, per task.

  C ─ Cross-Task Language-Conditioning Strength
      Correct vs. wrong-task prompt on sampled frames from each task.
      Violin plot + per-chunk-step divergence curves.
      Answer: "does swapping the task prompt actually change the trajectory?"

  D ─ Task Representation PCA
      Last-layer VLM hidden state at the STATE-token position, collected from
      both tasks under both prompts.
      PCA scatter coloured by (task, prompt) → 4 groups.
      Tests whether the model has distinct internal "task modes".

  E ─ Same-Frame Attention Shift: Sort vs. Stack Prompt
      ONE frame from each task, overlaid with attention maps under BOTH prompts
      for the same probe token (e.g. "red").
      Directly shows how language conditioning re-routes visual attention.

Usage (recommended):
 python src/lerobot/scripts/record_task_switch_why.py   
    --repo_id  "ethanCSL/svla_koch_sorting_n_stacking"   
    --ckpt     "ethanCSL/svla_koch_sorting_n_stacking"   
    --sort_episodes  "0,5,10,15,20"   
    --stack_episodes "100,105,110,115,120"   
    --sort_prompt  "put the red cube in the right box, and green cube in the left box."   
    --stack_prompt "put the green cube on top of red cube."   
    --probe_tokens "red,green,right,left,on,cube"   
    --frame_stride 5   
    --rename_map '{"observation.images.front":"observation.images.camera1","observation.images.top":"observation.images.camera2"}'
'''

import os
import math
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Arguments
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Task-Switch Analysis for SmolVLA")
parser.add_argument("--repo_id",        type=str,
                    default="ethanCSL/svla_koch_sorting_n_stacking")
parser.add_argument("--ckpt",           type=str,
                    default="ethanCSL/svla_koch_sorting_n_stacking")
parser.add_argument("--sort_episodes",  type=str, default="0,5,10,15,20",
                    help="Comma-sep episode indices for the SORTING task (ep 0–99).")
parser.add_argument("--stack_episodes", type=str, default="100,105,110,115,120",
                    help="Comma-sep episode indices for the STACKING task (ep 100–199).")
parser.add_argument("--sort_prompt",    type=str,
                    default="put the red cube in the right box, and green cube in the left box.")
parser.add_argument("--stack_prompt",   type=str,
                    default="put the green cube on top of red cube.")
parser.add_argument("--probe_tokens",   type=str, default="red,green,right,left,on,cube",
                    help="Comma-sep words to probe in the per-token heatmaps (Analysis A & E).")
parser.add_argument("--frame_stride",   type=int, default=5,
                    help="Sample every Nth frame for analyses C & D.")
parser.add_argument("--rename_map",     type=str,
                    default='{"observation.images.front":"observation.images.camera1",'
                            '"observation.images.top":"observation.images.camera2"}')
parser.add_argument("--video_backend",  type=str, default="pyav")
parser.add_argument("--output_dir",     type=str, default=None,
                    help="Directory to save figures. Default: scripts/task_switch_analysis/")
# Analysis A layers to average
parser.add_argument("--attn_layers",    type=str, default="0,7,15",
                    help="Comma-sep VLM layer indices to average for attention maps.")
args = parser.parse_args()

# ─── derived constants ────────────────────────────────────────────────────────
SORT_EPISODES  = [int(e.strip()) for e in args.sort_episodes.split(",")  if e.strip()]
STACK_EPISODES = [int(e.strip()) for e in args.stack_episodes.split(",") if e.strip()]
SORT_PROMPT    = args.sort_prompt
STACK_PROMPT   = args.stack_prompt
PROBE_TOKENS   = [t.strip() for t in args.probe_tokens.split(",") if t.strip()]
RENAME_MAP     = json.loads(args.rename_map)
ATTN_LAYERS_REQ = [int(l.strip()) for l in args.attn_layers.split(",") if l.strip()]

OUT_DIR = args.output_dir or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "task_switch_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load Dataset and Policy
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[LOAD] Dataset: {args.repo_id}")
dataset = LeRobotDataset(args.repo_id, batch_encoding_size=1,
                         video_backend=args.video_backend)
total_episodes = len(dataset.meta.episodes)
print(f"[LOAD] Total episodes: {total_episodes}")

print(f"[LOAD] Checkpoint: {args.ckpt}")
policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=RENAME_MAP)
policy.to(device)
policy.eval()

preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset.meta.stats)
vlm = policy.model.vlm_with_expert
vlm.attention_mode = "self_attn"   # prefix-only pass always uses self-attn

NUM_VLM_LAYERS = vlm.num_vlm_layers
ATTN_LAYERS = [l for l in ATTN_LAYERS_REQ if l < NUM_VLM_LAYERS]
if not ATTN_LAYERS:
    ATTN_LAYERS = [0, NUM_VLM_LAYERS // 2, NUM_VLM_LAYERS - 1]
print(f"[INFO] VLM layers: {NUM_VLM_LAYERS}  |  Averaging layers: {ATTN_LAYERS}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Episode / Frame helpers
# ─────────────────────────────────────────────────────────────────────────────
_cum = 0
_ep_starts: list[int] = []
for ep in dataset.meta.episodes:
    _ep_starts.append(_cum)
    _cum += ep["length"]

def get_frame_range(ep_idx: int) -> tuple[int, int]:
    s = _ep_starts[ep_idx]
    return s, s + dataset.meta.episodes[ep_idx]["length"]

def get_item(global_idx: int):
    return dataset[global_idx]

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Build preprocessed observation batch
# ─────────────────────────────────────────────────────────────────────────────
def build_batch(item, prompt: str) -> dict:
    """Build a preprocessed batch from a dataset item and task prompt."""
    obs: dict = {}
    for k, v in item.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if k in RENAME_MAP:
            obs[RENAME_MAP[k]] = v.unsqueeze(0).to(device)
        elif k.startswith("observation."):
            obs[k] = v.unsqueeze(0).to(device)

    # Pad any missing cameras that the policy expects with zeros
    real_cam_keys = [k for k in obs if "images" in k and "empty" not in k]
    if real_cam_keys:
        ref = obs[real_cam_keys[0]]
        for pad_key in ["observation.images.camera3",
                        "observation.images.empty_camera_0"]:
            if pad_key not in obs:
                obs[pad_key] = torch.zeros_like(ref)

    obs["task"] = prompt
    return preprocessor(obs)

def get_image_numpy(item, cam_src_key: str) -> np.ndarray | None:
    """Return (H, W, 3) uint8 numpy image from dataset item."""
    v = item.get(cam_src_key)
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        img = v.permute(1, 2, 0).float().numpy()
    else:
        img = np.array(v, dtype=np.float32)
    if img.max() <= 1.0:
        img = (img * 255)
    return img.astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Deterministic action prediction (fixed noise seed)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_action_fixed(batch_pp: dict, seed: int = 42) -> torch.Tensor:
    """Return predicted action chunk [chunk_size, real_dim] with fixed noise."""
    images, img_masks = policy.prepare_images(batch_pp)
    state             = policy.prepare_state(batch_pp)
    lang_tokens       = batch_pp[OBS_LANGUAGE_TOKENS]
    lang_masks        = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]
    bsize      = state.shape[0]
    chunk_size = policy.model.config.chunk_size
    max_dim    = policy.model.config.max_action_dim
    gen   = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((bsize, chunk_size, max_dim),
                        device=device, generator=gen, dtype=state.dtype)
    predicted = policy.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state, noise=noise
    )
    real_dim = policy.config.action_feature.shape[0]
    return predicted[0, :, :real_dim].float().cpu()

def angular_divergence_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    """Angular divergence (degrees) between two flattened action chunks."""
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    cos = F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).clamp(-1.0, 1.0)
    return float(np.degrees(np.arccos(cos.item())))

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Per-layer attention capture  (monkeypatch eager_attention_forward)
# ─────────────────────────────────────────────────────────────────────────────
# eager_attention_forward overwrites self.last_attn_weights at each layer.
# We replace the instance method with a wrapper that also appends to a list.
_orig_eager = vlm.eager_attention_forward   # bound method (self already bound)

def _capturing_eager(attention_mask, batch_size, head_dim, q, k, v):
    result = _orig_eager(attention_mask, batch_size, head_dim, q, k, v)
    if getattr(vlm, "_do_capture_attn", False):
        vlm._attn_layer_cache.append(vlm.last_attn_weights.clone())   # [B, H, L, L]
    return result

vlm.eager_attention_forward = _capturing_eager
vlm._do_capture_attn = False
vlm._attn_layer_cache: list = []

def run_prefix_and_capture(batch_pp: dict):
    """
    Embed the prefix and run the VLM self-attention forward pass.
    Returns:
        hidden_per_layer  – list of 16 × [1, L, D]  (after each VLM layer)
        attn_per_layer    – list of 16 × [1, H, L, L]
        num_img_tokens    – image tokens per camera
        num_images        – number of camera images fed to the model
    """
    for k, v in list(batch_pp.items()):
        if isinstance(v, torch.Tensor):
            batch_pp[k] = v.to(device)

    images, img_masks = policy.prepare_images(batch_pp)
    state             = policy.prepare_state(batch_pp)

    with torch.no_grad():
        num_img_tokens = int(vlm.embed_image(images[0]).shape[1])
    vlm._debug_num_img_tokens = num_img_tokens
    vlm._debug_num_images     = len(images)

    lang_tokens = batch_pp[OBS_LANGUAGE_TOKENS]
    lang_masks  = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad, prefix_att = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d  = make_att_2d_masks(prefix_pad, prefix_att)
    prefix_pos_ids = torch.cumsum(prefix_pad, dim=1) - 1

    # Reset per-layer cache then run
    vlm._attn_layer_cache = []
    vlm._do_capture_attn  = True
    with torch.no_grad():
        vlm.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
    vlm._do_capture_attn = False

    return (
        list(vlm.hidden_per_layer),   # 16 × [1, L, D]
        list(vlm._attn_layer_cache),  # 16 × [1, H, L, L]
        num_img_tokens,
        len(images),
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Token position finder
# ─────────────────────────────────────────────────────────────────────────────
def find_token_pos(prompt: str, word: str) -> int | None:
    """Return first subword-token index of `word` in tokenised `prompt`."""
    processor = vlm.processor
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"
    ids   = processor.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    words = processor.tokenizer.convert_ids_to_tokens(ids)
    for i, w in enumerate(words):
        clean = w.replace("Ġ", "").replace("▁", "").lower()
        if word.lower() in clean:
            return i
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Heatmap rendering
# ─────────────────────────────────────────────────────────────────────────────
def attn_to_heatmap_overlay(attn_1d: torch.Tensor,
                             orig_img: np.ndarray,
                             model_input_size: tuple[int, int] = (512, 512),
                             alpha: float = 0.55,
                             colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    Convert 1-D per-patch attention → coloured overlay on the original image.

    The model pads images with a black border (top + left) to reach
    `model_input_size` before passing to SigLIP.  We reverse that crop so the
    heatmap aligns with the unpadded content.
    """
    n = attn_1d.numel()
    grid = int(math.sqrt(n))
    if grid * grid < n:
        attn_1d = attn_1d[:grid * grid]
    heat2d = attn_1d.reshape(grid, grid).float().numpy()

    orig_h, orig_w = orig_img.shape[:2]
    tgt_h, tgt_w   = model_input_size

    # Reverse SmolVLA's resize-with-padding (pad_width left, pad_height top)
    ratio   = max(orig_w / tgt_w, orig_h / tgt_h)
    res_h   = int(orig_h / ratio)
    res_w   = int(orig_w / ratio)
    pad_h   = max(0, tgt_h - res_h)
    pad_w   = max(0, tgt_w - res_w)

    heat_t   = torch.tensor(heat2d).unsqueeze(0).unsqueeze(0)
    heat_512 = F.interpolate(heat_t, size=model_input_size,
                             mode="bilinear", align_corners=False)[0, 0]
    heat_valid = heat_512[pad_h: pad_h + res_h,
                          pad_w: pad_w + res_w].numpy()
    heat_orig  = cv2.resize(heat_valid, (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR)

    # Percentile-clip to suppress noise
    lo, hi = np.percentile(heat_orig, [1, 99])
    heat_n = np.clip((heat_orig - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    hmap_bgr = cv2.applyColorMap((heat_n * 255).astype(np.uint8), colormap)
    hmap_rgb = cv2.cvtColor(hmap_bgr, cv2.COLOR_BGR2RGB)
    overlay  = (alpha * hmap_rgb.astype(np.float32)
                + (1 - alpha) * orig_img.astype(np.float32)).astype(np.uint8)
    return overlay

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS A  –  Per-Token Attention Heatmaps
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis_A():
    """
    One frame per task × probe tokens.
    Layer-averaged (ATTN_LAYERS) head-averaged self-attention:
      language token → image patches.
    Inspired by the token-grounding maps in "Don't Blind Your VLA" (Fig. 3).
    """
    print("\n" + "=" * 60)
    print("Analysis A: Per-Token Attention Heatmaps")
    print("=" * 60)

    # Map dataset camera keys to display labels
    DISPLAY_CAMS = [
        ("observation.images.front", "front"),
        ("observation.images.top",   "top"),
    ]

    task_setups = [
        {"name": "Sorting",  "ep_idx": SORT_EPISODES[0],  "frame_off": 0, "prompt": SORT_PROMPT},
        {"name": "Stacking", "ep_idx": STACK_EPISODES[0], "frame_off": 0, "prompt": STACK_PROMPT},
    ]

    # Which probe tokens appear in each task's prompt?
    task_probe_maps: list[dict[str, int]] = []
    for setup in task_setups:
        pmap: dict[str, int] = {}
        for word in PROBE_TOKENS:
            pos = find_token_pos(setup["prompt"], word)
            if pos is not None:
                pmap[word] = pos
            else:
                print(f"  [A] '{word}' not found in {setup['name']} prompt → skip")
        task_probe_maps.append(pmap)

    all_words = sorted({w for pm in task_probe_maps for w in pm})
    if not all_words:
        print("[A] No probe tokens found — skipping Analysis A.")
        return

    n_tasks  = len(task_setups)
    n_tokens = len(all_words)
    n_cams   = len(DISPLAY_CAMS)

    # Rows: tasks × cameras   Cols: probe tokens
    fig, axes = plt.subplots(
        n_tasks * n_cams, n_tokens,
        figsize=(3.2 * n_tokens, 3.4 * n_tasks * n_cams),
        squeeze=False,
    )

    for t_i, (setup, pmap) in enumerate(zip(task_setups, task_probe_maps)):
        ep_start, ep_end = get_frame_range(setup["ep_idx"])
        f_idx = min(ep_start + setup["frame_off"], ep_end - 1)
        item  = get_item(f_idx)

        batch_pp = build_batch(item, setup["prompt"])
        _, attn_layers, num_img_tokens, num_images = run_prefix_and_capture(batch_pp)

        # Average attention over requested layers and all heads
        valid = [attn_layers[l] for l in ATTN_LAYERS if l < len(attn_layers)]
        if not valid:
            print(f"  [A] No valid attn layers for {setup['name']}, skipping.")
            continue
        # [H, L, L] → [L, L]
        attn_avg = torch.stack([a[0].float() for a in valid]).mean(0).mean(0)
        lang_offset = num_images * num_img_tokens   # image tokens come first

        for w_i, word in enumerate(all_words):
            tok_local_pos = pmap.get(word)   # position inside the tokenized prompt

            for c_i, (cam_src, cam_label) in enumerate(DISPLAY_CAMS):
                row = t_i * n_cams + c_i
                ax  = axes[row, w_i]

                orig_img = get_image_numpy(item, cam_src)

                if tok_local_pos is None or orig_img is None:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="grey")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue

                # Absolute position in the prefix sequence
                seq_pos   = lang_offset + tok_local_pos

                # Image tokens for this camera (cam_i-th block)
                cam_start = c_i * num_img_tokens
                cam_end   = cam_start + num_img_tokens

                if seq_pos >= attn_avg.shape[0]:
                    ax.text(0.5, 0.5, "OOB", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="orange")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue

                heat_1d  = attn_avg[seq_pos, cam_start:cam_end].cpu()
                overlay  = attn_to_heatmap_overlay(heat_1d, orig_img)

                ax.imshow(overlay)
                ax.set_xticks([]); ax.set_yticks([])
                if t_i == 0 and c_i == 0:
                    ax.set_title(f'"{word}"', fontsize=11, fontweight="bold")
                if w_i == 0:
                    ax.set_ylabel(f"{setup['name']}\n{cam_label}", fontsize=8)

    fig.suptitle(
        f"Analysis A ─ Per-Token Attention Heatmaps\n"
        f"VLM self-attn (layers {ATTN_LAYERS}, head-avg): "
        f"language token → image patches\n"
        f"Well-grounded = token highlights the correct object/region",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "A_per_token_heatmaps.png")
    plt.savefig(out, bbox_inches="tight", dpi=130)
    print(f"[A] Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS B  –  Prompt Sensitivity (Action Divergence)
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis_B():
    """
    Same 5 frames per task × 4 prompts.
    Measures how much the predicted action chunk changes when the prompt
    deviates from the correct task instruction.
    Large bars → model is strongly conditioned on language.
    """
    print("\n" + "=" * 60)
    print("Analysis B: Prompt Sensitivity (Action Divergence)")
    print("=" * 60)

    OPPOSITE_PROMPT = "put the green cube in the right box, and red cube in the left box."
    EMPTY_PROMPT    = "."

    task_configs = [
        {"name": "Sorting",  "episodes": SORT_EPISODES[:3],
         "correct": SORT_PROMPT,  "wrong": STACK_PROMPT},
        {"name": "Stacking", "episodes": STACK_EPISODES[:3],
         "correct": STACK_PROMPT, "wrong": SORT_PROMPT},
    ]
    prompt_labels  = ["correct", "wrong-task", "opposite-color", "empty"]
    prompt_colors  = {"correct": "#27ae60", "wrong-task": "#e74c3c",
                      "opposite-color": "#e67e22", "empty": "#95a7b5"}
    N_FRAMES_PER_EP = 5   # first 5 frames per episode

    results: dict[str, dict[str, list[float]]] = {
        t["name"]: {k: [] for k in prompt_labels}
        for t in task_configs
    }

    for task in task_configs:
        probes = {
            "correct":        task["correct"],
            "wrong-task":     task["wrong"],
            "opposite-color": OPPOSITE_PROMPT,
            "empty":          EMPTY_PROMPT,
        }
        for ep_idx in task["episodes"]:
            ep_start, ep_end = get_frame_range(ep_idx)
            frames = list(range(ep_start, min(ep_start + N_FRAMES_PER_EP, ep_end)))
            for f_idx in frames:
                item       = get_item(f_idx)
                a_correct  = predict_action_fixed(build_batch(item, task["correct"]))
                for p_name, p_text in probes.items():
                    if p_name == "correct":
                        results[task["name"]][p_name].append(0.0)
                        continue
                    a_p = predict_action_fixed(build_batch(item, p_text))
                    results[task["name"]][p_name].append(
                        angular_divergence_vec(a_correct, a_p))

        for k, v in results[task["name"]].items():
            if v:
                print(f"  {task['name']:10s}  {k:15s}  "
                      f"mean={np.mean(v):.2f}°  median={np.median(v):.2f}°  n={len(v)}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x     = np.arange(len(prompt_labels))
    width = 0.6

    for ax_i, (task_name, task_data) in enumerate(results.items()):
        vals   = [np.mean(task_data[k])  if task_data[k] else 0.0 for k in prompt_labels]
        errors = [np.std(task_data[k])   if task_data[k] else 0.0 for k in prompt_labels]
        colors = [prompt_colors[k] for k in prompt_labels]

        bars = axes[ax_i].bar(x, vals, width, yerr=errors, capsize=5,
                              color=colors, edgecolor="white", linewidth=0.6)
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(
            [p.replace("-", "\n") for p in prompt_labels], fontsize=9)
        axes[ax_i].set_title(f"{task_name} Task", fontsize=12)
        axes[ax_i].set_ylabel("Angular divergence (°) from correct prompt", fontsize=9)
        axes[ax_i].set_ylim(0, None)
        axes[ax_i].axhline(5, color="grey", linestyle="--", linewidth=0.8,
                           label="5° threshold")
        axes[ax_i].legend(fontsize=8)

        for bar, val in zip(bars, vals):
            if val > 0.3:
                axes[ax_i].text(bar.get_x() + bar.get_width() / 2,
                                val + 0.5, f"{val:.1f}°",
                                ha="center", va="bottom", fontsize=8)

    legend_handles = [
        mpatches.Patch(color=prompt_colors[k], label=k) for k in prompt_labels
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(prompt_labels), fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Analysis B ─ Prompt Sensitivity: Action Divergence from Correct Prompt\n"
        "Large bars → model uses language to condition motor trajectories",
        fontsize=11,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "B_prompt_sensitivity.png")
    plt.savefig(out, bbox_inches="tight", dpi=130)
    print(f"[B] Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS C  –  Cross-Task Language-Conditioning Strength
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis_C():
    """
    Correct prompt vs. wrong-task prompt across many frames per task.
    Left: violin distribution of angular divergence.
    Right: mean ± std per-chunk-step divergence profile.
    """
    print("\n" + "=" * 60)
    print("Analysis C: Cross-Task Language-Conditioning Strength")
    print("=" * 60)

    task_configs = [
        {"name": "Sorting",  "episodes": SORT_EPISODES,
         "correct": SORT_PROMPT,  "wrong": STACK_PROMPT, "color": "#3498db"},
        {"name": "Stacking", "episodes": STACK_EPISODES,
         "correct": STACK_PROMPT, "wrong": SORT_PROMPT,  "color": "#e74c3c"},
    ]

    all_divs: dict[str, np.ndarray] = {}
    chunk_profiles: dict[str, np.ndarray] = {}
    chunk_size = policy.config.chunk_size

    for task in task_configs:
        divs, profiles = [], []
        for ep_idx in task["episodes"]:
            ep_start, ep_end = get_frame_range(ep_idx)
            sampled = list(range(ep_start, ep_end, args.frame_stride))
            for f_idx in sampled:
                item = get_item(f_idx)
                a_c  = predict_action_fixed(build_batch(item, task["correct"]))
                a_w  = predict_action_fixed(build_batch(item, task["wrong"]))
                # Whole-chunk angular divergence
                divs.append(angular_divergence_vec(a_c, a_w))
                # Per-step profile
                step_divs = []
                for s in range(chunk_size):
                    cos = F.cosine_similarity(
                        a_c[s].float().unsqueeze(0),
                        a_w[s].float().unsqueeze(0),
                    ).clamp(-1.0, 1.0)
                    step_divs.append(np.degrees(np.arccos(cos.item())))
                profiles.append(step_divs)

        all_divs[task["name"]]      = np.array(divs)
        chunk_profiles[task["name"]] = np.array(profiles)   # [N, chunk_size]
        print(f"  {task['name']:10s}  n={len(divs)}  "
              f"mean={np.mean(divs):.2f}°  median={np.median(divs):.2f}°  "
              f"p95={np.percentile(divs, 95):.2f}°")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    task_names = list(all_divs.keys())
    colors     = [t["color"] for t in task_configs]

    # ── Left: violin + jitter ─────────────────────────────────────────────────
    ax = axes[0]
    parts = ax.violinplot([all_divs[t] for t in task_names],
                          positions=[1, 2], showmedians=True, showextrema=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.65)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2.5)
    rng = np.random.default_rng(0)
    for i, t in enumerate(task_names):
        jitter = rng.uniform(-0.12, 0.12, len(all_divs[t]))
        ax.scatter(np.full(len(all_divs[t]), i + 1) + jitter,
                   all_divs[t], s=9, alpha=0.35, color=colors[i], zorder=3)
    ax.set_xticks([1, 2]); ax.set_xticklabels(task_names, fontsize=11)
    ax.set_ylabel("Angular divergence (°)  correct vs. wrong-task prompt", fontsize=9)
    ax.set_title("Action Divergence Distribution\n"
                 "(whole action chunk, correct vs. wrong-task prompt)", fontsize=10)

    # ── Right: per-step divergence profile ────────────────────────────────────
    ax2 = axes[1]
    x_steps = np.arange(chunk_size)
    for t_i, (t_name, color) in enumerate(zip(task_names, colors)):
        p     = chunk_profiles[t_name]
        mean  = p.mean(0)
        std   = p.std(0)
        ax2.plot(x_steps, mean, label=t_name, color=color, linewidth=2.2)
        ax2.fill_between(x_steps, mean - std, mean + std, alpha=0.18, color=color)
    ax2.set_xlabel("Action chunk step", fontsize=10)
    ax2.set_ylabel("Angular divergence (°)", fontsize=10)
    ax2.set_title("Per-Step Divergence across the Action Chunk\n"
                  "(mean ± std  over sampled frames)", fontsize=10)
    ax2.legend(fontsize=10)

    fig.suptitle(
        "Analysis C ─ Language-Conditioning Strength\n"
        "High divergence → the task prompt strongly steers the motor trajectory",
        fontsize=11,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "C_cross_task_divergence.png")
    plt.savefig(out, bbox_inches="tight", dpi=130)
    print(f"[C] Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS D  –  Task Representation PCA
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis_D():
    """
    Last VLM layer hidden state at the STATE-token position collected from
    both tasks under both prompts (4 groups total).
    PCA scatter shows whether the model has distinct internal task modes.
    """
    print("\n" + "=" * 60)
    print("Analysis D: Task Representation PCA")
    print("=" * 60)

    group_configs = [
        {"name": "Sorting  (correct)",  "eps": SORT_EPISODES,  "prompt": SORT_PROMPT,
         "color": "#2980b9", "marker": "o"},
        {"name": "Stacking (correct)",  "eps": STACK_EPISODES, "prompt": STACK_PROMPT,
         "color": "#c0392b", "marker": "s"},
        {"name": "Sorting  (wrong↑)",   "eps": SORT_EPISODES,  "prompt": STACK_PROMPT,
         "color": "#85c1e9", "marker": "o"},
        {"name": "Stacking (wrong↓)",   "eps": STACK_EPISODES, "prompt": SORT_PROMPT,
         "color": "#f1948a", "marker": "s"},
    ]

    feats_per_group: list[list[torch.Tensor]] = []
    for gcfg in group_configs:
        feats: list[torch.Tensor] = []
        for ep_idx in gcfg["eps"]:
            ep_start, ep_end = get_frame_range(ep_idx)
            for f_idx in range(ep_start, ep_end, args.frame_stride):
                item     = get_item(f_idx)
                batch_pp = build_batch(item, gcfg["prompt"])
                hidden, _, _, _ = run_prefix_and_capture(batch_pp)
                if not hidden:
                    continue
                # Last layer, state-token (last position in prefix)
                state_repr = hidden[-1][0, -1, :].detach().float().cpu()
                feats.append(state_repr)
        feats_per_group.append(feats)
        print(f"  {gcfg['name']:30s}: {len(feats)} points")

    all_feats = [f for grp in feats_per_group for f in grp]
    if len(all_feats) < 10:
        print("[D] Not enough points for PCA — skipping.")
        return

    X = torch.stack(all_feats).numpy()
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X)
    exp_var = pca.explained_variance_ratio_[:2].sum() * 100

    # ── Figure: two panels ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_i, ax in enumerate(axes):
        offset = 0
        for g_i, (gcfg, feats) in enumerate(zip(group_configs, feats_per_group)):
            n   = len(feats)
            pts = X2[offset: offset + n]
            # Right panel: only correct-prompt groups
            if ax_i == 1 and g_i >= 2:
                offset += n; continue
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=gcfg["color"], marker=gcfg["marker"],
                       s=35, alpha=0.75, label=gcfg["name"], linewidths=0)
            offset += n
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
        ax.legend(fontsize=8, loc="best")

    axes[0].set_title("All 4 groups (correct + wrong prompts)", fontsize=10)
    axes[1].set_title("Correct prompts only", fontsize=10)

    fig.suptitle(
        f"Analysis D ─ Task Representation PCA\n"
        f"Hidden state at STATE-token, VLM layer {NUM_VLM_LAYERS - 1}  |  "
        f"Explained variance: {exp_var:.1f}%\n"
        f"Well-separated clusters = model has distinct internal task modes",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "D_task_pca.png")
    plt.savefig(out, bbox_inches="tight", dpi=130)
    print(f"[D] Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS E  –  Same-Frame Attention Shift: Sort vs. Stack Prompt
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis_E():
    """
    VL-Think style: take ONE frame from each task, run with BOTH prompts,
    and visualise the attention map for a shared probe token (e.g. "red").
    Shows how the language prompt re-routes visual attention on the same image.
    """
    print("\n" + "=" * 60)
    print("Analysis E: Same-Frame Attention Shift (Sort vs. Stack Prompt)")
    print("=" * 60)

    # Token that appears in both prompts
    shared_tokens = [
        w for w in PROBE_TOKENS
        if (find_token_pos(SORT_PROMPT, w) is not None
            and find_token_pos(STACK_PROMPT, w) is not None)
    ]
    if not shared_tokens:
        print("[E] No token found in both prompts — skipping.")
        return
    SHARED_TOKEN = shared_tokens[0]
    print(f"  [E] Using shared token: '{SHARED_TOKEN}'")

    frame_setups = [
        {"name": "Frame from Sorting ep",  "ep_idx": SORT_EPISODES[0],  "frame_off": 0},
        {"name": "Frame from Stacking ep", "ep_idx": STACK_EPISODES[0], "frame_off": 0},
    ]
    prompts = [
        {"label": "Sort prompt",  "text": SORT_PROMPT},
        {"label": "Stack prompt", "text": STACK_PROMPT},
    ]

    n_frames  = len(frame_setups)
    n_prompts = len(prompts)
    n_cams    = 2   # front + top

    fig, axes = plt.subplots(
        n_frames * n_cams, n_prompts + 1,   # +1 col for raw image
        figsize=(4.0 * (n_prompts + 1), 3.6 * n_frames * n_cams),
        squeeze=False,
    )

    for f_i, fsetup in enumerate(frame_setups):
        ep_start, ep_end = get_frame_range(fsetup["ep_idx"])
        f_idx = min(ep_start + fsetup["frame_off"], ep_end - 1)
        item  = get_item(f_idx)

        display_cams = [
            ("observation.images.front", "front"),
            ("observation.images.top",   "top"),
        ]

        for c_i, (cam_src, cam_label) in enumerate(display_cams):
            row = f_i * n_cams + c_i
            orig_img = get_image_numpy(item, cam_src)

            # Column 0: raw image
            ax0 = axes[row, 0]
            if orig_img is not None:
                ax0.imshow(orig_img)
            else:
                ax0.text(0.5, 0.5, "N/A", ha="center", va="center",
                         transform=ax0.transAxes)
            ax0.set_xticks([]); ax0.set_yticks([])
            if c_i == 0:
                ax0.set_ylabel(fsetup["name"], fontsize=9, labelpad=4)
            if f_i == 0 and c_i == 0:
                ax0.set_title("Raw image", fontsize=10)

            # Columns 1+: attention under each prompt
            for p_i, prompt_cfg in enumerate(prompts):
                ax = axes[row, p_i + 1]
                if orig_img is None:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue

                tok_pos = find_token_pos(prompt_cfg["text"], SHARED_TOKEN)
                if tok_pos is None:
                    ax.text(0.5, 0.5, f"'{SHARED_TOKEN}'\nnot in prompt",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=8, color="grey")
                    ax.set_xticks([]); ax.set_yticks([])
                    if f_i == 0 and c_i == 0:
                        ax.set_title(prompt_cfg["label"], fontsize=10)
                    continue

                batch_pp = build_batch(item, prompt_cfg["text"])
                _, attn_layers, num_img_tokens, num_images = \
                    run_prefix_and_capture(batch_pp)

                valid = [attn_layers[l] for l in ATTN_LAYERS if l < len(attn_layers)]
                if not valid:
                    continue
                attn_avg    = torch.stack([a[0].float() for a in valid]).mean(0).mean(0)
                lang_offset = num_images * num_img_tokens
                seq_pos     = lang_offset + tok_pos
                cam_start   = c_i * num_img_tokens
                cam_end     = cam_start + num_img_tokens

                if seq_pos < attn_avg.shape[0]:
                    heat_1d = attn_avg[seq_pos, cam_start:cam_end].cpu()
                    overlay = attn_to_heatmap_overlay(heat_1d, orig_img)
                    ax.imshow(overlay)
                else:
                    ax.text(0.5, 0.5, "OOB", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="orange")
                ax.set_xticks([]); ax.set_yticks([])
                if f_i == 0 and c_i == 0:
                    ax.set_title(
                        f"{prompt_cfg['label']}\n"
                        f"token: '{SHARED_TOKEN}'", fontsize=9, fontweight="bold")
                if p_i == 0:
                    ax.set_ylabel(cam_label, fontsize=8)

    fig.suptitle(
        f"Analysis E ─ Same-Frame Attention Shift\n"
        f"Token '{SHARED_TOKEN}': where does the model look under each task prompt?\n"
        f"Different heatmaps on the SAME image = language genuinely re-routes attention",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "E_same_frame_attn_shift.png")
    plt.savefig(out, bbox_inches="tight", dpi=130)
    print(f"[E] Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Run all analyses
# ─────────────────────────────────────────────────────────────────────────────
run_analysis_A()
run_analysis_B()
run_analysis_C()
run_analysis_D()
run_analysis_E()

# ─────────────────────────────────────────────────────────────────────────────
# Write summary
# ─────────────────────────────────────────────────────────────────────────────
summary = f"""Task-Switch Analysis Summary
============================
Model checkpoint : {args.ckpt}
Dataset          : {args.repo_id}
Sort  prompt     : {SORT_PROMPT}
Stack prompt     : {STACK_PROMPT}
Sort  episodes   : {SORT_EPISODES}
Stack episodes   : {STACK_EPISODES}
Probe tokens     : {PROBE_TOKENS}
Attn layers avg  : {ATTN_LAYERS}
Frame stride     : {args.frame_stride}

Figures saved to: {OUT_DIR}
  A_per_token_heatmaps.png
  B_prompt_sensitivity.png
  C_cross_task_divergence.png
  D_task_pca.png
  E_same_frame_attn_shift.png

Interpretation guide
---------------------
A  Per-Token Heatmaps (VL-Think style):
   "red"/"green" tokens should highlight the correct coloured cube in BOTH tasks.
   "right"/"left"/"on" should attend to the correct spatial region.
   → Consistent heatmaps = preserved VLM grounding (good; avoids OpenVLA-style collapse).
   → Diffuse/wrong heatmaps = fine-tuning erased grounding (would be bad).

B  Prompt Sensitivity:
   wrong-task / opposite-color bars > 5° = model uses language to steer actions.
   If ALL bars are near 0°, the model ignores the prompt (bad for multi-task).

C  Cross-Task Divergence:
   Median divergence > 10° = language prompt is a strong driver of trajectory.
   Per-step plot shows whether errors compound across the action chunk.

D  Task Representation PCA:
   Two clearly separated clusters (Sorting vs. Stacking) under correct prompts
   = the model encodes task identity in its internal state.
   If wrong-prompt cluster overlaps the correct-prompt cluster, the model is
   confused by wrong prompts (indicates poor grounding, but may still work
   in practice if the visual context dominates).

E  Same-Frame Attention Shift:
   Same visual input, two prompts → different heatmap for token "red" / "green".
   This is the clearest single-image evidence that language re-routes visual attention.
"""

summary_path = os.path.join(OUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write(summary)
print(f"\n[DONE] Summary written to {summary_path}")
print("\n" + "=" * 60)
print(f"All analyses complete.  Output directory: {OUT_DIR}")
print("=" * 60)
