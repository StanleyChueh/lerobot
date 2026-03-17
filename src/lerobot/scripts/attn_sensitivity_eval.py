'''
Prompt Sensitivity Evaluation: SmolVLM2-500M vs SmolVLM2-2.2B
==============================================================

Scientific approach: measure how much each model's attention map
CHANGES when you change the prompt on the SAME image.

A model that truly reads the prompt should produce HIGH divergence
between different prompts. A model that ignores prompts will produce
nearly identical heatmaps regardless of what you say.

Metrics computed per model, per camera, averaged across sampled frames:
  - Prompt Sensitivity Score  : mean pairwise KL divergence across all prompt pairs
  - Attention Entropy         : mean entropy of each heatmap (low=focused, high=diffuse)
  - Cosine Similarity (same)  : self-consistency across frames (same prompt, different frames)
  - Correct vs Random KL      : KL between correct prompt and a random unrelated prompt

Usage:
    python src/lerobot/scripts/attn_sensitivity_eval.py \
        --repo_id "ethanCSL/svla_koch_sorting_n_stacking" \
        --episode 0 \
        --correct_prompt "Put the red cube in the right box, green cube in the left box." \
        --use_state

Output:
    attn_sensitivity_results/
        sensitivity_scores.png      ← bar chart comparing both models
        entropy_over_time.png       ← entropy per second for each model × prompt
        pairwise_kl_matrix.png      ← KL heatmap between all prompt pairs
        results_summary.txt         ← numerical summary
'''

import torch
import math
import numpy as np
import cv2
import os
import argparse
import json
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import cosine as cosine_dist

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

# ──────────────────────────────────────────────────────────────────────────────
# 1. Args
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--repo_id",        type=str, default="lerobot/svla_so100_pickplace")
parser.add_argument("--episode",        type=int, default=0)
parser.add_argument("--correct_prompt", type=str,
                    default="Put the red cube in the right box, green cube in the left box.")
parser.add_argument("--use_state",      action="store_true")
parser.add_argument("--video_backend",  type=str, default="pyav")
parser.add_argument("--offload",        action="store_true",
                    help="Move 500M to CPU between calls to save VRAM")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Probe prompt set
#    - correct_prompt  : the actual task
#    - opposite_prompt : same structure but wrong objects/locations
#    - unrelated_prompt: completely off-topic
#    - empty_prompt    : no task info at all
# ──────────────────────────────────────────────────────────────────────────────
PROBE_PROMPTS = {
    "correct":   args.correct_prompt,
    "opposite":  "Put the green cube in the right box, red cube in the left box.",
    "unrelated": "Navigate to the charging station and dock the robot arm.",
    "empty":     ".",
}
PROMPT_COLORS = {
    "correct":   "#2ecc71",
    "opposite":  "#e74c3c",
    "unrelated": "#3498db",
    "empty":     "#95a7b5",
}
print("\n[INFO] Probe prompts:")
for name, p in PROBE_PROMPTS.items():
    print(f"  {name:>10} : {p}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Build policies
# ──────────────────────────────────────────────────────────────────────────────
def build_policy(vlm_name, num_layers):
    cfg = SmolVLAConfig(
        vlm_model_name=vlm_name,
        load_vlm_weights=True,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=False,
        attention_mode="self_attn",
        device=str(device),
        empty_cameras=0,
        num_vlm_layers=num_layers,
    )
    cfg.normalization_mapping["STATE"] = NormalizationMode.IDENTITY
    cfg.input_features.update({
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.top":   PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    })
    pol = SmolVLAPolicy(config=cfg, dataset_stats=None).to(device)
    pol.reset()
    pol.model.vlm_with_expert.debug_attn = True
    return pol

print("\n[INFO] Loading SmolVLM2-500M ...")
policy_500m = build_policy("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", 8)

print("[INFO] Loading SmolVLM2-2.2B ...")
policy_2b   = build_policy("HuggingFaceTB/SmolVLM2-2.2B-Instruct", 12)

if args.offload:
    policy_500m.cpu()
    print("[INFO] --offload: 500M kept on CPU between passes")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Dataset
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Loading dataset: {args.repo_id}")
dataset     = LeRobotDataset(args.repo_id, root=None, batch_encoding_size=1,
                             video_backend=args.video_backend)
dataset_fps = dataset.fps
ep_meta     = dataset.meta.episodes[args.episode]
ep_length   = ep_meta["length"]
start_idx   = sum(dataset.meta.episodes[i]["length"] for i in range(args.episode))
end_idx     = start_idx + ep_length
print(f"[INFO] FPS={dataset_fps}, episode length={ep_length}, "
      f"~{ep_length // dataset_fps} seconds to process\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_attention(policy, item, prompt, dev):
    """
    Build batch from a dataset item + prompt and run prefix-only forward pass.
    Returns (heat_front_np, heat_top_np) — normalised to sum=1 (probability dist).
    """
    batch = {}
    for k, v in item.items():
        if k in ("observation.images.front", "observation.images.top"):
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            batch[k] = v.unsqueeze(0).to(dev)
        elif k == "observation.state" and args.use_state:
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            batch[OBS_STATE] = v.unsqueeze(0).to(dev)
    batch["task"] = prompt if prompt.endswith("\n") else prompt + "\n"

    policy.eval()
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tok = tokenizer(batch["task"], return_tensors="pt", padding="longest",
                    truncation=True, max_length=policy.config.tokenizer_max_length)
    batch[OBS_LANGUAGE_TOKENS]         = tok["input_ids"].to(dev)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].to(dev).bool()

    images, img_masks = policy.prepare_images(batch)
    with torch.no_grad():
        num_img_tok = int(policy.model.vlm_with_expert.embed_image(images[0]).shape[1])
    policy.model.vlm_with_expert._debug_num_img_tokens = num_img_tok
    policy.model.vlm_with_expert._debug_num_images     = len(images)

    if OBS_STATE in batch:
        state = policy.prepare_state(batch)
    else:
        bsz   = batch[OBS_LANGUAGE_TOKENS].shape[0]
        state = torch.zeros((bsz, policy.config.max_state_dim), device=dev, dtype=torch.float32)

    prefix_embs, prefix_pad, prefix_att = policy.model.embed_prefix(
        images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)

    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    att_2d  = make_att_2d_masks(prefix_pad, prefix_att)
    pos_ids = torch.cumsum(prefix_pad, dim=1) - 1

    with torch.no_grad():
        policy.model.vlm_with_expert.forward(
            attention_mask=att_2d, position_ids=pos_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True, fill_kv_cache=True)

    attn          = policy.model.vlm_with_expert.last_attn_weights[0].mean(0)   # [Q, K]
    num_lang      = batch[OBS_LANGUAGE_TOKENS].shape[1]
    img_start, img_end = 0, num_img_tok * len(images)
    lang_start    = img_end
    lang_end      = lang_start + num_lang

    lang_q        = attn[lang_start:lang_end].mean(0)                            # [K]
    h_front       = lang_q[img_start          : img_start + num_img_tok].detach().cpu()
    h_top         = lang_q[img_start + num_img_tok : img_end].detach().cpu()

    return _to_prob(h_front), _to_prob(h_top)


def _to_prob(heat_1d: torch.Tensor) -> np.ndarray:
    """Flatten, shift to ≥0, normalise to a probability distribution."""
    h = heat_1d.float().numpy()
    h = h - h.min()
    s = h.sum()
    return h / s if s > 1e-12 else np.ones_like(h) / len(h)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Symmetric KL: 0.5*(KL(P||Q) + KL(Q||P))"""
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    return float(0.5 * (scipy_entropy(p, q) + scipy_entropy(q, p)))


def attention_entropy(p: np.ndarray) -> float:
    """Shannon entropy of the attention distribution (higher = more diffuse)."""
    p = p + 1e-12;  p /= p.sum()
    return float(scipy_entropy(p))


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    return float(1.0 - cosine_dist(p.flatten(), q.flatten()))

# ──────────────────────────────────────────────────────────────────────────────
# 6. Storage for results
# ──────────────────────────────────────────────────────────────────────────────
prompt_names = list(PROBE_PROMPTS.keys())

# results[model_tag][camera][prompt_name] = list of values per second
results = {
    "500m": {"front": {p: [] for p in prompt_names},
             "top":   {p: [] for p in prompt_names}},
    "2.2b": {"front": {p: [] for p in prompt_names},
             "top":   {p: [] for p in prompt_names}},
}
# per-second pairwise KL: kl_per_sec[model_tag][camera][second_idx] = dict{(pA,pB): kl}
kl_per_sec = {m: {"front": {}, "top": {}} for m in ("500m", "2.2b")}

# ──────────────────────────────────────────────────────────────────────────────
# 7. Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
print("[INFO] Running evaluation...\n")
current_frame = 0
seconds_processed = 0

for global_idx in range(start_idx, end_idx):

    if current_frame % dataset_fps != 0:
        current_frame += 1
        continue

    second_idx = current_frame // dataset_fps
    item = dataset[global_idx]
    print(f"  Processing second {second_idx:04d} (frame {current_frame:05d}) ...", end=" ", flush=True)

    # Collect heatmaps for each model × each prompt
    maps = {"500m": {}, "2.2b": {}}

    for pname, ptxt in PROBE_PROMPTS.items():

        # --- 500M ---
        if args.offload:
            policy_500m.to(device)
        hf_500, ht_500 = run_attention(policy_500m, item, ptxt, device)
        if args.offload:
            policy_500m.cpu();  torch.cuda.empty_cache()

        # --- 2.2B ---
        hf_2b, ht_2b = run_attention(policy_2b, item, ptxt, device)

        maps["500m"][pname] = {"front": hf_500, "top": ht_500}
        maps["2.2b"][pname] = {"front": hf_2b,  "top": ht_2b}

        # Store entropy
        for model_tag, hf, ht in [("500m", hf_500, ht_500), ("2.2b", hf_2b, ht_2b)]:
            results[model_tag]["front"][pname].append(attention_entropy(hf))
            results[model_tag]["top"][pname].append(attention_entropy(ht))

    # Compute pairwise KL for this second
    for model_tag in ("500m", "2.2b"):
        for cam in ("front", "top"):
            kl_per_sec[model_tag][cam][second_idx] = {}
            for i, pA in enumerate(prompt_names):
                for pB in prompt_names[i+1:]:
                    kl = kl_divergence(maps[model_tag][pA][cam],
                                       maps[model_tag][pB][cam])
                    kl_per_sec[model_tag][cam][second_idx][(pA, pB)] = kl

    print("done")
    seconds_processed += 1
    current_frame += 1

print(f"\n[INFO] Processed {seconds_processed} seconds total.\n")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Aggregate statistics
# ──────────────────────────────────────────────────────────────────────────────
out_dir = "attn_sensitivity_results"
os.makedirs(out_dir, exist_ok=True)

def aggregate_kl(model_tag, cam):
    """Mean pairwise KL averaged over all seconds and all prompt pairs."""
    all_vals = []
    for sec_dict in kl_per_sec[model_tag][cam].values():
        all_vals.extend(sec_dict.values())
    return float(np.mean(all_vals)) if all_vals else 0.0


def pairwise_kl_matrix(model_tag, cam):
    """N×N matrix of mean KL between every prompt pair."""
    n = len(prompt_names)
    mat = np.zeros((n, n))
    for i, pA in enumerate(prompt_names):
        for j, pB in enumerate(prompt_names):
            if i == j:
                continue
            key = (pA, pB) if (pA, pB) in next(iter(kl_per_sec[model_tag][cam].values()), {}) \
                            else (pB, pA)
            vals = [kl_per_sec[model_tag][cam][s].get(key, 0.0)
                    for s in kl_per_sec[model_tag][cam]]
            mat[i, j] = float(np.mean(vals))
    return mat

# ──────────────────────────────────────────────────────────────────────────────
# 9. Plot 1: Prompt Sensitivity Score (mean pairwise KL) — bar chart
# ──────────────────────────────────────────────────────────────────────────────
cameras    = ["front", "top"]
model_tags = ["500m", "2.2b"]
labels     = ["500M (0.45B)\nFront", "500M (0.45B)\nTop",
              "2.2B\nFront",          "2.2B\nTop"]
scores     = [aggregate_kl(m, c) for m in model_tags for c in cameras]
colors     = ["#3498db", "#85c1e9", "#e74c3c", "#f1948a"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.8)
ax.set_ylabel("Prompt Sensitivity Score\n(mean pairwise symmetric KL divergence)", fontsize=11)
ax.set_title("Higher = More Prompt-Sensitive Attention\n"
             "(model changes attention more when you change the prompt)", fontsize=12)
for bar, val in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(scores) * 1.25)
ax.axhline(0, color="black", linewidth=0.5)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "sensitivity_scores.png"), dpi=150)
plt.close(fig)
print("[PLOT] sensitivity_scores.png")

# ──────────────────────────────────────────────────────────────────────────────
# 10. Plot 2: Pairwise KL matrices (500M vs 2.2B, front + top)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Mean Pairwise KL Divergence Between Prompts\n"
             "(higher diagonal-off values = more prompt-sensitive)", fontsize=13)

for row, cam in enumerate(cameras):
    for col, model_tag in enumerate(model_tags):
        mat  = pairwise_kl_matrix(model_tag, cam)
        ax   = axes[row][col]
        im   = ax.imshow(mat, cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(prompt_names)));  ax.set_xticklabels(prompt_names, rotation=30, ha="right")
        ax.set_yticks(range(len(prompt_names)));  ax.set_yticklabels(prompt_names)
        ax.set_title(f"{model_tag.upper()}  —  {cam} camera")
        fig.colorbar(im, ax=ax, fraction=0.046)
        for i in range(len(prompt_names)):
            for j in range(len(prompt_names)):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if mat[i,j] < mat.max() * 0.6 else "white")

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "pairwise_kl_matrix.png"), dpi=150)
plt.close(fig)
print("[PLOT] pairwise_kl_matrix.png")

# ──────────────────────────────────────────────────────────────────────────────
# 11. Plot 3: Attention Entropy over time (correct prompt only)
# ──────────────────────────────────────────────────────────────────────────────
seconds = list(range(seconds_processed))
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
fig.suptitle("Attention Entropy Over Time (correct prompt)\n"
             "Lower = more focused. If entropy is always high → model is confused.", fontsize=12)

for col, cam in enumerate(cameras):
    ax = axes[col]
    for model_tag, color, ls in [("500m", "#3498db", "-"), ("2.2b", "#e74c3c", "--")]:
        ent_vals = results[model_tag][cam]["correct"]
        ax.plot(seconds[:len(ent_vals)], ent_vals, label=model_tag.upper(),
                color=color, linestyle=ls, linewidth=1.5)
    ax.set_xlabel("Second")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title(f"{cam.capitalize()} camera")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "entropy_over_time.png"), dpi=150)
plt.close(fig)
print("[PLOT] entropy_over_time.png")

# ──────────────────────────────────────────────────────────────────────────────
# 12. Plot 4: Correct vs Unrelated KL over time (most diagnostic)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("KL Divergence: Correct Prompt vs Unrelated Prompt (per second)\n"
             "Higher = model actually changed attention when prompt changed", fontsize=12)

for col, cam in enumerate(cameras):
    ax = axes[col]
    for model_tag, color, ls in [("500m", "#3498db", "-"), ("2.2b", "#e74c3c", "--")]:
        kl_vals = [kl_per_sec[model_tag][cam][s].get(
                       ("correct", "unrelated"), kl_per_sec[model_tag][cam][s].get(
                       ("unrelated", "correct"), 0.0))
                   for s in sorted(kl_per_sec[model_tag][cam].keys())]
        ax.plot(range(len(kl_vals)), kl_vals, label=model_tag.upper(),
                color=color, linestyle=ls, linewidth=1.5)
        ax.axhline(np.mean(kl_vals), color=color, linestyle=":", linewidth=1,
                   label=f"{model_tag.upper()} mean={np.mean(kl_vals):.4f}")
    ax.set_xlabel("Second")
    ax.set_ylabel("Symmetric KL Divergence")
    ax.set_title(f"{cam.capitalize()} camera")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "correct_vs_unrelated_kl.png"), dpi=150)
plt.close(fig)
print("[PLOT] correct_vs_unrelated_kl.png")

# ──────────────────────────────────────────────────────────────────────────────
# 13. Numerical summary
# ──────────────────────────────────────────────────────────────────────────────
summary_lines = [
    "=" * 70,
    "PROMPT SENSITIVITY EVALUATION SUMMARY",
    f"Dataset     : {args.repo_id}  (episode {args.episode})",
    f"Seconds     : {seconds_processed}",
    f"Prompts     : {json.dumps(PROBE_PROMPTS, indent=2)}",
    "=" * 70,
    "",
    "── Prompt Sensitivity Score (mean pairwise symmetric KL) ──────────────",
    "   Higher is better: model changes attention more when prompt changes.",
    "",
]
for model_tag in model_tags:
    for cam in cameras:
        score = aggregate_kl(model_tag, cam)
        summary_lines.append(f"  {model_tag.upper():6s}  {cam:5s}  →  {score:.6f}")

summary_lines += [
    "",
    "── Attention Entropy (correct prompt, mean over time) ─────────────────",
    "   Lower = more focused attention.",
    "",
]
for model_tag in model_tags:
    for cam in cameras:
        ent = float(np.mean(results[model_tag][cam]["correct"]))
        summary_lines.append(f"  {model_tag.upper():6s}  {cam:5s}  →  entropy = {ent:.4f} nats")

summary_lines += [
    "",
    "── Correct vs Unrelated KL (mean over time) ───────────────────────────",
    "   The single most diagnostic number: does attention shift for wrong prompt?",
    "",
]
for model_tag in model_tags:
    for cam in cameras:
        kl_vals = [kl_per_sec[model_tag][cam][s].get(
                       ("correct","unrelated"), kl_per_sec[model_tag][cam][s].get(
                       ("unrelated","correct"), 0.0))
                   for s in sorted(kl_per_sec[model_tag][cam].keys())]
        summary_lines.append(
            f"  {model_tag.upper():6s}  {cam:5s}  →  KL = {np.mean(kl_vals):.6f}  "
            f"(std {np.std(kl_vals):.6f})"
        )

summary_lines.append("")
summary_lines.append("=" * 70)

summary_txt = "\n".join(summary_lines)
print("\n" + summary_txt)

txt_path = os.path.join(out_dir, "results_summary.txt")
with open(txt_path, "w") as f:
    f.write(summary_txt + "\n")

print(f"\n[DONE] All outputs saved to: {os.path.abspath(out_dir)}/")
print("  sensitivity_scores.png       ← main bar chart")
print("  pairwise_kl_matrix.png       ← full N×N KL between all prompt pairs")
print("  entropy_over_time.png        ← attention focus over episode")
print("  correct_vs_unrelated_kl.png  ← most diagnostic: does prompt actually matter?")
print("  results_summary.txt          ← all numbers in plain text")
