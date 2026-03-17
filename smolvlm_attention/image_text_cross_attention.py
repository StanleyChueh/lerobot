"""
SmolVLM-500M Attention Visualization
=====================================
Given a single image + text prompt, this script:
  1. Loads SmolVLM-500M-Instruct with output_attentions=True
  2. Runs a forward pass and captures all self-attention weights
  2b. Reconstructs an attention-only decoder hidden-state rollout and
      compares it against the real hidden states
  3. Identifies which token positions are image tokens vs text tokens
  4. Plots:
     (a) Input image
     (b) Token-type map (image vs text)
     (c) Per-token image/text attention fraction (stacked bar)
     (d) Mean attention heatmap (avg across all layers & heads)
     (e) Last-layer attention heatmap
     (f) Per-layer image attention from the last query token
     (g) Attention overlaid on image patches (spatial heatmap)
  5. Generates the model's response

Outputs are saved to smolvlm_attention/results_500m/
"""

import os
import math
import glob
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / save-only
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle as MplRect
from PIL import Image
import torch

warnings.filterwarnings("ignore")

from transformers import AutoProcessor, AutoModelForImageTextToText

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID    = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMAGE_PATH  = os.path.join(os.path.dirname(__file__), "input_images", "single_cam_sorting_front.png")
PROMPT_TARGET_PAIRS = [
    ("Do you see green block", "green"),
    ("Do you see red block", "red"),
    # "blue" is absent from the scene → model should answer "No"
    # A Yes-vs-No contrast gives a much stronger gradient signal difference.
    ("Do you see blue block", "blue"),
]
# Active pair for the main single-run analysis below.
# Change the index (0 / 1 / …) to switch which prompt drives the main plots.
prompt, TARGET_WORD = PROMPT_TARGET_PAIRS[0]
PROMPT = prompt
OUT_DIR     = os.path.join(os.path.dirname(__file__), "results_500m")
os.makedirs(OUT_DIR, exist_ok=True)

# Approximate ground-truth boxes for the sample image used in this script.
# Coordinates are in the original image pixel frame as (x0, y0, x1, y1), half-open.
GROUND_TRUTH_BOXES = {
    "single_cam_sorting_front.png": {
        "green": [(82, 272, 146, 336)],
        # The red target appears in multiple connected regions in this frame.
        "red": [(244, 108, 315, 213), (260, 278, 310, 352)],
    }
}

GLOBAL_VIEW_WEIGHT = 0.15


def resolve_local_snapshot(model_id):
    repo_dir = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"),
        f"models--{model_id.replace('/', '--')}",
        "snapshots",
    )
    candidates = sorted(glob.glob(os.path.join(repo_dir, "*")))
    return candidates[-1] if candidates else None


def normalise_for_display(x):
    x = np.asarray(x, dtype=float)
    p05 = np.percentile(x, 5)
    p95 = np.percentile(x, 95)
    return np.clip((x - p05) / max(p95 - p05, 1e-9), 0.0, 1.0)


def resize_map(x, width, height):
    if width <= 0 or height <= 0:
        return np.zeros((max(0, height), max(0, width)), dtype=float)
    return np.array(
        Image.fromarray(np.asarray(x, dtype=np.float32), mode="F").resize(
            (width, height), Image.BILINEAR
        ),
        dtype=float,
    )


def infer_image_token_geometry(processor, model, image, n_img_tokens):
    probe = processor.image_processor([[image]], return_row_col_info=True)
    rows = int(probe["rows"][0][0]) if probe.get("rows") else 0
    cols = int(probe["cols"][0][0]) if probe.get("cols") else 0

    image_seq_len = int(
        getattr(processor, "image_seq_len", getattr(model, "image_seq_len", 0))
    )
    if image_seq_len <= 0:
        raise ValueError("Could not determine SmolVLM image_seq_len from processor/model.")

    block_side = int(round(math.sqrt(image_seq_len)))
    if block_side * block_side != image_seq_len:
        raise ValueError(f"image_seq_len={image_seq_len} is not a square number.")

    local_views = rows * cols if rows > 0 and cols > 0 else 0
    total_views = local_views + 1 if local_views else 1
    if total_views * image_seq_len != n_img_tokens:
        fallback_views = max(1, n_img_tokens // image_seq_len)
        if fallback_views * image_seq_len != n_img_tokens:
            raise ValueError(
                f"Image tokens ({n_img_tokens}) do not align with image_seq_len={image_seq_len}."
            )
        rows = cols = 0
        local_views = 0
        total_views = fallback_views

    return {
        "rows": rows,
        "cols": cols,
        "image_seq_len": image_seq_len,
        "block_side": block_side,
        "local_views": local_views,
        "total_views": total_views,
    }


def split_patch_vector(patch_vector, geom):
    patch_vector = np.asarray(patch_vector, dtype=float)
    expected = geom["total_views"] * geom["image_seq_len"]
    if patch_vector.shape[0] != expected:
        raise ValueError(f"Expected {expected} image tokens, got {patch_vector.shape[0]}.")
    return patch_vector.reshape(geom["total_views"], geom["image_seq_len"])


def project_patch_vector_to_canvas(patch_vector, geom, image_size, include_global=False):
    img_w, img_h = image_size
    block_side = geom["block_side"]
    blocks = split_patch_vector(patch_vector, geom)

    if geom["local_views"] == 0:
        return resize_map(blocks[0].reshape(block_side, block_side), img_w, img_h)

    canvas = np.zeros((img_h, img_w), dtype=float)
    weights = np.zeros((img_h, img_w), dtype=float)
    x_edges = np.rint(np.linspace(0, img_w, geom["cols"] + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, img_h, geom["rows"] + 1)).astype(int)

    block_idx = 0
    for row in range(geom["rows"]):
        for col in range(geom["cols"]):
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]
            patch_grid = blocks[block_idx].reshape(block_side, block_side)
            patch_canvas = resize_map(patch_grid, x1 - x0, y1 - y0)
            canvas[y0:y1, x0:x1] += patch_canvas
            weights[y0:y1, x0:x1] += 1.0
            block_idx += 1

    if include_global and geom["total_views"] > geom["local_views"]:
        global_canvas = resize_map(blocks[-1].reshape(block_side, block_side), img_w, img_h)
        canvas += GLOBAL_VIEW_WEIGHT * global_canvas
        weights += GLOBAL_VIEW_WEIGHT

    weights[weights < 1e-12] = 1.0
    return canvas / weights


def lookup_target_boxes(image_path, target_word):
    image_name = os.path.basename(image_path)
    return GROUND_TRUTH_BOXES.get(image_name, {}).get(target_word.lower().strip(), [])


def evaluate_localisation(canvas_raw, boxes):
    if not boxes:
        return None

    mask = np.zeros_like(canvas_raw, dtype=bool)
    img_h, img_w = canvas_raw.shape
    for x0, y0, x1, y1 in boxes:
        x0 = max(0, min(img_w, int(x0)))
        x1 = max(0, min(img_w, int(x1)))
        y0 = max(0, min(img_h, int(y0)))
        y1 = max(0, min(img_h, int(y1)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True

    if not mask.any():
        return None

    peak_y, peak_x = np.unravel_index(np.argmax(canvas_raw), canvas_raw.shape)
    total_mass = max(float(canvas_raw.sum()), 1e-12)
    return {
        "peak_xy": (int(peak_x), int(peak_y)),
        "pointing_hit": bool(mask[peak_y, peak_x]),
        "mass_in_box": float(canvas_raw[mask].sum() / total_mass),
    }


def pick_attribution_token(logits, tokenizer, top_k=12):
    top_ids = torch.topk(logits, k=min(top_k, logits.shape[-1])).indices.tolist()
    for token_id in top_ids:
        text = tokenizer.decode([token_id], skip_special_tokens=True).strip()
        if text:
            return token_id, text

    token_id = int(torch.argmax(logits).item())
    text = tokenizer.decode([token_id], skip_special_tokens=True).strip() or tokenizer.convert_ids_to_tokens([token_id])[0]
    return token_id, text


def flatten_cosine(a, b):
    a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
    denom = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


def top_token_strings(logits, tokenizer, top_k=5):
    top_ids = torch.topk(logits, k=min(top_k, logits.shape[-1])).indices.tolist()
    toks = []
    for token_id in top_ids:
        text = tokenizer.decode([token_id], skip_special_tokens=True).strip()
        if not text:
            text = tokenizer.convert_ids_to_tokens([token_id])[0]
        toks.append(text)
    return toks


def analyse_decoder_self_attention_rollout(
    model,
    processor,
    inputs,
    img_positions,
    txt_positions,
    device,
):
    text_layers = model.model.text_model.layers
    attn_outputs = [None] * len(text_layers)
    handles = []

    def make_hook(layer_idx):
        def _hook(module, args, kwargs, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            attn_outputs[layer_idx] = attn_out.detach().cpu()

        return _hook

    for layer_idx, layer in enumerate(text_layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(layer_idx), with_kwargs=True))

    try:
        with torch.no_grad():
            traced = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    hidden_states = [hs[0].detach().cpu() for hs in traced.hidden_states]
    usable_layers = min(len(text_layers), len(hidden_states) - 1)
    attn_only_states = [hidden_states[0].clone()]
    per_layer = []

    for layer_idx in range(usable_layers):
        attn_out = attn_outputs[layer_idx]
        if attn_out is None:
            raise RuntimeError(f"Missing captured self-attention output for decoder layer {layer_idx}.")

        attn_delta = attn_out[0]
        full_before = hidden_states[layer_idx]
        full_after = hidden_states[layer_idx + 1]
        full_delta = full_after - full_before
        attn_only_next = attn_only_states[-1] + attn_delta
        mlp_residual = full_after - attn_only_next

        per_layer.append(
            {
                "layer": layer_idx,
                "attn_vs_full_delta_cos": flatten_cosine(attn_delta, full_delta),
                "rollout_vs_full_hidden_cos": flatten_cosine(attn_only_next, full_after),
                "attn_norm_ratio": float(
                    torch.linalg.vector_norm(attn_delta).item()
                    / max(torch.linalg.vector_norm(full_delta).item(), 1e-12)
                ),
                "mlp_residual_norm_ratio": float(
                    torch.linalg.vector_norm(mlp_residual).item()
                    / max(torch.linalg.vector_norm(full_after).item(), 1e-12)
                ),
            }
        )
        attn_only_states.append(attn_only_next)

    final_full = hidden_states[usable_layers]
    final_attn_only = attn_only_states[-1]
    final_metrics = {
        "all": flatten_cosine(final_attn_only, final_full),
        "image": flatten_cosine(final_attn_only[img_positions], final_full[img_positions])
        if len(img_positions) > 0
        else 0.0,
        "text": flatten_cosine(final_attn_only[txt_positions], final_full[txt_positions])
        if len(txt_positions) > 0
        else 0.0,
    }

    attn_only_in = final_attn_only.unsqueeze(0).to(device=device, dtype=model.lm_head.weight.dtype)
    with torch.no_grad():
        attn_only_logits = model.lm_head(model.model.text_model.norm(attn_only_in))[0, -1, :].detach().cpu()
    real_logits = traced.logits[0, -1, :].detach().cpu()

    real_top = top_token_strings(real_logits, processor.tokenizer, top_k=5)
    attn_top = top_token_strings(attn_only_logits, processor.tokenizer, top_k=5)

    if final_metrics["all"] >= 0.90:
        verdict = "strong: self-attention alone preserves most decoder-state direction"
    elif final_metrics["all"] >= 0.75:
        verdict = "partial: self-attention carries meaningful signal, but MLP/residual mixing still matters"
    else:
        verdict = "weak: self-attention alone does not stay close to the real decoder state"

    return {
        "per_layer": per_layer,
        "final_hidden_cosine": final_metrics,
        "real_top_tokens": real_top,
        "attn_only_top_tokens": attn_top,
        "real_next_token": real_top[0] if real_top else "",
        "attn_only_next_token": attn_top[0] if attn_top else "",
        "verdict": verdict,
    }


def plot_self_attention_rollout_diagnostics(report, out_path):
    layers = [row["layer"] for row in report["per_layer"]]
    attn_delta_cos = [row["attn_vs_full_delta_cos"] for row in report["per_layer"]]
    rollout_cos = [row["rollout_vs_full_hidden_cos"] for row in report["per_layer"]]
    attn_norm_ratio = [row["attn_norm_ratio"] for row in report["per_layer"]]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, facecolor="white")
    fig.suptitle("SmolVLM Decoder Self-Attention Rollout Diagnostics", fontsize=13, fontweight="bold")

    axes[0].plot(layers, attn_delta_cos, marker="o", color="#c0392b")
    axes[0].axhline(0.9, linestyle="--", linewidth=1, color="#999999")
    axes[0].set_ylabel("cos(attn delta,\nfull layer delta)")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)

    axes[1].plot(layers, rollout_cos, marker="o", color="#1f618d")
    axes[1].axhline(0.9, linestyle="--", linewidth=1, color="#999999")
    axes[1].set_ylabel("cos(attn-only hidden,\nreal hidden)")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)

    axes[2].plot(layers, attn_norm_ratio, marker="o", color="#117a65")
    axes[2].axhline(1.0, linestyle="--", linewidth=1, color="#999999")
    axes[2].set_ylabel("||attn delta|| /\n||full layer delta||")
    axes[2].set_xlabel("Decoder layer")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_answer_conditioned_attribution(
    p_str,
    image,
    processor,
    model,
    device,
    img_positions_override=None,
    answer_token_id=None,
    N_SEM=8,
):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p_str}]}]
    pt   = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inp  = processor(text=pt, images=[image], return_tensors="pt").to(device)
    ids  = inp["input_ids"][0]

    img_tok_id = model.config.image_token_id
    img_pos_l  = np.where((ids == img_tok_id).cpu().numpy())[0]
    if img_positions_override is not None:
        img_pos_l = np.asarray(img_positions_override, dtype=int)

    bare_ids = processor.tokenizer.encode(p_str, add_special_tokens=False)
    full_ids = ids.tolist()
    start = next(
        (si for si in range(len(full_ids) - len(bare_ids) + 1)
         if full_ids[si : si + len(bare_ids)] == bare_ids),
        None,
    )
    if start is None:
        raise ValueError(f"Prompt not found in token sequence for: {p_str!r}")
    prompt_positions = list(range(start, start + len(bare_ids)))

    frozen_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad_(False)
            frozen_params.append(name)

    model.zero_grad()
    torch.cuda.empty_cache()

    try:
        input_embeds = model.get_input_embeddings()(inp["input_ids"]).detach()
        input_embeds.requires_grad_(True)

        with torch.enable_grad():
            out = model(
                input_ids=inp["input_ids"],
                inputs_embeds=input_embeds,
                attention_mask=inp.get("attention_mask"),
                pixel_values=inp.get("pixel_values"),
                pixel_attention_mask=inp.get("pixel_attention_mask"),
                output_attentions=True,
                use_cache=False,
            )

        logits = out.logits[0, -1, :]
        if answer_token_id is None:
            answer_token_id, answer_token_text = pick_attribution_token(logits, processor.tokenizer)
        else:
            answer_token_text = (
                processor.tokenizer.decode([answer_token_id], skip_special_tokens=True).strip()
                or processor.tokenizer.convert_ids_to_tokens([answer_token_id])[0]
            )

        atts = out.attentions
        n_layers_l = len(atts)
        sem_lyrs_l = list(range(n_layers_l - min(N_SEM, n_layers_l), n_layers_l))
        for l in sem_lyrs_l:
            if atts[l] is not None and atts[l].requires_grad:
                atts[l].retain_grad()

        score = logits[answer_token_id]
        score.backward()

        prompt_scores = input_embeds.grad[0].norm(dim=-1).detach().cpu().numpy()

        maps = []
        n_with_grad = 0
        for l in sem_lyrs_l:
            a = atts[l][0]
            grad = atts[l].grad

            a_np = a.detach().cpu().numpy()
            if grad is not None:
                g_np = grad[0].detach().cpu().numpy()
                n_with_grad += 1
            else:
                g_np = np.ones_like(a_np)

            last_q = a_np.shape[2] - 1
            ga = np.abs(g_np[:, last_q, :][:, img_pos_l]) * a_np[:, last_q, :][:, img_pos_l]
            ga = ga / np.maximum(ga.sum(1, keepdims=True), 1e-12)
            maps.append(ga.mean(0))

    finally:
        for name, param in model.named_parameters():
            if name in frozen_params:
                param.requires_grad_(True)
        model.zero_grad()
        torch.cuda.empty_cache()

    return {
        "prompt_scores": prompt_scores,
        "patch_scores": np.mean(maps, axis=0),
        "prompt_positions": prompt_positions,
        "answer_token_id": int(answer_token_id),
        "answer_token_text": answer_token_text,
        "grad_layers_available": n_with_grad,
    }

# ──────────────────────────────────────────────────────────────────────────────
#  1. Load model & processor
# ──────────────────────────────────────────────────────────────────────────────
print("Loading SmolVLM-500M-Instruct …")
MODEL_SOURCE = resolve_local_snapshot(MODEL_ID) or MODEL_ID
if MODEL_SOURCE != MODEL_ID:
    print(f"  ✓ using local snapshot: {MODEL_SOURCE}")

processor = AutoProcessor.from_pretrained(
    MODEL_SOURCE,
    local_files_only=(MODEL_SOURCE != MODEL_ID),
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_SOURCE,
    dtype=torch.float32,
    attn_implementation="eager",   # required for output_attentions=True
    local_files_only=(MODEL_SOURCE != MODEL_ID),
)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"  ✓ loaded on {device}")

# ──────────────────────────────────────────────────────────────────────────────
#  2. Load image
# ──────────────────────────────────────────────────────────────────────────────
print("Loading image …")
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"  ✓ image path: {IMAGE_PATH}")
print(f"  ✓ image size: {image.size}")

# ──────────────────────────────────────────────────────────────────────────────
#  3. Build prompt and tokenise
# ──────────────────────────────────────────────────────────────────────────────
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }
]
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(device)
input_ids = inputs["input_ids"][0]          # (seq_len,)
seq_len   = input_ids.shape[0]
print(f"  ✓ tokenised: {seq_len} tokens")

# ──────────────────────────────────────────────────────────────────────────────
#  4. Forward pass — capture attention weights
# ──────────────────────────────────────────────────────────────────────────────
print("Running forward pass with output_attentions=True …")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# outputs.attentions: tuple[num_layers] of (batch, heads, seq, seq)
attentions = outputs.attentions
num_layers = len(attentions)
num_heads  = attentions[0].shape[1]
print(f"  ✓ captured {num_layers} layers × {num_heads} heads")

# ──────────────────────────────────────────────────────────────────────────────
#  5. Identify image / text token positions
# ──────────────────────────────────────────────────────────────────────────────
image_token_id = model.config.image_token_id
is_image       = (input_ids == image_token_id).cpu().numpy()   # bool (seq_len,)
img_positions  = np.where(is_image)[0]
txt_positions  = np.where(~is_image)[0]
print(f"  ✓ image tokens: {len(img_positions)}   text tokens: {len(txt_positions)}")

tokens = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
token_labels = [
    "<img>" if is_image[i] else (tok.replace("▁", "_") or f"[{i}]")
    for i, tok in enumerate(tokens)
]

# ──────────────────────────────────────────────────────────────────────────────
#  5b. Decoder self-attention rollout → hidden-state simulation
# ──────────────────────────────────────────────────────────────────────────────
print("Tracing decoder self-attention contributions …")
self_attn_rollout = analyse_decoder_self_attention_rollout(
    model,
    processor,
    inputs,
    img_positions,
    txt_positions,
    device,
)
rollout_plot_path = os.path.join(OUT_DIR, "self_attention_hidden_state_rollout_500m.png")
plot_self_attention_rollout_diagnostics(self_attn_rollout, rollout_plot_path)
print(f"  ✓ saved rollout diagnostics → {rollout_plot_path}")
print(
    "  ✓ final hidden cosine:"
    f" all={self_attn_rollout['final_hidden_cosine']['all']:.4f}"
    f" image={self_attn_rollout['final_hidden_cosine']['image']:.4f}"
    f" text={self_attn_rollout['final_hidden_cosine']['text']:.4f}"
)
print(
    f"  ✓ real next token vs attn-only next token: "
    f"\"{self_attn_rollout['real_next_token']}\" vs "
    f"\"{self_attn_rollout['attn_only_next_token']}\""
)
print(f"  ✓ self-attention verdict: {self_attn_rollout['verdict']}")

# ──────────────────────────────────────────────────────────────────────────────
#  6. Aggregate attention matrices
# ──────────────────────────────────────────────────────────────────────────────
# mean_attn_layers[l] → (seq, seq)  head-averaged per layer
mean_attn_layers = [
    attentions[l][0].mean(dim=0).cpu().numpy()
    for l in range(num_layers)
]

# global mean across all layers & heads
mean_attn = np.stack(mean_attn_layers).mean(axis=0)   # (seq, seq)
last_attn  = mean_attn_layers[-1]                      # (seq, seq)

# ──────────────────────────────────────────────────────────────────────────────
#  7. Per-token image vs text attention fraction
# ──────────────────────────────────────────────────────────────────────────────
img_frac = mean_attn[:, img_positions].sum(axis=1)  # (seq,)
txt_frac = mean_attn[:, txt_positions].sum(axis=1)  # (seq,)

# ──────────────────────────────────────────────────────────────────────────────
#  8b. Spatial attention: content-word queries → image patches
#
#  Key insight: using "?" or averaging ALL text tokens washes out signal
#  because punctuation self-attends and 832 image tokens dilute the softmax.
#  Instead: use only CONTENT WORDS ("red", "cube", …) as queries, only from
#  the last N_SEM SEMANTIC layers (early layers do positional/syntactic work).
#  Row-normalise within image positions so each query donates 1 unit of
#  attention across patches, removing the 876-token denominator effect.
# ──────────────────────────────────────────────────────────────────────────────
n_img_pre = len(img_positions)
N_SEM     = min(8, num_layers)   # how many late layers to treat as semantic
sem_layers = list(range(num_layers - N_SEM, num_layers))

# ── Locate exact prompt-word positions (reused in section 9b) ────────────────
_prompt_ids_bare = processor.tokenizer.encode(prompt, add_special_tokens=False)
_full_ids_list   = input_ids.tolist()
_prompt_start    = None
for _si in range(len(_full_ids_list) - len(_prompt_ids_bare) + 1):
    if _full_ids_list[_si : _si + len(_prompt_ids_bare)] == _prompt_ids_bare:
        _prompt_start = _si
        break

if _prompt_start is not None:
    _p_positions = list(range(_prompt_start, _prompt_start + len(_prompt_ids_bare)))
    _p_toks = processor.tokenizer.convert_ids_to_tokens(
        [_full_ids_list[i] for i in _p_positions]
    )
    _p_words = [t.replace("\u2581", " ").replace("\u0120", " ").strip() or t for t in _p_toks]
else:
    _p_positions = [int(p) for p in txt_positions]
    _p_words = [
        tokens[i].replace("\u2581", " ").replace("\u0120", " ").strip() or tokens[i]
        for i in txt_positions
    ]

# Content words = substantive, non-stop-word prompt tokens
STOPWORDS = {"do", "you", "see", "the", "a", "an", "is", "are", "in", "on", "at",
             "it", "?", ".", ",", "!", ";", ":", "and", "or", "not",
             "User", "assistant", "<end_of_utterance>"}
content_positions = [
    pos for pos, w in zip(_p_positions, _p_words)
    if w.lower().strip() not in STOPWORDS and w.strip() and not w.startswith("<")
]
if not content_positions:
    content_positions = _p_positions   # fallback: all prompt tokens

_query_words = [_p_words[_p_positions.index(p)] for p in content_positions
                if p in _p_positions]
print(f"  ✓ content word queries for spatial map: {_query_words}")

# ── Spatial map: conservative aggregation ─────────────────────────────────────
# Use only content-word query tokens from late semantic layers.
# For each query token:
#   1) restrict attention to image-token positions
#   2) renormalize inside the image-token subspace
#   3) average across heads within each late layer
#   4) average across late layers
# Finally average across content-word queries.
# This is less visually sharp than head-selection heuristics, but more stable.
per_query_maps = []

for q_pos in content_positions:
    q_maps = []
    for l in sem_layers:
        a = attentions[l][0].cpu().numpy()   # (heads, seq, seq)
        row_img_heads = a[:, q_pos, :][:, img_positions]   # (heads, n_img_tokens)

        # normalize only inside image-token subspace
        denom = row_img_heads.sum(axis=1, keepdims=True)
        denom[denom < 1e-12] = 1.0
        row_img_heads = row_img_heads / denom

        # average all heads in this layer
        q_maps.append(row_img_heads.mean(axis=0))

    # average late layers for this query
    per_query_maps.append(np.mean(q_maps, axis=0))

# average across content words
spatial_attn_raw = np.mean(per_query_maps, axis=0)

print(
    f"  ✓ spatial_attn_raw: range=[{spatial_attn_raw.min():.5f}, "
    f"{spatial_attn_raw.max():.5f}], std={spatial_attn_raw.std():.6f}"
)

# ──────────────────────────────────────────────────────────────────────────────
#  9. Spatial attention map → image patch grid
# ──────────────────────────────────────────────────────────────────────────────
n_img_tokens = len(img_positions)
image_geom = infer_image_token_geometry(processor, model, image, n_img_tokens)
print(
    "  ✓ SmolVLM image-token geometry:"
    f" locals={image_geom['rows']}x{image_geom['cols']}"
    f"  blocks={image_geom['total_views']}"
    f"  tokens_per_block={image_geom['image_seq_len']}"
    f"  block_side={image_geom['block_side']}"
)

spatial_canvas_raw = project_patch_vector_to_canvas(
    spatial_attn_raw,
    image_geom,
    image.size,
    include_global=False,
)
spatial_canvas_display = normalise_for_display(spatial_canvas_raw)

hot_y, hot_x = np.unravel_index(np.argmax(spatial_canvas_raw), spatial_canvas_raw.shape)
print(
    f"  ✓ content-word hotspot: pixel=({hot_x},{hot_y}) "
    f"raw range=[{spatial_canvas_raw.min():.6f}, {spatial_canvas_raw.max():.6f}]"
)

# ──────────────────────────────────────────────────────────────────────────────
#  9b. Answer-conditioned prompt + image attribution
# ──────────────────────────────────────────────────────────────────────────────
p_positions = _p_positions
p_words     = _p_words

main_attr = compute_answer_conditioned_attribution(
    prompt,
    image,
    processor,
    model,
    device,
    img_positions_override=img_positions,
    N_SEM=N_SEM,
)
answer_token_id = main_attr["answer_token_id"]
answer_token_text = main_attr["answer_token_text"]
answer_patch_attn = main_attr["patch_scores"]

p_importance = np.array([main_attr["prompt_scores"][pos] for pos in p_positions], dtype=float)
p_min, p_max = p_importance.min(), p_importance.max()
if p_max - p_min > 1e-9:
    p_attn_norm = (p_importance - p_min) / (p_max - p_min)
else:
    p_attn_norm = np.ones(len(p_positions)) / max(1, len(p_positions))

print(
    f'  ✓ answer-conditioned attribution token: "{answer_token_text}" '
    f"(id={answer_token_id}, grad_layers={main_attr['grad_layers_available']})"
)
print(f"  ✓ prompt words + answer-conditioned importance: {list(zip(p_words, p_attn_norm.round(3).tolist()))}")

target_candidates = [
    (pos, w) for pos, w in zip(p_positions, p_words)
    if w.lower().strip() == TARGET_WORD.lower().strip()
]

if not target_candidates:
    raise ValueError(
        f'TARGET_WORD="{TARGET_WORD}" not found in prompt tokens: {p_words}'
    )

target_pos, target_word = target_candidates[0][0], target_candidates[0][1].lower().strip()
print(f'  ✓ selected explicit target word for image map: "{target_word}"')

# build target-word-only patch map conservatively:
# average heads within each late layer, then average across late layers
target_maps = []
for l in sem_layers:
    a = attentions[l][0].cpu().numpy()  # (heads, seq, seq)
    row_img_heads = a[:, target_pos, :][:, img_positions]  # (heads, n_img_tokens)

    denom = row_img_heads.sum(axis=1, keepdims=True)
    denom[denom < 1e-12] = 1.0
    row_img_heads = row_img_heads / denom

    target_maps.append(row_img_heads.mean(axis=0))

target_patch_attn = np.mean(target_maps, axis=0)

# layer-wise target-token -> all image tokens
layer_target_to_img = [
    float(mean_attn_layers[l][target_pos, img_positions].sum())
    for l in range(num_layers)
]

target_canvas_raw = project_patch_vector_to_canvas(
    answer_patch_attn,
    image_geom,
    image.size,
    include_global=False,
)
target_canvas_display = normalise_for_display(target_canvas_raw)
target_boxes = lookup_target_boxes(IMAGE_PATH, target_word)
target_loc_metrics = evaluate_localisation(target_canvas_raw, target_boxes)
hot_y, hot_x = np.unravel_index(np.argmax(target_canvas_raw), target_canvas_raw.shape)
print(f"  ✓ selected target word for spatial map: {target_word}")
print(f"  ✓ target raw hotspot: pixel=({hot_x},{hot_y})")
if target_loc_metrics is not None:
    print(
        "  ✓ localisation:"
        f" pointing_hit={target_loc_metrics['pointing_hit']}"
        f"  mass_in_box={target_loc_metrics['mass_in_box']:.3f}"
        f"  peak_xy={target_loc_metrics['peak_xy']}"
    )

# ──────────────────────────────────────────────────────────────────────────────
#  10. Plot — Figure 1: main attention dashboard
# ──────────────────────────────────────────────────────────────────────────────
cap = min(80, seq_len)

fig = plt.figure(figsize=(22, 18))
fig.suptitle(
    f'SmolVLM-500M Attention Analysis\nPrompt: "{PROMPT}"',
    fontsize=13, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── (a) Input image ───────────────────────────────────────────────────────────
ax_img = fig.add_subplot(gs[0, 0])
ax_img.imshow(image)
ax_img.set_title("Input Image", fontweight="bold", fontsize=10)
ax_img.axis("off")

# ── (b) Token-type map ────────────────────────────────────────────────────────
ax_tok = fig.add_subplot(gs[0, 1])
type_row = is_image.astype(float).reshape(1, -1)
im_tok = ax_tok.imshow(type_row, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
ax_tok.set_title("Token Types  (red=image · blue=text)", fontweight="bold", fontsize=9)
ax_tok.set_xlabel("Token position")
ax_tok.set_yticks([])
plt.colorbar(im_tok, ax=ax_tok, fraction=0.08, pad=0.04)
if len(img_positions) > 0:
    ax_tok.axvline(img_positions[0],  color="green",  lw=1.2, linestyle="--", label="img start")
    ax_tok.axvline(img_positions[-1], color="orange", lw=1.2, linestyle="--", label="img end")
ax_tok.legend(fontsize=7)

# ── (c) Attention overlaid on image patches ──────────────────────────────────
ax_ov = fig.add_subplot(gs[0, 2])
ax_ov.imshow(image)
ax_ov.imshow(target_canvas_display, cmap="hot", alpha=0.55, vmin=0, vmax=1)
for x0, y0, x1, y1 in target_boxes:
    ax_ov.add_patch(MplRect((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="cyan", facecolor="none"))
ax_ov.set_title(
    f'Answer-Conditioned Image Attribution\n(first answer token "{answer_token_text}" → image)',
    fontweight="bold", fontsize=9,
)
ax_ov.axis("off")

# ── (d) Image vs text attention fraction stacked bar ─────────────────────────
ax_bar = fig.add_subplot(gs[1, :])
x = np.arange(seq_len)
ax_bar.bar(x, img_frac, color="#e05c5c", alpha=0.85, label="→ image tokens", width=1.0)
ax_bar.bar(x, txt_frac, bottom=img_frac, color="#5c8de0", alpha=0.85, label="→ text tokens", width=1.0)
ax_bar.set_title(
    "Per-Token Attention Fraction — averaged across all layers & heads",
    fontweight="bold", fontsize=10,
)
ax_bar.set_xlabel("Token position")
ax_bar.set_ylabel("Attention fraction")
ax_bar.legend(fontsize=9)
if len(img_positions) > 0:
    ax_bar.axvspan(img_positions[0], img_positions[-1], alpha=0.10, color="red")
    ax_bar.text(
        (img_positions[0] + img_positions[-1]) / 2, 1.02,
        "← image token span →",
        ha="center", va="bottom", fontsize=8, color="red",
        transform=ax_bar.get_xaxis_transform(),
    )

# ── (e) Mean attention heatmap (first `cap` tokens) ──────────────────────────
ax_heat = fig.add_subplot(gs[2, :2])
im_h = ax_heat.imshow(
    mean_attn[:cap, :cap], aspect="auto", cmap="viridis", interpolation="nearest"
)
ax_heat.set_title(
    f"Mean Attention Heatmap — all layers & heads (first {cap} tokens)",
    fontweight="bold", fontsize=10,
)
ax_heat.set_xlabel("Key position")
ax_heat.set_ylabel("Query position")
plt.colorbar(im_h, ax=ax_heat, fraction=0.03, pad=0.02)
for pos in img_positions[img_positions < cap][::max(1, len(img_positions) // 10)]:
    ax_heat.axvline(pos, color="red", alpha=0.20, linewidth=0.7)
    ax_heat.axhline(pos, color="red", alpha=0.20, linewidth=0.7)

# ── (f) Per-layer image attention ────────────────────────────────────────────
ax_layer = fig.add_subplot(gs[2, 2])
ax_layer.plot(layer_target_to_img, marker="o", markersize=4, color="#e05c5c", linewidth=2)
ax_layer.fill_between(range(num_layers), layer_target_to_img, alpha=0.20, color="#e05c5c")
ax_layer.set_title(
    f'Target-word raw attention → image-token mass per layer\n("{target_word}", diagnostic only)',
    fontweight="bold", fontsize=9,
)
ax_layer.set_xlabel("Layer")
ax_layer.set_ylabel("Attention mass")
ax_layer.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
ax_layer.grid(True, alpha=0.3)

out_path = os.path.join(OUT_DIR, "attention_dashboard_500m.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n✓ Saved dashboard → {out_path}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
#  11. Plot — Figure 2: per-head attention for sampled layers
# ──────────────────────────────────────────────────────────────────────────────
sample_layers = sorted({0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1})
cap2 = min(60, seq_len)

fig2, axes2 = plt.subplots(
    len(sample_layers), num_heads,
    figsize=(num_heads * 2.5, len(sample_layers) * 2.8),
)
fig2.suptitle(
    f'SmolVLM-500M  Per-Head Attention (first {cap2} tokens)\nPrompt: "{PROMPT}"',
    fontsize=12, fontweight="bold",
)

for row, layer_idx in enumerate(sample_layers):
    a = attentions[layer_idx][0].cpu().numpy()   # (heads, seq, seq)
    for head in range(num_heads):
        ax = axes2[row, head] if len(sample_layers) > 1 else axes2[head]
        ax.imshow(a[head, :cap2, :cap2], aspect="auto", cmap="hot", interpolation="nearest")
        if row == 0:
            ax.set_title(f"Head {head}", fontsize=8)
        if head == 0:
            ax.set_ylabel(f"L{layer_idx}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        # cyan vertical lines at image-token positions
        for pos in img_positions[img_positions < cap2]:
            ax.axvline(pos, color="cyan", alpha=0.25, linewidth=0.4)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "per_head_attention_500m.png")
fig2.savefig(out2, dpi=120, bbox_inches="tight")
print(f"✓ Saved per-head grid  → {out2}")
plt.close(fig2)

# ──────────────────────────────────────────────────────────────────────────────
#  12. Figure 3 — Prompt word highlight + attention-based image focus
# ──────────────────────────────────────────────────────────────────────────────
# ── Spotlight overlay: high-attention = original colour, low-attention = black ─
from PIL import ImageFilter
img_w_orig, img_h_orig = image.size
# Gaussian blur on the projected canvas for smooth spotlight edges.
spatial_pil = Image.fromarray((target_canvas_display * 255).astype(np.uint8))
spatial_pil = spatial_pil.filter(ImageFilter.GaussianBlur(radius=4.0))
attn_up = np.array(spatial_pil).astype(float) / 255.0  # (H, W) in [0, 1]
img_f   = np.array(image).astype(float) / 255.0   # (H, W, 3)
# power 1.5: high-attention patches stay bright, low-attention → dark/black
# FLOOR keeps low-attention areas as dark-but-visible (like the reference image)
# rather than pure black, so the full image structure remains readable.
# FLOOR=0.18: low-attention regions stay dark but the image structure is still
# readable (pure 0 would be pitch black, indistinguishable from empty space).
# power=1.5: gentler curve so mid-attention regions are not wiped out.
# FLOOR=0.0 → truly black where model ignores; power=0.6 keeps mid-range bright
FLOOR    = 0.0
bright   = np.power(np.clip(attn_up, 0.0, 1.0), 0.6)[:, :, np.newaxis]
dark_img = (img_f * bright * 255).clip(0, 255).astype(np.uint8)

# Use RAW projected canvas for crop center, not display-normalized map.
flat = target_canvas_raw.reshape(-1)
k = max(200, int(0.005 * flat.size))
top_idx = np.argpartition(flat, -k)[-k:]

rows, cols = np.unravel_index(top_idx, target_canvas_raw.shape)
weights = flat[top_idx].astype(float)

if weights.sum() < 1e-12:
    # fallback to argmax if something degenerate happens
    cy, cx = np.unravel_index(np.argmax(target_canvas_raw), target_canvas_raw.shape)
else:
    weights = weights / weights.sum()
    cy = np.sum((rows + 0.5) * weights)
    cx = np.sum((cols + 0.5) * weights)

cx_px = int(cx)
cy_px = int(cy)

cx_px = max(0, min(img_w_orig - 1, cx_px))
cy_px = max(0, min(img_h_orig - 1, cy_px))

print(f"  ✓ crop centre (top-k weighted centroid) at pixel ({cx_px}, {cy_px}) in {img_w_orig}x{img_h_orig} image")

# tighter crop than 60%; otherwise it swallows too much background
cw = int(img_w_orig * 0.35)
ch = int(img_h_orig * 0.35)

px0 = max(0, cx_px - cw // 2)
py0 = max(0, cy_px - ch // 2)
px1 = min(img_w_orig, px0 + cw)
py1 = min(img_h_orig, py0 + ch)

if px1 - px0 < cw:
    px0 = max(0, px1 - cw)
if py1 - py0 < ch:
    py0 = max(0, py1 - ch)

# ── Helper: render inline word chips coloured by attention ────────────────────
def draw_prompt_highlight(ax, words, scores, fontsize=15):
    """
    Render words as coloured boxes on a white axis.
    scores should already be min-max normalised to [0, 1] within the word set.
    0 = white (low attention), 1 = deep red (high attention).
    """
    def score_to_color(w):
        # Pure white → deep red; gives maximum visible contrast
        w = float(w)
        return (0.98, 1.0 - 0.88 * w, 1.0 - 0.88 * w, 1.0)  # RGBA

    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layout parameters (in axes-fraction units)
    x0, y0  = 0.02, 0.82
    x_max   = 0.98
    line_h  = 0.26
    gap     = 0.012        # horizontal gap between chips
    px_per_char = 0.014    # rough width-per-character in axes fraction
    pad_x   = 0.030        # extra horizontal padding inside chip

    x, y = x0, y0
    for word, w in zip(words, scores):
        if not word.strip():
            continue
        chip_w = len(word) * px_per_char + pad_x
        chip_w = max(0.04, min(0.30, chip_w))
        # Wrap to next line if needed
        if x + chip_w > x_max:
            x  = x0
            y -= line_h
        if y < 0.05:
            break
        rgba = score_to_color(w)
        lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        fc   = "black" if lum > 0.42 else "white"
        ax.text(
            x + chip_w / 2, y, word,
            ha="center", va="center",
            fontsize=fontsize, color=fc,
            fontfamily="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.30", facecolor=rgba,
                      edgecolor="none", alpha=1.0),
            transform=ax.transAxes, zorder=3,
        )
        x += chip_w + gap

    # Colour-bar legend at the bottom
    grad_data = np.linspace(0, 1, 256)
    grad_colors = np.array([score_to_color(v) for v in grad_data])[:, :3].reshape(1, 256, 3)
    ax.imshow(grad_colors,
              extent=[0.02, 0.40, -0.14, -0.04],
              aspect="auto", transform=ax.transAxes,
              zorder=2, clip_on=False)
    ax.text(0.02, -0.17, "low attn",  fontsize=8, color="#888",
            transform=ax.transAxes, va="top")
    ax.text(0.40, -0.17, "high attn", fontsize=8, color="#c0392b",
            ha="right", transform=ax.transAxes, va="top")


# ── Figure layout ─────────────────────────────────────────────────────────────
fig3 = plt.figure(figsize=(14, 11), facecolor="white")
fig3.suptitle(
    f'SmolVLM-500M  |  Prompt & Image Attention Focus\nPrompt: "{PROMPT}"',
    fontsize=13, fontweight="bold", y=0.99, color="#222",
)
gs3 = gridspec.GridSpec(
    2, 2, figure=fig3,
    height_ratios=[0.85, 1.5],
    hspace=0.55, wspace=0.18,
)

# ── (a) Prompt words highlighted — spans full top row ─────────────────────────
ax_txt = fig3.add_subplot(gs3[0, :])
ax_txt.set_title(
    f'Prompt token salience for first answer token "{answer_token_text}"',
    fontsize=9, pad=6, loc="left", color="#333",
)
draw_prompt_highlight(ax_txt, p_words, p_attn_norm, fontsize=14)

# ── (b) Darkened image overlay + cyan crop box ────────────────────────────────
ax_ov3 = fig3.add_subplot(gs3[1, 0])
ax_ov3.imshow(dark_img)
for x0, y0, x1, y1 in target_boxes:
    ax_ov3.add_patch(MplRect((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="lime", facecolor="none"))
ax_ov3.add_patch(MplRect(
    (px0, py0), px1 - px0, py1 - py0,
    linewidth=2.5, edgecolor="cyan", facecolor="none",
))
ax_ov3.set_title(
    f'Answer-conditioned image attribution\n(first token "{answer_token_text}", cyan box = top-k centroid crop)',
    fontweight="bold", fontsize=9,
)
ax_ov3.axis("off")

# ── (c) Full image with attention spotlight ───────────────────────────────────
# Same complete image as the left panel, but WITHOUT the crop box overlay,
# so the viewer can clearly see: bright = model focuses here, black = ignored.
ax_crop = fig3.add_subplot(gs3[1, 1])
dark_img_pil = Image.fromarray(dark_img)
crop_dark = dark_img_pil.crop((px0, py0, px1, py1))
ax_crop.imshow(crop_dark)
ax_crop.set_title(
    f'High-attribution crop\n(first answer token = "{answer_token_text}")',
    fontweight="bold", fontsize=9,
)
ax_crop.axis("off")

out3 = os.path.join(OUT_DIR, "prompt_image_attention_500m.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
print(f"✓ Saved prompt+image attn → {out3}")
plt.close(fig3)

# ──────────────────────────────────────────────────────────────────────────────
#  13. Generate model response
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating model response …")
with torch.no_grad():
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
    )
response_text = processor.decode(
    gen_ids[0][seq_len:], skip_special_tokens=True
)

print("─" * 60)
print(f"Prompt : {PROMPT}")
print(f"Answer : {response_text}")
print("─" * 60)

# ──────────────────────────────────────────────────────────────────────────────
#  14. Summary
# ──────────────────────────────────────────────────────────────────────────────
peak_layer = int(np.argmax(layer_target_to_img))
print("\n── Attention Summary ──────────────────────────────────────────")
print(f"  Total tokens         : {seq_len}")
print(f"  Image tokens         : {len(img_positions)}  ({100*len(img_positions)/seq_len:.1f}%)")
print(f"  Text tokens          : {len(txt_positions)}  ({100*len(txt_positions)/seq_len:.1f}%)")
print(f"  Avg img-attn frac    : {img_frac.mean():.4f}  (over all query positions)")
print(f'  Target word          : "{target_word}"')
print(f'  First answer token   : "{answer_token_text}"')
print(f"  Peak target-img layer: {peak_layer}  (value = {layer_target_to_img[peak_layer]:.4f})")
print(
    "  Attn-only hidden cos : "
    f"{self_attn_rollout['final_hidden_cosine']['all']:.4f}"
    f"  (img={self_attn_rollout['final_hidden_cosine']['image']:.4f},"
    f" txt={self_attn_rollout['final_hidden_cosine']['text']:.4f})"
)
print(
    f"  Next token check     : real=\"{self_attn_rollout['real_next_token']}\""
    f'  attn-only="{self_attn_rollout["attn_only_next_token"]}"'
)
print(f"  Self-attn verdict    : {self_attn_rollout['verdict']}")
if target_loc_metrics is not None:
    print(f"  Pointing hit         : {target_loc_metrics['pointing_hit']}")
    print(f"  Attention in box     : {target_loc_metrics['mass_in_box']:.4f}")
print(f"  Outputs saved in     : {OUT_DIR}")

# ──────────────────────────────────────────────────────────────────────────────
#  15. Compare attention across ALL PROMPT_TARGET_PAIRS
#      Re-run a forward pass for every (prompt, target_word) pair, then:
#        • print Pearson r / cosine similarity between spatial attention maps
#        • save a side-by-side overlay figure  → results_500m/prompt_comparison.png
#
#  If Pearson r ≈ 1.0  → model attends to the same image regions regardless of
#                         the prompt, i.e. the prompt has NO effect on attention.
#  If Pearson r << 1.0 → the prompt genuinely shifts where the model looks.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("PROMPT COMPARISON — verifying attention changes with prompt …")


def compute_target_patch_attn(p_str, t_word, image, processor, model, device, N_SEM=8):
    """Forward pass → target-word spatial attention over image patches."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p_str}]}]
    pt   = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inp  = processor(text=pt, images=[image], return_tensors="pt").to(device)
    ids  = inp["input_ids"][0]

    with torch.no_grad():
        out = model(**inp, output_attentions=True)

    atts      = out.attentions
    n_layers  = len(atts)
    sem_lyrs  = list(range(n_layers - min(N_SEM, n_layers), n_layers))
    img_tok_id = model.config.image_token_id
    img_pos_l  = np.where((ids == img_tok_id).cpu().numpy())[0]

    # locate target word inside this tokenization
    full_ids = ids.tolist()
    bare_ids = processor.tokenizer.encode(p_str, add_special_tokens=False)
    start = next(
        (si for si in range(len(full_ids) - len(bare_ids) + 1)
         if full_ids[si : si + len(bare_ids)] == bare_ids),
        None,
    )
    if start is None:
        raise ValueError(f"Prompt not found in token sequence for: {p_str!r}")

    p_pos_l = list(range(start, start + len(bare_ids)))
    p_toks_l = processor.tokenizer.convert_ids_to_tokens([full_ids[i] for i in p_pos_l])
    p_words_l = [t.replace("\u2581", " ").replace("\u0120", " ").strip() or t
                 for t in p_toks_l]

    cands = [(pos, w) for pos, w in zip(p_pos_l, p_words_l)
             if w.lower().strip() == t_word.lower().strip()]
    if not cands:
        raise ValueError(f'Target word "{t_word}" not found in tokens {p_words_l}')
    tgt_pos = cands[0][0]

    maps = []
    for l in sem_lyrs:
        a   = atts[l][0].cpu().numpy()                       # (heads, seq, seq)
        row = a[:, tgt_pos, :][:, img_pos_l]                 # (heads, n_img)
        row = row / np.maximum(row.sum(1, keepdims=True), 1e-12)
        maps.append(row.mean(0))

    return np.mean(maps, axis=0)   # (n_img_tokens,)


# ── Run every pair ─────────────────────────────────────────────────────────────
cmp_patch_attns = {}
cmp_responses   = {}

for p_str, t_word in PROMPT_TARGET_PAIRS:
    print(f'\n  Prompt: "{p_str}"  target: "{t_word}"')

    patch = compute_target_patch_attn(p_str, t_word, image, processor, model, device)
    cmp_patch_attns[(p_str, t_word)] = patch

    msgs_r = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p_str}]}]
    pt_r   = processor.apply_chat_template(msgs_r, add_generation_prompt=True)
    inp_r  = processor(text=pt_r, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inp_r, max_new_tokens=60, do_sample=False)
    resp = processor.decode(gen[0][inp_r["input_ids"].shape[1]:], skip_special_tokens=True)
    cmp_responses[(p_str, t_word)] = resp
    print(f'    → Answer: "{resp}"')

# ── Numerical similarity between every pair of maps ───────────────────────────
pairs_list = list(cmp_patch_attns.items())
print("\n── Spatial Attention Map Similarity ──────────────────────────────")
for i in range(len(pairs_list)):
    for j in range(i + 1, len(pairs_list)):
        (p1, t1), a1 = pairs_list[i]
        (p2, t2), a2 = pairs_list[j]
        corr    = float(np.corrcoef(a1, a2)[0, 1])
        cos_sim = float(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-12))
        l1_diff = float(np.abs(a1 - a2).mean())

        base_scale = float(0.5 * (np.mean(np.abs(a1)) + np.mean(np.abs(a2))) + 1e-12)
        rel_diff = l1_diff / base_scale
        print(f'  "{p1}" [{t1}]')
        print(f'  "{p2}" [{t2}]')
        print(f"    Pearson r  = {corr:.4f}  (1.0 = identical map, 0 = unrelated)")
        print(f"    Cosine sim = {cos_sim:.4f}")
        print(f"    Mean |Δ|   = {l1_diff:.6f}")
        print(f"    Rel |Δ|    = {rel_diff:.4f}  (difference relative to map magnitude)")
        if corr >= 0.95 or cos_sim >= 0.95:
            if rel_diff < 0.10:
                print("    ⚠  Maps are highly similar — prompt change has only a very weak effect on this attention map.")
            else:
                print("    △  Maps remain highly correlated, but relative differences are not negligible.")
        elif corr >= 0.80 or cos_sim >= 0.80:
            print("    △  Maps changed somewhat, but still share strong overall structure.")
        else:
            print("    ✓  Maps differ clearly — attention responds substantially to prompt content.")

# ── Side-by-side visual comparison ────────────────────────────────────────────
n_cols = len(PROMPT_TARGET_PAIRS)
fig_cmp, axes_cmp = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10), facecolor="white")
if n_cols == 1:
    axes_cmp = np.array(axes_cmp).reshape(2, 1)

fig_cmp.suptitle(
    "Prompt Comparison — Raw Target-Word Attention (diagnostic)\n"
    "Top: attention heatmap  |  Bottom: overlay on image",
    fontsize=12, fontweight="bold",
)

for col, (p_str, t_word) in enumerate(PROMPT_TARGET_PAIRS):
    patch  = cmp_patch_attns[(p_str, t_word)]
    g_raw = project_patch_vector_to_canvas(patch, image_geom, image.size, include_global=False)
    g_disp = normalise_for_display(g_raw)

    axes_cmp[0, col].imshow(g_disp, cmap="hot", vmin=0, vmax=1)
    axes_cmp[0, col].set_title(f'"{p_str}"\ntarget="{t_word}"', fontsize=9, fontweight="bold")
    axes_cmp[0, col].axis("off")

    axes_cmp[1, col].imshow(image)
    axes_cmp[1, col].imshow(g_disp, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    axes_cmp[1, col].set_title(f'Answer: "{cmp_responses[(p_str, t_word)][:60]}"', fontsize=8)
    axes_cmp[1, col].axis("off")

plt.tight_layout()
out_cmp = os.path.join(OUT_DIR, "prompt_comparison.png")
fig_cmp.savefig(out_cmp, dpi=130, bbox_inches="tight", facecolor="white")
print(f"\n✓ Saved comparison figure → {out_cmp}")
plt.close(fig_cmp)

raw_pairs_list = list(cmp_patch_attns.items())
for i in range(len(raw_pairs_list)):
    for j in range(i + 1, len(raw_pairs_list)):
        (p1, t1), a1 = raw_pairs_list[i]
        (p2, t2), a2 = raw_pairs_list[j]
        diff = (
            project_patch_vector_to_canvas(a2, image_geom, image.size, include_global=False)
            - project_patch_vector_to_canvas(a1, image_geom, image.size, include_global=False)
        )
        fig_diff, ax_diff = plt.subplots(1, 1, figsize=(6, 5), facecolor="white")
        vmax = np.max(np.abs(diff)) + 1e-12
        im = ax_diff.imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax_diff.set_title(f'Spatial attention difference\n"{t2}" - "{t1}"', fontweight="bold")
        ax_diff.set_xlabel("Image x")
        ax_diff.set_ylabel("Image y")
        plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
        out_diff = os.path.join(OUT_DIR, f"prompt_diff_{t2}_minus_{t1}.png")
        fig_diff.savefig(out_diff, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig_diff)
        print(f"✓ Saved difference map → {out_diff}")

# ──────────────────────────────────────────────────────────────────────────────
#  16. Gradient × Attention comparison  (much more sensitive to prompt changes)
#
#  WHY raw attention is a poor discriminator here:
#    In a causal VLM, image tokens are a prefix → their key vectors are computed
#    with NO text context (causal masking).  Both prompts share the SAME image
#    keys, so "green" and "red" queries attend to nearly the same distribution
#    → Pearson r ≈ 0.97 by construction, not because the model ignores the prompt.
#
#  Gradient × Attention (GradCAM-style) fixes this:
#    ∂(logit[target_word]) / ∂(attention_weight[layer,head,img_pos])
#    × attention_weight[layer,head,img_pos]
#    This asks: "which image patches, if attended to more, would most raise the
#    probability of THIS target word?"  The gradient flows back through the
#    entire residual stream, so it IS conditioned on the specific output word.
#
#  Expected: maps for "green" and "red" should differ noticeably, especially
#  near the two blocks in the lower-left of the scene.
# ──────────────────────────────────────────────────────────────────────────────
# ── Free main forward-pass tensors before the gradient backward pass ──────────
# The `attentions` tuple from section 4 holds 30 × (1,8,875,875) float32
# tensors on GPU ≈ 735 MB.  The lm_head in the grad forward needs ~166 MB for
# its logits output, which fits only after these are released.
del outputs, attentions, mean_attn_layers, mean_attn, last_attn
torch.cuda.empty_cache()

print("\n" + "─" * 60)
print("GRADIENT × ATTENTION comparison (prompt-conditioned) …")


def compute_grad_attn_map(p_str, image, processor, model, device, N_SEM=8):
    """
    Returns grad×attn spatial map over image patches for the given prompt.

    Steps:
      1. Freeze the vision encoder + connector so PyTorch does NOT store their
         intermediate activations for backward (fixes the OOM).  The LM decoder
         text-token embeddings still require grad, so the LM attention weights
         remain in the computation graph.
      2. Forward pass with output_attentions=True.
      3. Call retain_grad() on the late-layer LM attention tensors BEFORE
         backward (non-leaf tensors drop their grad otherwise).
      4. Score = logit of the target word at the last token position.
      5. Backprop: dScore / d(attn_weight[l,h,last_q,img_key]).
      6. grad×attn = |grad| × attn, averaged over heads and late layers.
      7. Restore vision-encoder requires_grad and clear CUDA cache.
    """
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p_str}]}]
    pt   = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inp  = processor(text=pt, images=[image], return_tensors="pt").to(device)
    ids  = inp["input_ids"][0]

    img_tok_id = model.config.image_token_id
    img_pos_l  = np.where((ids == img_tok_id).cpu().numpy())[0]

    # ── Bug fix 1: freeze vision encoder + connector ──────────────────────────
    # With enable_grad(), PyTorch stores every intermediate tensor in the
    # vision encoder for backward.  The vision encoder processes a
    # high-resolution image through many attention layers → huge memory spike.
    # Since we only need d(logit)/d(LM_attn_weight), we freeze everything
    # upstream of the LM decoder so the graph is never built through it.
    frozen_params = []
    for name, param in model.named_parameters():
        if ("vision_model" in name or "connector" in name) and param.requires_grad:
            param.requires_grad_(False)
            frozen_params.append(name)

    torch.cuda.empty_cache()
    model.zero_grad()

    try:
        with torch.enable_grad():
            out = model(**inp, output_attentions=True)

        atts       = out.attentions                          # tuple[n_layers] of (1,H,S,S)
        n_layers_l = len(atts)
        sem_lyrs_l = list(range(n_layers_l - min(N_SEM, n_layers_l), n_layers_l))

        # ── Bug fix 2: retain_grad() BEFORE backward ──────────────────────────
        # out.attentions[l] is an intermediate (non-leaf) tensor.
        # PyTorch silently drops gradients for non-leaf tensors unless
        # retain_grad() is called explicitly before backward().
        for l in sem_lyrs_l:
            if atts[l] is not None and atts[l].requires_grad:
                atts[l].retain_grad()

        with torch.enable_grad():
            last_logits = out.logits[0, -1, :]   # (vocab,)
            answer_token_id_l, answer_token_text_l = pick_attribution_token(last_logits, processor.tokenizer)
            score = last_logits[answer_token_id_l]
            score.backward()

    finally:
        # Always restore requires_grad, even if an exception occurs
        for name, param in model.named_parameters():
            if name in frozen_params:
                param.requires_grad_(True)
        model.zero_grad()
        torch.cuda.empty_cache()

    maps = []
    n_with_grad = 0
    for l in sem_lyrs_l:
        a    = atts[l][0]                         # (heads, seq, seq)
        grad = atts[l].grad                       # (1, heads, seq, seq) or None

        a_np = a.detach().cpu().numpy()
        if grad is not None:
            g_np = grad[0].detach().cpu().numpy() # (heads, seq, seq)
            n_with_grad += 1
        else:
            # Fallback: treat all gradient weights as 1 (degrades to plain attn)
            g_np = np.ones_like(a_np)

        # grad×attn at query=last token, key=image-token positions
        last_q = a_np.shape[2] - 1
        ga = np.abs(g_np[:, last_q, :][:, img_pos_l]) * a_np[:, last_q, :][:, img_pos_l]
        ga = ga / np.maximum(ga.sum(1, keepdims=True), 1e-12)
        maps.append(ga.mean(0))                   # average over heads

    print(f"    grad available on {n_with_grad}/{len(sem_lyrs_l)} late layers"
          + ("" if n_with_grad > 0 else "  ← fell back to plain attention!"))
    return np.mean(maps, axis=0), answer_token_text_l  # (n_img_tokens,), label


grad_attn_maps = {}
grad_attn_answer_tokens = {}
for p_str, t_word in PROMPT_TARGET_PAIRS:
    print(f'  Computing grad×attn for "{p_str}" / target="{t_word}" …')
    gmap, answer_tok = compute_grad_attn_map(p_str, image, processor, model, device)
    grad_attn_maps[(p_str, t_word)] = gmap
    grad_attn_answer_tokens[(p_str, t_word)] = answer_tok
    print(f'    using first answer token "{answer_tok}"')
    print(f"    range=[{gmap.min():.6f}, {gmap.max():.6f}]  std={gmap.std():.6f}")

# ── Numerical comparison ──────────────────────────────────────────────────────
ga_pairs = list(grad_attn_maps.items())
print("\n── Grad×Attn Map Similarity (compare to raw-attn above) ─────────")
for i in range(len(ga_pairs)):
    for j in range(i + 1, len(ga_pairs)):
        (p1, t1), g1 = ga_pairs[i]
        (p2, t2), g2 = ga_pairs[j]
        corr    = float(np.corrcoef(g1, g2)[0, 1])
        cos_sim = float(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-12))
        l1_diff = float(np.abs(g1 - g2).mean())
        base    = float(0.5 * (np.abs(g1).mean() + np.abs(g2).mean()) + 1e-12)
        rel     = l1_diff / base
        # Determine if answer flipped (Yes→No or No→Yes) between the two prompts
        r1 = cmp_responses.get((p1, t1), "")
        r2 = cmp_responses.get((p2, t2), "")
        ans1 = "yes" if "yes" in r1.lower() else ("no" if "no" in r1.lower() else "?")
        ans2 = "yes" if "yes" in r2.lower() else ("no" if "no" in r2.lower() else "?")
        ans_flip = (ans1 != ans2 and "?" not in (ans1, ans2))

        print(f'  "{p1}" [{t1}] → "{ans1}"  vs  "{p2}" [{t2}] → "{ans2}"')
        print(f"    Pearson r  = {corr:.4f}")
        print(f"    Cosine sim = {cos_sim:.4f}")
        print(f"    Rel |Δ|    = {rel:.4f}")
        # Grad×attn is structurally more correlated than raw attention because it
        # still multiplies the shared image-key softmax weights.  An appropriate
        # "clearly different" threshold for grad×attn is ~0.85, not 0.90.
        # For same-answer pairs (both Yes or both No) higher correlation is expected.
        threshold = 0.85
        if corr < threshold:
            if ans_flip:
                print(f"    ✓  Maps differ (r<{threshold}) AND answer flipped ({ans1}→{ans2})"
                      f" — strong evidence prompt changes model behaviour.")
            else:
                print(f"    ✓  Maps differ (r<{threshold}) even with same answer"
                      f" — prompt shifts spatial focus.")
        elif rel > 0.40:
            print(f"    △  r={corr:.3f} but Rel|Δ|={rel:.3f} — moderate spatial shift"
                  + (f" (answer flipped: {ans1}→{ans2})" if ans_flip else "") + ".")
        else:
            if ans_flip:
                print(f"    △  Answer flipped ({ans1}→{ans2}) but spatial maps similar"
                      f" — discrimination is in FFN/residual stream, not attention weights.")
            else:
                print(f"    ⚠  Same answer, similar maps — only a single color word differs,"
                      f" weak contrast for this metric.")

# ── Side-by-side grad×attn figure ────────────────────────────────────────────
n_cols_g = len(PROMPT_TARGET_PAIRS)
fig_g, axes_g = plt.subplots(2, n_cols_g, figsize=(6 * n_cols_g, 10), facecolor="white")
if n_cols_g == 1:
    axes_g = np.array(axes_g).reshape(2, 1)

fig_g.suptitle(
    "Prompt Comparison — Answer-Conditioned Gradient × Attention\n"
    "Top: grad×attn heatmap  |  Bottom: overlay on image",
    fontsize=12, fontweight="bold",
)

for col, (p_str, t_word) in enumerate(PROMPT_TARGET_PAIRS):
    gmap   = grad_attn_maps[(p_str, t_word)]
    g_raw = project_patch_vector_to_canvas(gmap, image_geom, image.size, include_global=False)
    g_disp = normalise_for_display(g_raw)

    axes_g[0, col].imshow(g_disp, cmap="hot", vmin=0, vmax=1)
    axes_g[0, col].set_title(
        f'"{p_str}"\nanswer_tok="{grad_attn_answer_tokens[(p_str, t_word)]}"\n(grad×attn)',
        fontsize=9,
        fontweight="bold",
    )
    axes_g[0, col].axis("off")

    axes_g[1, col].imshow(image)
    axes_g[1, col].imshow(g_disp, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    axes_g[1, col].set_title(
        f'Answer: "{cmp_responses[(p_str, t_word)][:60]}"', fontsize=8
    )
    axes_g[1, col].axis("off")

plt.tight_layout()
out_g = os.path.join(OUT_DIR, "grad_attn_comparison.png")
fig_g.savefig(out_g, dpi=130, bbox_inches="tight", facecolor="white")
print(f"\n✓ Saved grad×attn comparison → {out_g}")
plt.close(fig_g)

# ── Difference map: grad×attn — all pairs ────────────────────────────────────
ga_pairs_list = list(grad_attn_maps.items())
for i in range(len(ga_pairs_list)):
    for j in range(i + 1, len(ga_pairs_list)):
        (p1, t1), g1 = ga_pairs_list[i]
        (p2, t2), g2 = ga_pairs_list[j]
        gdiff = (
            project_patch_vector_to_canvas(g2, image_geom, image.size, include_global=False)
            - project_patch_vector_to_canvas(g1, image_geom, image.size, include_global=False)
        )
        fig_gd, ax_gd = plt.subplots(1, 1, figsize=(6, 5), facecolor="white")
        vmax_g = np.max(np.abs(gdiff)) + 1e-12
        im_gd = ax_gd.imshow(gdiff, cmap="bwr", vmin=-vmax_g, vmax=vmax_g, interpolation="nearest")
        ax_gd.set_title(
            f'Grad×Attn difference\n"{t2}" − "{t1}"\n'
            f"(red = more important for '{t2}', blue = more important for '{t1}')",
            fontweight="bold", fontsize=9,
        )
        ax_gd.set_xlabel("Image x")
        ax_gd.set_ylabel("Image y")
        plt.colorbar(im_gd, ax=ax_gd, fraction=0.046, pad=0.04)
        out_gd = os.path.join(OUT_DIR, f"grad_attn_diff_{t2}_minus_{t1}.png")
        fig_gd.savefig(out_gd, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig_gd)
        print(f"✓ Saved grad×attn difference map → {out_gd}")
