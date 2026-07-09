"""
Figure 2(a)-style analysis: "Meaningful patterns in top value vector tokens"
Reproduces the bar chart from the π0-FAST paper (arxiv 2509.00328) applied to
SmolVLA with frozen VLM.

For each FFN layer, compute the fraction of value vectors whose top-k promoted
tokens form:
  - interpretable patterns: any coherent cluster (semantic, syntactic, or subword)
  - semantically meaningful patterns: specifically height/spatial concepts

NOTE on scale vs the paper:
  PaliGemma (3B, d_ff≈16384) → ~50-80% interpretable per layer.
  SmolVLM-500M (d_ff=2560, 6x fewer neurons) → neurons are more polysemantic
  by necessity, so interpretable fractions are genuinely lower (~10-30% with
  broad detection). The two bars being identical confirms the frozen VLM claim.

Since the VLM is frozen during VLA fine-tuning, both the base SmolVLM and the
fine-tuned VLA should produce identical bars — this script demonstrates that.

Usage:
    python figure2a_meaningful_value_vectors.py
"""

import re
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ── Semantic keyword categories ───────────────────────────────────────────────

# Height / spatial (used for BOTH interpretable and meaningful)
HEIGHT_SPATIAL = [
    "high", "low", "above", "below", "up", "down", "top", "bottom",
    "upper", "lower", "height", "elevation", "over", "under",
    "higher", "raise", "tall", "altitude", "vertical", "ceiling",
    "floor", "summit", "peak", "lift", "ascend", "descend",
]

# Other broad semantic domains for interpretable
COLOR_KW       = ["red", "green", "blue", "yellow", "white", "black",
                  "orange", "purple", "pink", "brown", "cyan", "gray",
                  "grey", "violet", "gold", "silver", "amber", "indigo"]
NUMBER_KW      = ["one", "two", "three", "four", "five", "six", "seven",
                  "eight", "nine", "ten", "first", "second", "third",
                  "zero", "dozen", "hundred", "thousand", "million"]
ACTION_KW      = ["move", "pick", "place", "grab", "lift", "push", "pull",
                  "reach", "hold", "drop", "grasp", "carry", "take", "put",
                  "slide", "rotate", "press", "turn", "open", "close",
                  "start", "stop", "begin", "end", "run", "walk", "jump"]
DIRECTION_KW   = ["left", "right", "forward", "backward", "north", "south",
                  "east", "west", "front", "back", "side", "lateral",
                  "horizontal", "toward", "away"]
OBJECT_KW      = ["cube", "block", "ball", "box", "table", "robot", "arm",
                  "hand", "gripper", "object", "item", "tool", "container",
                  "plate", "bowl", "cup", "bottle", "chair", "door", "wall"]
TEMPORAL_KW    = ["before", "after", "while", "during", "until", "when",
                  "then", "next", "last", "final", "first", "second", "third",
                  "always", "never", "once", "twice", "again"]
NEGATION_KW    = ["not", "never", "no", "non", "un", "dis", "anti", "without",
                  "neither", "nor", "hardly", "barely", "unless"]
MEDICAL_KW     = ["tion", "ical", "ology", "itis", "emia", "pathy", "osis",
                  "oma", "ectomy", "plasty", "scopy"]
CODE_PUNCT_KW  = ["__()", "def ", "return", "import", "class ", "for ",
                  "while", "None", "True", "False", "print", "\\n", "\\t"]
SIZE_KW        = ["large", "small", "big", "tiny", "huge", "wide", "narrow",
                  "long", "thick", "thin", "deep", "shallow", "heavy", "light"]
GEOGRAPHY_KW   = ["land", "shire", "ville", "berg", "burg", "ton", "ington",
                  "ford", "burg", "field", "haven", "port", "mouth"]

INTERPRETABLE_CATEGORIES = [
    HEIGHT_SPATIAL, COLOR_KW, NUMBER_KW, ACTION_KW, DIRECTION_KW,
    OBJECT_KW, TEMPORAL_KW, NEGATION_KW, SIZE_KW, GEOGRAPHY_KW,
    MEDICAL_KW, CODE_PUNCT_KW,
]

MEANINGFUL_HEIGHT_CATEGORIES = [HEIGHT_SPATIAL]


# ── Structural / subword pattern detectors ────────────────────────────────────

def _is_single_char(tokens):
    """Tokens are mostly single characters (single-letter pattern)."""
    singles = sum(1 for t in tokens if len(t.strip()) <= 2)
    return singles >= 4


def _is_short_function_word(tokens):
    """Tokens are short function words / prefixes ≤4 chars."""
    short = sum(1 for t in tokens if 1 <= len(t.strip()) <= 4)
    return short >= 5


def _is_suffix_cluster(tokens):
    """Tokens share a common 3-4 char suffix (e.g., -ing, -tion, -ness)."""
    common_suffixes = [
        "ing", "tion", "ness", "ment", "ance", "ence", "ity", "ism",
        "ist", "ous", "ful", "less", "able", "ible", "ary", "ery",
        "ory", "ive", "ize", "ise", "ify", "ology", "ography",
    ]
    for sfx in common_suffixes:
        count = sum(1 for t in tokens if t.strip().lower().endswith(sfx))
        if count >= 3:
            return True
    return False


def _is_prefix_cluster(tokens):
    """Tokens share a common 2-4 char prefix."""
    common_prefixes = [
        "re", "pre", "un", "dis", "over", "under", "inter", "intra",
        "trans", "micro", "macro", "hyper", "hypo", "anti", "auto",
        "co", "de", "out", "sub", "super", "multi", "semi",
    ]
    for pfx in common_prefixes:
        count = sum(1 for t in tokens if t.strip().lower().startswith(pfx) and len(t.strip()) > len(pfx))
        if count >= 3:
            return True
    return False


def _is_punctuation_cluster(tokens):
    """Tokens are mostly punctuation or code symbols."""
    symbols = sum(1 for t in tokens if re.fullmatch(r'[\W_]+', t.strip()))
    return symbols >= 4


def _has_structural_pattern(tokens):
    return (
        _is_single_char(tokens)
        or _is_short_function_word(tokens)
        or _is_suffix_cluster(tokens)
        or _is_prefix_cluster(tokens)
        or _is_punctuation_cluster(tokens)
    )


# ── Classification ────────────────────────────────────────────────────────────

def tokens_match_category(tokens, category, min_matches=2):
    tokens_lower = [t.lower() for t in tokens]
    count = sum(
        1 for t in tokens_lower
        if any(kw in t for kw in category)
    )
    return count >= min_matches


def classify_value_vector(top_tokens):
    """
    Returns (is_interpretable, is_meaningful).
    is_meaningful ⊆ is_interpretable.
    """
    is_meaningful = any(
        tokens_match_category(top_tokens, cat)
        for cat in MEANINGFUL_HEIGHT_CATEGORIES
    )
    keyword_interp = any(
        tokens_match_category(top_tokens, cat)
        for cat in INTERPRETABLE_CATEGORIES
    )
    structural_interp = _has_structural_pattern(top_tokens)
    is_interpretable = is_meaningful or keyword_interp or structural_interp
    return is_interpretable, is_meaningful


@torch.no_grad()
def compute_per_layer_fractions_from_vlm(text_model, lm_head_weight, tokenizer,
                                          top_k_tokens=10, device="cpu"):
    """
    Core computation: for each FFN layer, classify each value vector's top tokens.
    Returns dict with per-layer lists of interpretable/meaningful fractions.
    """
    W_out = lm_head_weight.to(device=device, dtype=torch.float32)
    num_layers = len(text_model.layers)

    per_layer_interpretable = []
    per_layer_meaningful = []

    for layer_idx in range(num_layers):
        W_value = text_model.layers[layer_idx].mlp.down_proj.weight.detach().to(
            device=device, dtype=torch.float32
        )  # [d_model, d_ff]

        token_logits = W_out @ W_value          # [vocab_size, d_ff]
        _, top_ids = torch.topk(token_logits, k=top_k_tokens, dim=0)
        top_ids_t = top_ids.transpose(0, 1).cpu().numpy()  # [d_ff, k]

        n_vectors = W_value.shape[1]
        n_interp = 0
        n_mean = 0

        for neuron_idx in range(n_vectors):
            decoded = [
                tokenizer.decode([int(tid)]).replace("\n", " ").strip()
                for tid in top_ids_t[neuron_idx]
            ]
            interp, mean = classify_value_vector(decoded)
            if interp:
                n_interp += 1
            if mean:
                n_mean += 1

        per_layer_interpretable.append(n_interp / n_vectors)
        per_layer_meaningful.append(n_mean / n_vectors)
        print(f"  Layer {layer_idx:2d}: interpretable={n_interp/n_vectors:.3f}  "
              f"meaningful={n_mean/n_vectors:.3f}")

    return {
        "interpretable": per_layer_interpretable,
        "meaningful": per_layer_meaningful,
        "num_layers": num_layers,
    }


@torch.no_grad()
def compute_from_smolvla(policy_path, top_k_tokens=10, device=None):
    """Load SmolVLA and compute per-layer fractions from its frozen VLM."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading SmolVLA: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.eval()
    policy.to(device)

    vlm_model = policy.model.vlm_with_expert.vlm
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    W_out = vlm_model.lm_head.weight.detach()

    result = compute_per_layer_fractions_from_vlm(
        text_model, W_out, tokenizer, top_k_tokens=top_k_tokens, device=device
    )

    del policy
    torch.cuda.empty_cache()
    return result


def plot_figure2a(base_results, vla_results,
                  base_label="SmolVLA Base (pre-task)",
                  vla_label="Height-Steering VLA",
                  save_path="figure2a_height_meaningful_value_vectors.png"):
    """
    Plot grouped bar chart matching Figure 2(a) from the π0-FAST paper.
    Light bars = interpretable fraction; dark overlaid bars = meaningful fraction.
    """
    num_layers = base_results["num_layers"]
    x = np.arange(num_layers)
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(max(12, num_layers * 0.75), 5.5))

    # Interpretable (light) bars
    ax.bar(x - bar_w / 2, base_results["interpretable"], bar_w,
           color="#90EE90", alpha=0.85, label=f"{base_label} (interpretable)")
    ax.bar(x + bar_w / 2, vla_results["interpretable"], bar_w,
           color="#48CAC8", alpha=0.85, label=f"{vla_label} (interpretable)")

    # Semantically meaningful (dark, overlaid) bars
    ax.bar(x - bar_w / 2, base_results["meaningful"], bar_w,
           color="#228B22", label=f"{base_label} (height-meaningful)")
    ax.bar(x + bar_w / 2, vla_results["meaningful"], bar_w,
           color="#005F5F", label=f"{vla_label} (height-meaningful)")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Fraction meaningful value vectors", fontsize=12)
    ax.set_title(
        "Meaningful patterns in top value vector tokens\n"
        "(frozen VLM → both models share identical FFN weights → bars are equal)",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_layers)], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate expected difference vs paper
    note = (
        "Note: SmolVLM-500M (d_ff=2560) has 6× fewer FFN neurons than PaliGemma (d_ff≈16384).\n"
        "Smaller models have more polysemantic neurons → lower interpretable fractions are expected."
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="#555555",
             style="italic", wrap=True)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150)
    print(f"\n[*] Figure saved → {save_path}")
    plt.close()


if __name__ == "__main__":
    VLA_PATH = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2"

    # The VLA is built on SmolVLM2-500M. Since the VLM is frozen, loading just
    # the VLA model is sufficient — but we also load the base to prove they match.
    BASE_PATH = "lerobot/smolvla_base"      # base SmolVLA before task fine-tuning

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n[1/2] Computing fractions for fine-tuned VLA...")
    vla_results = compute_from_smolvla(VLA_PATH, top_k_tokens=10, device=device)

    print("\n[2/2] Computing fractions for base SmolVLA (frozen VLM reference)...")
    base_results = compute_from_smolvla(BASE_PATH, top_k_tokens=10, device=device)

    print("\n[*] Plotting...")
    plot_figure2a(
        base_results=base_results,
        vla_results=vla_results,
        base_label="SmolVLA Base (pre-task)",
        vla_label="Height-Steering VLA",
        save_path="figure2a_height_meaningful_value_vectors.png",
    )

    # Summary: check if bars are identical (they should be if VLM is frozen)
    base_i = np.array(base_results["interpretable"])
    vla_i = np.array(vla_results["interpretable"])
    base_m = np.array(base_results["meaningful"])
    vla_m = np.array(vla_results["meaningful"])

    print("\n── Verification: are the two models' FFN stats identical? ──")
    print(f"  Max |Δ interpretable| = {np.max(np.abs(base_i - vla_i)):.6f}")
    print(f"  Max |Δ meaningful|    = {np.max(np.abs(base_m - vla_m)):.6f}")
    if np.allclose(base_i, vla_i) and np.allclose(base_m, vla_m):
        print("  ✓ Bars are IDENTICAL — confirms VLM was frozen during fine-tuning.")
    else:
        print("  ✗ Bars DIFFER — VLM was NOT fully frozen (some FFN weights changed).")
