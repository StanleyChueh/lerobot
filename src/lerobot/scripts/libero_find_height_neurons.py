#!/usr/bin/env python3
"""
Find height-concept keyword neurons in SmolVLA (paper Appendix C.3 method).

Uses `lerobot_reord_top_token.py`'s semantic embedding extraction +
keyword-frequency ranking to find lm_expert FFN neurons whose top-k
predicted tokens contain height-related words ("high", "low", etc.).

Output: outputs/libero_height_neurons.json
  {
    "high": {"layer_idx": [neuron_idx, ...]},   -- top neurons for HIGH concept
    "low":  {"layer_idx": [neuron_idx, ...]}    -- top neurons for LOW concept
  }

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_find_height_neurons.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero \\
    --top-n 20 \\
    --output outputs/libero_height_neurons.json

Then use --neurons-json outputs/libero_height_neurons.json in libero_eval_steering.py.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

HEIGHT_KEYWORDS = {
    "high": {
        "pos": ["high", "above", "upper", "raise", "raised", "lift", "elevated", "aloft"],
        "neg": ["thigh", "highway", "highlight"],
    },
    "low": {
        "pos": ["low", "below", "lower", "lower", "down", "drop", "beneath", "floor"],
        "neg": ["follow", "allow", "slow", "blow", "glow", "yellow", "hollow",
                "flow", "below", "elbow", "download"],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", default="ethanCSL/svla_franka_pick_n_place_vla_steering_libero")
    ap.add_argument("--top-n",       type=int, default=20, help="Top neurons per concept")
    ap.add_argument("--top-k-tokens",type=int, default=10, help="Top-k tokens per neuron (more = richer)")
    ap.add_argument("--output",      default="outputs/libero_height_neurons.json")
    ap.add_argument("--keywords-json", default=None,
                    help="Optional override JSON {concept: {pos:[...], neg:[...]}}")
    args = ap.parse_args()

    # Lazy import to avoid polluting global namespace
    from lerobot.scripts.lerobot_reord_top_token import (
        load_model_bundle,
        extract_semantic_embeddings,
        run_keyword_based_ranking,
        print_keyword_ranking_summary,
    )

    # ── Load model and extract semantic embeddings ─────────────────────────────
    print(f"Loading model: {args.policy_path}")
    policy, text_model, lm_head_weight, tokenizer, device = load_model_bundle(
        policy_family="smolvla",
        policy_path=args.policy_path,
    )
    bundle = extract_semantic_embeddings(
        text_model, lm_head_weight, tokenizer,
        top_k_tokens=args.top_k_tokens, device=device,
    )
    del policy, text_model  # free VRAM
    import gc; gc.collect()

    # ── Keyword ranking ────────────────────────────────────────────────────────
    if args.keywords_json:
        with open(args.keywords_json) as f:
            keywords = json.load(f)
    else:
        keywords = HEIGHT_KEYWORDS

    results = run_keyword_based_ranking(
        metadata=bundle["metadata"],
        tokenizer=tokenizer,
        concept_keywords_map=keywords,
        top_n=args.top_n,
    )

    print_keyword_ranking_summary(results)

    # ── Build neurons.json in the format expected by libero_eval_steering.py ──
    neurons_out: dict[str, dict[str, list[int]]] = {}
    for concept, ranked in results.items():
        layer_map: dict[str, list[int]] = {}
        for entry in ranked:
            if entry["match_count"] == 0:
                continue
            layer_str = str(entry["layer"])
            layer_map.setdefault(layer_str, []).append(entry["neuron"])
        if layer_map:
            neurons_out[concept] = layer_map

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(neurons_out, f, indent=2)
    print(f"\n[✓] Saved → {out}")
    print(f"    Concepts: {list(neurons_out.keys())}")
    for concept, lm in neurons_out.items():
        total = sum(len(v) for v in lm.values())
        print(f"    {concept}: {len(lm)} layers, {total} neurons")
    print(f"\nUse with:")
    print(f"  --neurons-json {out} --conditions keyword_high keyword_low")


if __name__ == "__main__":
    main()
