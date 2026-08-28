#!/usr/bin/env python3
"""
Merge multiple COAST .pt artifacts into one via conceptor OR algebra.

  A ∨ B = ¬(¬A ∧ ¬B)   (Jaeger Boolean conceptor OR)

Use case: collect grasp-phase conceptors for tasks 0, 1, 2, 3 (each .pt captures
the grasp primitive for one object / scene arrangement), then merge them into a
single cross-object grasp primitive conceptor for zero-shot transfer to task 4.

Usage:
  python src/lerobot/scripts/libero_merge_conceptors.py \\
    --inputs outputs/grasp_primitive/task0.pt \\
             outputs/grasp_primitive/task1.pt \\
             outputs/grasp_primitive/task2.pt \\
             outputs/grasp_primitive/task3.pt \\
    --output outputs/grasp_primitive/merged_or.pt

Then evaluate on task 4 (zero-shot):
  python src/lerobot/scripts/libero_osc_eval.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_natural \\
    --task-idx 4 --task "..." \\
    --conditions none coast_success \\
    --coast-path outputs/grasp_primitive/merged_or.pt \\
    --coast-beta 0.35 --coast-layer-lo 10 --coast-layer-hi 11 \\
    --n-rollouts 50 --max-steps 400 --n-action-steps 10
"""

import argparse
from pathlib import Path

import numpy as np
import torch


# ── Conceptor algebra (mirrors libero_compute_coast.py) ───────────────────────

def _conceptor_not(C: np.ndarray) -> np.ndarray:
    return np.eye(C.shape[0], dtype=np.float64) - C


def _conceptor_and(A: np.ndarray, B: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    d = A.shape[0]
    I = np.eye(d, dtype=np.float64)
    A_inv = np.linalg.pinv(A + eps * I, rcond=eps)
    B_inv = np.linalg.pinv(B + eps * I, rcond=eps)
    inner = A_inv + B_inv - I
    C = np.linalg.pinv(inner + eps * I, rcond=eps)
    return (C + C.T) * 0.5


def _conceptor_or(A: np.ndarray, B: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """A ∨ B = ¬(¬A ∧ ¬B)."""
    return _conceptor_not(_conceptor_and(_conceptor_not(A), _conceptor_not(B), eps))


def _or_reduce(matrices: list, eps: float = 1e-5) -> np.ndarray:
    """Fold OR across a list of conceptor matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = _conceptor_or(result, m, eps)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Merge COAST .pt files with conceptor OR (∨) algebra.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Two or more .pt files produced by libero_compute_coast.py")
    ap.add_argument("--output", required=True,
                    help="Path for the merged .pt artifact")
    ap.add_argument("--eps", type=float, default=1e-5,
                    help="Regularisation for pseudo-inverse (default: 1e-5)")
    ap.add_argument("--direction", choices=["pos", "neg", "both"], default="pos",
                    help="Which conceptor direction to merge: pos / neg / both (default: pos)")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"Loading {len(inputs)} artifact(s) …")
    arts = [torch.load(p, map_location="cpu", weights_only=False) for p in inputs]

    for i, (p, a) in enumerate(zip(inputs, arts)):
        if a.get("kind") != "coast":
            raise ValueError(f"{p}: expected kind='coast', got {a.get('kind')}")
        print(f"  [{i}] {p.name}  split={a['split']}  "
              f"meaning={a['meaning']}  layers={len(a['layers'])}")

    # Union of layer keys available in ALL artifacts
    all_keys = [set(a["layers"].keys()) for a in arts]
    common_keys = sorted(all_keys[0].intersection(*all_keys[1:]), key=int)
    print(f"\nLayers present in ALL inputs: {common_keys}")

    dirs_to_merge = (["pos", "neg"] if args.direction == "both"
                     else [args.direction])

    merged_layers = {}
    for key in common_keys:
        merged_layers[key] = {}
        d = arts[0]["layers"][key]["d"]
        merged_layers[key]["d"] = d
        for dirn in dirs_to_merge:
            matrices = [a["layers"][key][dirn].numpy().astype(np.float64)
                        for a in arts]
            merged = _or_reduce(matrices, eps=args.eps).astype(np.float32)
            merged_layers[key][dirn] = torch.from_numpy(merged)
        # Fill in missing direction with identity if only one was merged
        for dirn in ["pos", "neg"]:
            if dirn not in merged_layers[key]:
                merged_layers[key][dirn] = torch.eye(d)

    # Diagnostic: mean eigenvalue per layer
    tr = [float(np.trace(merged_layers[k]["pos"].numpy())) / merged_layers[k]["d"]
          for k in merged_layers]
    print(f"\nMerged OR conceptor  mean-eigval  min={min(tr):.4f} max={max(tr):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind":         "coast",
        "split":        "merged_or",
        "meaning":      {"pos": "grasp_primitive_or", "neg": "approach_or"},
        "positive_only": arts[0].get("positive_only", False),
        "n_layers":     arts[0]["n_layers"],
        "aperture":     arts[0]["aperture"],
        "z_pos_cm":     float(np.mean([a.get("z_pos_cm", 0) for a in arts])),
        "z_neg_cm":     float(np.mean([a.get("z_neg_cm", 0) for a in arts])),
        "n_success":    sum(a.get("n_success", 0) for a in arts),
        "n_rollouts":   sum(a.get("n_rollouts", 0) for a in arts),
        "source_files": [str(p) for p in inputs],
        "layers":       merged_layers,
        "hook_point":   "lm_expert.layers[i].mlp  (forward hook on output)",
    }, out)
    print(f"[✓] Saved merged conceptor ({len(merged_layers)} layers) → {out}")


if __name__ == "__main__":
    main()
