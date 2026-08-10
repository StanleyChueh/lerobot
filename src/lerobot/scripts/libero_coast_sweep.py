#!/usr/bin/env python3
"""
Efficient COAST layer/β sweep for the SmolVLA OSC model (paper arxiv:2605.17144).

Loads the policy ONCE, evaluates the baseline ONCE over a fixed set of demo init
states, then applies each (layer ℓ, β) COAST config and re-runs the SAME init
states. Because the model uses a deterministic eval-noise seed and the init
states are fixed, every condition is a PAIRED comparison against the identical
baseline rollouts — so a success-rate delta is attributable to the steering.

Reports, per config: success%, grasp%, and the paired improvement over baseline.

Usage:
  SMOLVLA_DEBUG=0 conda run -n lerobot python src/lerobot/scripts/libero_coast_sweep.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_natural \\
    --coast-path outputs/coast_repro/coast_task6.pt \\
    --task-idx 6 --task "pick up the black bowl next to the cookie box and place it on the plate" \\
    --n-rollouts 25 --max-steps 400 --n-action-steps 10 \\
    --layers 4 6 8 10 --betas 0.3 0.5 0.7 1.0 \\
    --out outputs/coast_repro/sweep_task6.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Must be set before the CUDA context / first cuBLAS call for deterministic matmuls.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--coast-path", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--task", default="pick the akita black bowl between the plate "
                    "and the ramekin and place it on the plate")
    ap.add_argument("--n-rollouts", type=int, default=25,
                    help="fixed init states evaluated per condition (paired); "
                         "ignored if --init-indices is given")
    ap.add_argument("--init-indices", type=int, nargs="+", default=None,
                    help="explicit demo init-state indices to evaluate (e.g. the failing "
                         "states + a sample of successes) — fast targeted sweep")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--num-steps", type=int, default=None,
                    help="flow-matching denoising steps (config.num_steps); MUST match "
                         "the setting used when the COAST conceptors were fit.")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 6, 8, 10],
                    help="expert layer indices ℓ to try (each steered alone)")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--direction", default="success",
                    help="COAST direction alias to steer toward (success/pos/high)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/coast_repro/sweep.txt")
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device
    from libero.libero import benchmark
    from lerobot.scripts.libero_eval_steering import setup_coast, clear_steering, enable_determinism
    from lerobot.scripts.libero_osc_eval import (
        make_env, find_demo_file, load_demo_init_states, run_rollout, BDDL_ROOT)

    enable_determinism()   # reproducible, history-independent rollouts (paired comparison)
    device = get_safe_torch_device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  (deterministic algorithms ON)")

    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    fname = suite_obj.get_task_bddl_files()[args.task_idx]
    bddl_path = BDDL_ROOT / args.suite / fname
    print(f"Task {args.task_idx}: {fname}")

    demo_file = find_demo_file(args.suite, fname)
    all_init = load_demo_init_states(demo_file)
    if args.init_indices is not None:
        idxs = [i for i in args.init_indices if 0 <= i < len(all_init)]
        init_states = [all_init[i] for i in idxs]
        print(f"Using {len(init_states)} explicit init states: {idxs}")
    else:
        K = min(args.n_rollouts, len(all_init))
        init_states = all_init[:K]
        print(f"Using first {K} fixed init states (paired eval)")
    K = len(init_states)

    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.config.n_action_steps = args.n_action_steps
    if args.num_steps is not None:
        policy.config.num_steps = args.num_steps
        print(f"flow-matching num_steps = {policy.config.num_steps}")
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy_path)

    env = make_env(bddl_path)
    obs0 = env.reset()
    plate_key = next((k for k in obs0 if "plate" in k.lower()
                      and k.endswith("_pos") and "to_robot" not in k), None)

    def eval_condition():
        """Run K paired rollouts under the CURRENT steering; return (grasp, success, per_state_success)."""
        g = s = 0
        per = []
        for i in range(K):
            r = run_rollout(env, init_states[i], policy, preprocessor, postprocessor,
                            device, args.task, args.max_steps, plate_key, phase_gate=False)
            g += int(r["grasped"]); s += int(r["on_plate"]); per.append(int(r["on_plate"]))
        return g, s, per

    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    # ── baseline (once) ─────────────────────────────────────────────────────────
    clear_steering(policy)
    t0 = time.time()
    g0, s0, base_per = eval_condition()
    log(f"\n{'='*72}")
    log(f"COAST sweep — task {args.task_idx}  ({Path(args.policy_path).name})")
    log(f"K={K} paired init states, direction={args.direction}, coast={Path(args.coast_path).name}")
    log(f"{'='*72}")
    log(f"{'config':<22}{'grasp%':>8}{'success%':>10}{'Δsucc':>8}  {'(succ/K)':>10}")
    log(f"{'baseline (none)':<22}{100*g0/K:>7.0f}%{100*s0/K:>9.0f}%{'—':>8}  {f'{s0}/{K}':>10}")
    log(f"  baseline time: {(time.time()-t0)/60:.1f} min")

    # ── sweep (layer, beta) ─────────────────────────────────────────────────────
    best = None
    for L in args.layers:
        for b in args.betas:
            clear_steering(policy)
            setup_coast(policy, args.coast_path, args.direction, beta=b,
                        layer_lo=L, layer_hi=L + 1)
            g, s, per = eval_condition()
            d = s - s0
            # count flips: baseline-fail → steered-success (the wins) vs regressions
            wins = sum(1 for a, c in zip(base_per, per) if c and not a)
            loss = sum(1 for a, c in zip(base_per, per) if a and not c)
            tag = f"L{L} β{b}"
            flag = ""
            if best is None or s > best[1]:
                best = (tag, s, b, L); flag = "  <-- best"
            log(f"{tag:<22}{100*g/K:>7.0f}%{100*s/K:>9.0f}%{d:>+8}  "
                f"{f'{s}/{K}':>10}  (+{wins}/-{loss}){flag}")

    clear_steering(policy)
    env.close()
    log(f"\nBest: {best[0]}  success={best[1]}/{K} ({100*best[1]/K:.0f}%)  "
        f"vs baseline {s0}/{K} ({100*s0/K:.0f}%)  Δ={best[1]-s0:+d}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\n[✓] sweep table → {args.out}")


if __name__ == "__main__":
    main()
