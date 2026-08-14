#!/usr/bin/env python3
"""
DiMaS closed-loop ALPHA SWEEP (dose-response) for the paper checkpoint.
──────────────────────────────────────────────────────────────────────
Loads the policy ONCE and sweeps interpolation strength alpha for both the
'high' and 'low' DiMaS directions at a fixed layer, running k rollouts each.
Reports the PREDICTED-ACTION feature (the quantity DiMaS steers, per
arXiv:2607.14280) + realized carry-height + task success, so a monotonic
alpha->feature trend is direct dose-response evidence of steering.

Usage:
  python src/lerobot/scripts/libero_dimas_alpha_sweep.py \
    --policy-path HuggingFaceVLA/smolvla_libero --input-format standard \
    --dimas-path outputs/dimas_paper_repro/layerscan_speed/L09.pt \
    --directions low --alphas 0 3 5 8 --n 10 --feature-name speed
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lerobot.scripts.libero_osc_eval as osc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--input-format", choices=["custom", "standard"], default="standard")
    ap.add_argument("--dimas-path", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--task", default="pick up the black bowl between the plate "
                    "and the ramekin and place it on the plate")
    ap.add_argument("--directions", nargs="+", default=["high", "low"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0, 3, 5, 8])
    ap.add_argument("--no-gate", action="store_true", default=True)
    ap.add_argument("--gate", dest="no_gate", action="store_false")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--feature-name", default="speed")
    ap.add_argument("--step-ms", nargs="+", type=int, default=None,
                    help="denoising steps to inject at (paper injects at ONE step). "
                         "If given, sweep these at alphas[0]; else sweep alphas at all-steps.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    osc.INPUT_FORMAT = args.input_format
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device
    from lerobot.scripts.libero_eval_steering import (
        setup_dimas, clear_steering, enable_determinism)
    from libero.libero import benchmark

    enable_determinism()
    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")

    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    fname = suite_obj.get_task_bddl_files()[args.task_idx]
    bddl_path = osc.BDDL_ROOT / args.suite / fname
    init_states = osc.load_demo_init_states(osc.find_demo_file(args.suite, fname))

    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.config.n_action_steps = args.n_action_steps
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy_path)

    env = osc.make_env(bddl_path)
    obs0 = env.reset()
    plate_key = next((k for k in obs0 if "plate" in k.lower()
                      and k.endswith("_pos") and "to_robot" not in k), None)

    def run_block(label, apply_fn):
        apply_fn()
        pa_sp, pa_dz, carry, real_sp, succ = [], [], [], [], 0
        for ri in range(args.n):
            r = osc.run_rollout(env, init_states[ri % len(init_states)], policy, pre, post,
                                device, args.task, args.max_steps, plate_key)
            if len(r["act_speed"]):
                pa_sp.append(float(np.mean(r["act_speed"])))
                pa_dz.append(float(np.mean(r["act_dz"])))
            e = np.asarray(r["eef_heights"]) * 100
            if len(e) > 4:
                lo, hi = int(0.2 * len(e)), int(0.8 * len(e))
                carry.append(float(np.mean(e[lo:hi])))
            p = np.asarray(r["eef_xyz"]) * 100                    # realized EEF velocity
            if len(p) > 1:
                real_sp.append(float(np.mean(np.linalg.norm(np.diff(p, axis=0), axis=1))))
            succ += int(r["on_plate"])
        row = (label, float(np.mean(pa_sp)), float(np.mean(pa_dz)),
               float(np.mean(carry)), 100.0 * succ / args.n, float(np.mean(real_sp)))
        print(f"  {row[0]:<20} pred_speed={row[1]:.4f}  real_eef_speed={row[5]:.3f}cm/step  "
              f"pred_dz={row[2]:+.4f}  carry={row[3]:.1f}cm  success={row[4]:.0f}%", flush=True)
        return row

    art_layer = torch.load(args.dimas_path, map_location="cpu", weights_only=False)["layer"]
    mode = "STEP-M sweep" if args.step_ms else "ALPHA sweep"
    print(f"\n=== {mode}  feature={args.feature_name}  layer=L{art_layer}  "
          f"gate={'OFF' if args.no_gate else 'ON'}  ns={args.n_action_steps}  n={args.n} ===")
    rows = [run_block("baseline(a=0)", lambda: clear_steering(policy))]
    if args.step_ms:                     # sweep denoising step at a fixed alpha
        a = args.alphas[0]
        for d in args.directions:
            for m in args.step_ms:
                rows.append(run_block(
                    f"dimas_{d}_m{m}_a{a:g}",
                    lambda d=d, a=a, m=m: setup_dimas(policy, args.dimas_path, d, alpha=a,
                                                      gate=not args.no_gate, step_m=m)))
    else:                                # sweep alpha, inject at all steps
        for d in args.directions:
            for a in args.alphas:
                if a == 0:
                    continue
                rows.append(run_block(
                    f"dimas_{d}_a{a:g}",
                    lambda d=d, a=a: setup_dimas(policy, args.dimas_path, d, alpha=a,
                                                 gate=not args.no_gate)))
    clear_steering(policy); env.close()

    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump([{"cond": r[0], "pred_speed": r[1], "pred_dz": r[2],
                    "carry_cm": r[3], "success_pct": r[4], "real_eef_speed": r[5]} for r in rows],
                  open(args.out, "w"), indent=2)
        print(f"[✓] {args.out}")


if __name__ == "__main__":
    main()
