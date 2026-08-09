#!/usr/bin/env python3
"""
Fast OPEN-LOOP CAA alpha-sweep diagnostic.

For a handful of carry-phase frames (from the same HF dataset used to build the
CAA vectors), measure the mean commanded vertical action dz = action[2] under
none / caa_high(+a) / caa_low(-a) across a range of alphas.

This is fast (no env stepping) and answers two questions before you spend
15 min/condition on a closed-loop eval:
  1. DIRECTION: is dz_high > dz_none > dz_low ?  (vector sign correct)
  2. MAGNITUDE: at what alpha does dz separate WITHOUT the non-vertical action
     components blowing up (a proxy for "task still intact")?

The reported ||act|| is the L2 norm of the FULL 7D action; if it explodes far
above the baseline, the policy is going OOD (task will break at that alpha).

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_caa_alpha_sweep.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_three_cams_40k \\
    --dataset-repo-id ethanCSL/svla_franka_pick_n_place_vla_steering_libero_three_cams \\
    --caa-path outputs/caa_three_cams_40k.pt \\
    --alphas 0.2 0.3 0.5 0.8 1.2 2.0 3.0 --n-frames 40
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch


@contextlib.contextmanager
def _silence_stdout():
    """Suppress the model's [DEBUG] prints so the sweep table stays clean."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lerobot.scripts.libero_compute_caa_osc import (
    build_obs, detect_camera_keys, episode_bounds, episode_carry_heights,
    iter_carry_frames_dataset, DEFAULT_TASK,
)
from lerobot.scripts.libero_eval_steering import setup_caa, clear_steering


def collect_frames(ds, n_frames, stride):
    """Grab up to n_frames carry-phase frames spanning the height range."""
    agent_key, wrist_key, front_key = detect_camera_keys(ds)
    bounds = episode_bounds(ds)
    zmap = episode_carry_heights(ds, bounds)
    ordered = sorted(zmap, key=zmap.get)
    # sample a spread of episodes (low, mid, high) so the probe is representative
    k = max(1, len(ordered) // 3)
    picks = ordered[:2] + ordered[len(ordered)//2 - 1:len(ordered)//2 + 1] + ordered[-2:]
    frames = []
    for agent, wrist, front, s8 in iter_carry_frames_dataset(
            ds, picks, bounds, agent_key, wrist_key, front_key, stride=stride):
        frames.append((agent, wrist, front, s8))
        if len(frames) >= n_frames:
            break
    return frames


def measure(policy, pre, post, device, frames, task):
    """Mean dz and mean ||full action|| over the frames, under current steering."""
    from lerobot.utils.control_utils import predict_action
    dzs, norms = [], []
    for agent, wrist, front, s8 in frames:
        policy.reset()
        with torch.no_grad(), _silence_stdout():
            act = predict_action(
                observation=build_obs(agent, wrist, s8, front), policy=policy,
                device=device, preprocessor=pre, postprocessor=post,
                use_amp=False, task=task)
        a = (act.detach().cpu().numpy() if torch.is_tensor(act) else np.asarray(act)).reshape(-1)
        dzs.append(float(a[2]))
        norms.append(float(np.linalg.norm(a[:6])))   # 6D pose delta (exclude gripper)
    return float(np.mean(dzs)), float(np.mean(norms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-revision", default="main")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--caa-path", required=True)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0])
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--layer-lo", type=int, default=None,
                    help="steer only layers [layer-lo, layer-hi)")
    ap.add_argument("--layer-hi", type=int, default=None)
    ap.add_argument("--exclude-last", action="store_true",
                    help="skip the final (often degenerate high-norm) expert layer")
    ap.add_argument("--normalize", action="store_true",
                    help="unit-normalize each layer vector (portable alpha)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nLoading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.config.n_action_steps = 1
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config,
                                         pretrained_path=args.policy_path)

    print(f"Loading dataset: {args.dataset_repo_id}")
    ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root,
                        revision=args.dataset_revision)
    frames = collect_frames(ds, args.n_frames, args.stride)
    print(f"Probe frames: {len(frames)}\n")

    # baseline
    clear_steering(policy)
    dz0, nrm0 = measure(policy, pre, post, device, frames, args.task)
    print(f"{'alpha':>7} | {'dz_low':>8} {'dz_none':>8} {'dz_high':>8} | "
          f"{'sep(H-L)':>9} | {'|a|_low':>8} {'|a|_none':>8} {'|a|_high':>8} | task-intact?")
    print("-" * 100)
    print(f"{'0.0':>7} | {dz0:>8.3f} {dz0:>8.3f} {dz0:>8.3f} | {'0.000':>9} | "
          f"{nrm0:>8.3f} {nrm0:>8.3f} {nrm0:>8.3f} | (baseline)")

    best = None
    for a in args.alphas:
        with _silence_stdout():
            setup_caa(policy, args.caa_path, alpha=+a, layer_lo=args.layer_lo,
                      layer_hi=args.layer_hi, exclude_last=args.exclude_last,
                      normalize=args.normalize)
        dzH, nrmH = measure(policy, pre, post, device, frames, args.task)
        with _silence_stdout():
            setup_caa(policy, args.caa_path, alpha=-a, layer_lo=args.layer_lo,
                      layer_hi=args.layer_hi, exclude_last=args.exclude_last,
                      normalize=args.normalize)
        dzL, nrmL = measure(policy, pre, post, device, frames, args.task)
        clear_steering(policy)
        sep = dzH - dzL
        # "task intact" heuristic: steered action norm within 1.6x of baseline
        blow = max(nrmH, nrmL) / max(nrm0, 1e-6)
        intact = "yes" if blow < 1.6 else ("marginal" if blow < 2.2 else "NO (OOD)")
        flag = ""
        if sep > 0 and blow < 1.6:
            score = sep / blow
            if best is None or score > best[1]:
                best = (a, score); flag = "  <- candidate"
        print(f"{a:>7.2f} | {dzL:>8.3f} {dz0:>8.3f} {dzH:>8.3f} | {sep:>9.3f} | "
              f"{nrmL:>8.3f} {nrm0:>8.3f} {nrmH:>8.3f} | {intact}{flag}")

    print("\nInterpretation:")
    print("  dz_high should be > dz_none > dz_low  (positive separation = correct direction).")
    print("  Pick the LARGEST alpha whose 'task-intact?' is still 'yes' → strongest safe steering.")
    if best:
        print(f"\n  >>> Recommended alpha to try closed-loop: {best[0]:.2f} "
              f"(best separation-per-OOD-cost)")


if __name__ == "__main__":
    main()
