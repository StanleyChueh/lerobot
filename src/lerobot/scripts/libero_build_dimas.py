#!/usr/bin/env python3
"""
Build a DiMaS (Distribution-Matching Steering) artifact for an OSC SmolVLA.

Reference: Khayatan et al., "DiMaS: Distribution Matching for Steering VLA
Models" (arXiv:2607.14280). Classical linear/CAA steering fails on flow-matching
VLAs because behavioral features are linearly DECODABLE but not linearly
STEERABLE — a fixed mean-difference shift can't transport the low-feature
representation distribution onto the high-feature one (different shapes, not just
means). DiMaS instead learns an OPTIMAL-TRANSPORT map D-  ->  D+ between the tails
of the feature distribution, gates it with a "feature-absent" classifier, and
interpolates with alpha to preserve task success.

This script (TRAINING phase):
  1. Runs the base policy over dataset carry frames; captures the action-expert
     `layer.mlp` output (pooled over tokens) at the target layer, for every
     flow-matching denoising step (num_steps per frame).
       NOTE: SmolVLA's fused forward never calls layer.forward(), so we hook
       layer.mlp — the reachable action-expert point (see libero_compute_caa_osc).
  2. Labels each representation by a scalar behavioral feature computed from the
     frame's PREDICTED action:
         speed  = || predicted (dx,dy,dz) ||        (translational speed)
         height = predicted dz                       (signed vertical command)
  3. Splits reps into tails: D_low = {phi <= q_tau}, D_high = {phi >= q_{1-tau}}.
  4. Learns barycentric OT maps (entropic Sinkhorn) in BOTH directions:
         high: D_low  -> D_high   (steer a low rep up)
         low : D_high -> D_low    (steer a high rep down)
  5. Fits a logistic gate separating low vs high (used at inference to only steer
     reps that lack the target feature).

Inference is done by setup_dimas() in libero_eval_steering.py.

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_build_dimas.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_three_cams_40k \\
    --dataset-repo-id ethanCSL/svla_franka_pick_n_place_vla_steering_libero_three_cams \\
    --feature speed --layer 8 --output outputs/dimas_three_cams_speed_L8.pt \\
    --n-eps 60 --stride 8
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lerobot.scripts.libero_compute_caa_osc import (
    build_obs, detect_camera_keys, episode_bounds, episode_carry_heights,
    iter_carry_frames_dataset, iter_carry_frames, DEFAULT_TASK,
)


@contextlib.contextmanager
def _silence_stdout():
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def feature_from_action(act7, feature):
    """Scalar behavioral feature from a predicted 7D OSC action [dx,dy,dz,...,grip]."""
    a = np.asarray(act7, dtype=np.float64).reshape(-1)
    if feature == "speed":
        return float(np.linalg.norm(a[:3]))        # translational speed
    if feature == "height":
        return float(a[2])                          # signed vertical command dz
    raise ValueError(f"unknown feature {feature}")


def sinkhorn_plan(C, reg=0.05, n_iter=200):
    """Entropic OT plan between two uniform empirical distributions given cost C."""
    n, m = C.shape
    a = torch.full((n,), 1.0 / n, device=C.device, dtype=C.dtype)
    b = torch.full((m,), 1.0 / m, device=C.device, dtype=C.dtype)
    K = torch.exp(-C / reg) + 1e-30
    u = torch.ones_like(a)
    for _ in range(n_iter):
        v = b / (K.t() @ u + 1e-30)
        u = a / (K @ v + 1e-30)
    return u[:, None] * K * v[None, :]


def barycentric_map(source, target, reg=0.05, n_iter=200):
    """T(source_i) = sum_j P_ij target_j / sum_j P_ij  (barycentric OT projection)."""
    C = torch.cdist(source, target, p=2) ** 2
    C = C / (C.median() + 1e-12)          # scale cost so reg is meaningful
    P = sinkhorn_plan(C, reg=reg, n_iter=n_iter)
    mapped = (P @ target) / (P.sum(dim=1, keepdim=True) + 1e-30)
    return mapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset-repo-id", default=None)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--dataset-revision", default="main")
    ap.add_argument("--high-hdf5", default=None)
    ap.add_argument("--low-hdf5", default=None)
    ap.add_argument("--feature", choices=["speed", "height"], required=True)
    ap.add_argument("--layer", type=int, default=8,
                    help="action-expert layer to steer (intermediate layers work best)")
    ap.add_argument("--tau", type=float, default=0.30,
                    help="tail fraction: D_low=bottom tau, D_high=top tau of the feature")
    ap.add_argument("--max-samples", type=int, default=1500,
                    help="cap reps per tail (subsample for tractable NN at inference)")
    ap.add_argument("--sinkhorn-reg", type=float, default=0.05)
    ap.add_argument("--sinkhorn-iter", type=int, default=200)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-eps", type=int, default=50)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()
    if not args.dataset_repo_id and not (args.high_hdf5 and args.low_hdf5):
        ap.error("provide --dataset-repo-id OR both --high-hdf5/--low-hdf5")

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nLoading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config,
                                         pretrained_path=args.policy_path)

    layers = policy.model.vlm_with_expert.lm_expert.layers
    n_layers = len(layers)
    if not (0 <= args.layer < n_layers):
        ap.error(f"--layer must be in [0,{n_layers})")
    print(f"lm_expert: {n_layers} layers; steering layer = {args.layer}; "
          f"num_steps(denoise)={policy.config.num_steps}")

    # ── hook that captures pooled mlp-output reps at the target layer ──────────
    frame_reps = []   # filled per predict_action (num_steps entries), then drained
    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        frame_reps.append(h.detach().mean(dim=1).squeeze(0).cpu().float())
    handle = layers[args.layer].mlp.register_forward_hook(_hook)

    # ── frame iterator ─────────────────────────────────────────────────────────
    if args.dataset_repo_id:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        print(f"Loading dataset: {args.dataset_repo_id} (rev {args.dataset_revision})")
        ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root,
                            revision=args.dataset_revision)
        agent_key, wrist_key, front_key = detect_camera_keys(ds)
        bounds = episode_bounds(ds)
        # use a spread of episodes across the height range for good feature coverage
        zmap = episode_carry_heights(ds, bounds)
        eps = sorted(zmap, key=zmap.get)
        if args.n_eps < len(eps):
            idx = np.linspace(0, len(eps) - 1, args.n_eps).round().astype(int)
            eps = [eps[i] for i in idx]
        make_iter = lambda: iter_carry_frames_dataset(
            ds, eps, bounds, agent_key, wrist_key, front_key, stride=args.stride)
    else:
        def make_iter():
            for a, w, f, s in iter_carry_frames(args.high_hdf5, args.n_eps, stride=args.stride):
                yield a, w, f, s
            for a, w, f, s in iter_carry_frames(args.low_hdf5, args.n_eps, stride=args.stride):
                yield a, w, f, s

    # ── collect (rep, feature) pairs ───────────────────────────────────────────
    all_reps, all_feats = [], []
    n_frames = 0
    for frame in tqdm(make_iter(), desc="extracting reps", unit="frame"):
        agent, wrist, front, s8 = frame
        frame_reps.clear()
        with torch.no_grad(), _silence_stdout():
            act = predict_action(
                observation=build_obs(agent, wrist, s8, front), policy=policy,
                device=device, preprocessor=pre, postprocessor=post,
                use_amp=False, task=args.task)
        policy.reset()
        act = (act.detach().cpu().numpy() if torch.is_tensor(act) else np.asarray(act)).reshape(-1)
        phi = feature_from_action(act, args.feature)
        # every denoising-step rep for this frame shares the frame's action feature
        for r in frame_reps:
            all_reps.append(r.numpy())
            all_feats.append(phi)
        n_frames += 1
    handle.remove()

    R = np.asarray(all_reps, dtype=np.float32)       # (N, d)
    F = np.asarray(all_feats, dtype=np.float64)      # (N,)
    print(f"\nCollected {len(R)} reps from {n_frames} frames  (d={R.shape[1]})")
    print(f"feature[{args.feature}] min/med/max = "
          f"{F.min():.4f}/{np.median(F):.4f}/{F.max():.4f}")

    # ── standardize reps (for NN + gate + OT cost) ─────────────────────────────
    mean = R.mean(0)
    std = R.std(0) + 1e-6
    Rs = (R - mean) / std

    # ── quantile tails ─────────────────────────────────────────────────────────
    q_lo = np.quantile(F, args.tau)
    q_hi = np.quantile(F, 1.0 - args.tau)
    lo_mask = F <= q_lo
    hi_mask = F >= q_hi
    print(f"tails (tau={args.tau}): q_lo={q_lo:.4f} (n={lo_mask.sum()}), "
          f"q_hi={q_hi:.4f} (n={hi_mask.sum()})")
    if lo_mask.sum() < 10 or hi_mask.sum() < 10:
        raise RuntimeError("too few reps in a tail — increase --n-eps or --tau")

    def _subsample(mask):
        idx = np.where(mask)[0]
        if len(idx) > args.max_samples:
            idx = np.random.default_rng(0).choice(idx, args.max_samples, replace=False)
        return idx

    lo_idx = _subsample(lo_mask)
    hi_idx = _subsample(hi_mask)
    D_low = torch.from_numpy(Rs[lo_idx]).to(device)     # standardized
    D_high = torch.from_numpy(Rs[hi_idx]).to(device)
    print(f"OT sets: D_low={tuple(D_low.shape)}, D_high={tuple(D_high.shape)}")

    # ── barycentric OT maps in both directions (standardized space) ────────────
    print("Solving Sinkhorn OT: low->high ...", flush=True)
    high_target = barycentric_map(D_low, D_high, args.sinkhorn_reg, args.sinkhorn_iter)  # T(low)->high
    print("Solving Sinkhorn OT: high->low ...", flush=True)
    low_target = barycentric_map(D_high, D_low, args.sinkhorn_reg, args.sinkhorn_iter)   # T(high)->low

    # sanity: does transport move the feature-proxy (mean rep) toward the target tail?
    shift_hi = (high_target.mean(0) - D_low.mean(0)).norm().item()
    shift_lo = (low_target.mean(0) - D_high.mean(0)).norm().item()
    print(f"mean transport shift  low->high={shift_hi:.3f}  high->low={shift_lo:.3f} (std units)")

    # ── logistic gate: P(high | rep_std) ───────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    Xg = np.concatenate([Rs[lo_idx], Rs[hi_idx]], 0)
    yg = np.concatenate([np.zeros(len(lo_idx)), np.ones(len(hi_idx))])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xg, yg)
    gate_acc = clf.score(Xg, yg)
    print(f"gate (logistic low-vs-high) train acc = {gate_acc:.3f}  "
          f"(>0.9 = feature is decodable, DiMaS applicable)")

    # ── save artifact ──────────────────────────────────────────────────────────
    art = {
        "kind": "dimas",
        "feature": args.feature,
        "layer": args.layer,
        "num_steps": policy.config.num_steps,
        "policy_path": args.policy_path,
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "tau": args.tau, "q_lo": q_lo, "q_hi": q_hi,
        # standardized source samples + their barycentric OT images (standardized)
        "high": {"source": D_low.cpu(),  "target": high_target.cpu()},   # steer low->high
        "low":  {"source": D_high.cpu(), "target": low_target.cpu()},    # steer high->low
        "gate_w": torch.from_numpy(clf.coef_[0].astype(np.float32)),
        "gate_b": float(clf.intercept_[0]),
        "gate_acc": gate_acc,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, out)
    print(f"\n[✓] Saved DiMaS artifact → {out}")
    print(f"    Use: libero_osc_eval.py --conditions none dimas_high dimas_low "
          f"--dimas-path {out} --dimas-alpha 0.5")


if __name__ == "__main__":
    main()
