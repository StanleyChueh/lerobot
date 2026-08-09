#!/usr/bin/env python3
"""
DiMaS steering-LAYER scan (paper Appendix E ablation).

One extraction pass captures action-expert mlp reps at EVERY layer, builds a
DiMaS OT artifact per layer, then open-loop probes the dz (height) separation
of each so we can pick the best steering layer BEFORE paying for closed-loop.

Motivation: layer 8/16 was a guess. Deeper action-expert layers are more
linearly separable (paper Sec 5), and steering a LATER layer leaves less network
downstream to "self-correct" the intervention — the main failure mode of
height steering on SmolVLA.

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_dimas_layer_scan.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_three_cams_40k \\
    --dataset-repo-id ethanCSL/svla_franka_pick_n_place_vla_steering_libero_three_cams \\
    --feature height --out-dir outputs/dimas_layerscan \\
    --n-eps 40 --stride 10 --probe-alpha 8.0
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
    iter_carry_frames_dataset, DEFAULT_TASK,
)
from lerobot.scripts.libero_build_dimas import feature_from_action, barycentric_map
from lerobot.scripts.libero_eval_steering import setup_dimas, clear_steering

_dn = open(os.devnull, "w")
@contextlib.contextmanager
def silence():
    o = sys.stdout; sys.stdout = _dn
    try: yield
    finally: sys.stdout = o


def build_artifact_for_layer(R, F, layer, num_steps, policy_path, feature,
                             tau, max_samples, reg, n_iter, device):
    mean = R.mean(0); std = R.std(0) + 1e-6
    Rs = (R - mean) / std
    q_lo = np.quantile(F, tau); q_hi = np.quantile(F, 1 - tau)
    lo_mask = F <= q_lo; hi_mask = F >= q_hi
    rng = np.random.default_rng(0)
    def sub(m):
        idx = np.where(m)[0]
        return rng.choice(idx, max_samples, replace=False) if len(idx) > max_samples else idx
    lo_idx, hi_idx = sub(lo_mask), sub(hi_mask)
    D_low = torch.from_numpy(Rs[lo_idx]).to(device)
    D_high = torch.from_numpy(Rs[hi_idx]).to(device)
    with silence():
        high_t = barycentric_map(D_low, D_high, reg, n_iter)
        low_t = barycentric_map(D_high, D_low, reg, n_iter)
    from sklearn.linear_model import LogisticRegression
    Xg = np.concatenate([Rs[lo_idx], Rs[hi_idx]]); yg = np.concatenate([np.zeros(len(lo_idx)), np.ones(len(hi_idx))])
    clf = LogisticRegression(max_iter=2000).fit(Xg, yg)
    return {
        "kind": "dimas", "feature": feature, "layer": layer, "num_steps": num_steps,
        "policy_path": policy_path,
        "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        "tau": tau, "q_lo": q_lo, "q_hi": q_hi,
        "high": {"source": D_low.cpu(), "target": high_t.cpu()},
        "low":  {"source": D_high.cpu(), "target": low_t.cpu()},
        "gate_w": torch.from_numpy(clf.coef_[0].astype(np.float32)),
        "gate_b": float(clf.intercept_[0]), "gate_acc": clf.score(Xg, yg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--dataset-revision", default="main")
    ap.add_argument("--feature", choices=["speed", "height"], default="height")
    ap.add_argument("--out-dir", default="outputs/dimas_layerscan")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--n-eps", type=int, default=40)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--max-samples", type=int, default=1500)
    ap.add_argument("--sinkhorn-reg", type=float, default=0.05)
    ap.add_argument("--sinkhorn-iter", type=int, default=200)
    ap.add_argument("--probe-alpha", type=float, default=8.0)
    ap.add_argument("--probe-frames", type=int, default=40)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nLoading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device); policy.config.n_action_steps = 1
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy_path)
    layers = policy.model.vlm_with_expert.lm_expert.layers
    n_layers = len(layers)
    num_steps = policy.config.num_steps

    # hook ALL layers' mlp
    per_layer = [[] for _ in range(n_layers)]
    handles = []
    for i, lyr in enumerate(layers):
        def mk(idx):
            def hook(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                per_layer[idx].append(h.detach().mean(dim=1).squeeze(0).cpu().float())
            return hook
        handles.append(lyr.mlp.register_forward_hook(mk(i)))

    print(f"Loading dataset: {args.dataset_repo_id}")
    ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, revision=args.dataset_revision)
    ak, wk, fk = detect_camera_keys(ds)
    bounds = episode_bounds(ds); zmap = episode_carry_heights(ds, bounds)
    eps = sorted(zmap, key=zmap.get)
    if args.n_eps < len(eps):
        idx = np.linspace(0, len(eps) - 1, args.n_eps).round().astype(int)
        eps = [eps[i] for i in idx]

    # extraction: reps per layer + per-frame feature (repeated across denoise steps)
    R_layers = [[] for _ in range(n_layers)]
    feats = []
    for a, w, f, s in tqdm(iter_carry_frames_dataset(ds, eps, bounds, ak, wk, fk, stride=args.stride),
                           desc="extract(all layers)", unit="frame"):
        for buf in per_layer: buf.clear()
        with torch.no_grad(), silence():
            act = predict_action(observation=build_obs(a, w, s, f), policy=policy, device=device,
                                 preprocessor=pre, postprocessor=post, use_amp=False, task=args.task)
        policy.reset()
        act = (act.detach().cpu().numpy() if torch.is_tensor(act) else np.asarray(act)).reshape(-1)
        phi = feature_from_action(act, args.feature)
        m = len(per_layer[0])   # denoising steps captured this frame
        for li in range(n_layers):
            for r in per_layer[li]:
                R_layers[li].append(r.numpy())
        feats.extend([phi] * m)
    for h in handles: h.remove()

    F = np.asarray(feats, dtype=np.float64)
    print(f"\nExtracted {len(F)} reps/layer.  feature[{args.feature}] "
          f"min/med/max={F.min():.3f}/{np.median(F):.3f}/{F.max():.3f}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    arts = {}
    for li in range(n_layers):
        R = np.asarray(R_layers[li], dtype=np.float32)
        art = build_artifact_for_layer(R, F, li, num_steps, args.policy_path, args.feature,
                                       args.tau, args.max_samples, args.sinkhorn_reg,
                                       args.sinkhorn_iter, device)
        p = out_dir / f"L{li:02d}.pt"; torch.save(art, p); arts[li] = p

    # ── open-loop probe: dz separation per layer at probe-alpha ────────────────
    probe = []
    for a, w, f, s in iter_carry_frames_dataset(ds, eps[:5], bounds, ak, wk, fk, stride=15):
        probe.append((a, w, f, s))
        if len(probe) >= args.probe_frames: break

    def meas():
        vals = []
        for a, w, f, s in probe:
            policy.reset()
            with torch.no_grad(), silence():
                act = predict_action(observation=build_obs(a, w, s, f), policy=policy, device=device,
                                     preprocessor=pre, postprocessor=post, use_amp=False, task=args.task)
            act = (act.detach().cpu().numpy() if torch.is_tensor(act) else np.asarray(act)).reshape(-1)
            vals.append(float(act[2]) if args.feature == "height" else float(np.linalg.norm(act[:3])))
        return float(np.mean(vals))

    clear_steering(policy); base = meas()
    print(f"\n{'layer':>6} {'gate_acc':>9} {'high_d':>9} {'low_d':>9} {'sep':>9}   (probe alpha={args.probe_alpha}, base={base:+.4f})")
    print("-" * 60)
    results = []
    for li in range(n_layers):
        with silence(): setup_dimas(policy, str(arts[li]), "high", alpha=args.probe_alpha, gate=False)
        hi = meas()
        with silence(): setup_dimas(policy, str(arts[li]), "low", alpha=args.probe_alpha, gate=False)
        lo = meas()
        with silence(): clear_steering(policy)
        art = torch.load(arts[li], map_location="cpu", weights_only=False)
        sep = hi - lo
        results.append((li, sep, art["gate_acc"]))
        print(f"{li:>6} {art['gate_acc']:>9.3f} {hi-base:>+9.4f} {lo-base:>+9.4f} {sep:>+9.4f}")

    results.sort(key=lambda r: -r[1])
    print("\nTop layers by |dz separation| (open-loop):")
    for li, sep, acc in results[:5]:
        print(f"  L{li}: sep={sep:+.4f}  gate_acc={acc:.3f}  ->  {arts[li]}")
    best = results[0][0]
    print(f"\n[✓] Best open-loop layer = L{best} ({arts[best]})")
    print(f"    Closed-loop it:  libero_osc_eval.py --conditions none dimas_high dimas_low "
          f"--dimas-path {arts[best]} --dimas-alpha {args.probe_alpha}")


if __name__ == "__main__":
    main()
