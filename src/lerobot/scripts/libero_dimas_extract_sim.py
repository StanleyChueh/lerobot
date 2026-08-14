#!/usr/bin/env python3
"""
DiMaS self-labeled rep extraction from CLOSED-LOOP SIM rollouts (paper-faithful).
──────────────────────────────────────────────────────────────────────────────
For the paper checkpoint HuggingFaceVLA/smolvla_libero (arXiv:2607.14280), which
has NO offline dataset downloaded here. Instead of reading a LeRobotDataset, we
run the policy in the LIBERO sim, hook every action-expert layer's mlp, and
SELF-LABEL each captured representation by the model's OWN predicted action
(speed = ||Δxyz||, height = Δz) — exactly DiMaS's self-labeling.

Reps are collected ONCE and labeled for BOTH features (same residual stream,
different scalar label), so one rollout pass builds both speed and height
artifacts for all layers. Then a cheap open-loop probe ranks layers per feature.

Usage:
  python src/lerobot/scripts/libero_dimas_extract_sim.py \
    --policy-path HuggingFaceVLA/smolvla_libero --input-format standard \
    --suite libero_spatial --task-idx 0 \
    --task "pick up the black bowl between the plate and the ramekin and place it on the plate" \
    --n-eps 20 --out-dir outputs/dimas_paper_repro
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lerobot.scripts.libero_build_dimas import feature_from_action  # noqa: E402
from lerobot.scripts.libero_dimas_layer_scan import build_artifact_for_layer, silence  # noqa: E402
import lerobot.scripts.libero_osc_eval as osc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--input-format", choices=["custom", "standard"], default="standard")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--task", default="pick up the black bowl between the plate "
                    "and the ramekin and place it on the plate")
    ap.add_argument("--out-dir", default="outputs/dimas_paper_repro")
    ap.add_argument("--n-eps", type=int, default=20, help="# init states / rollouts to collect from")
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--rep-stride", type=int, default=1, help="keep every k-th control step's reps")
    ap.add_argument("--features", nargs="+", default=["speed", "height"])
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--sinkhorn-reg", type=float, default=0.05)
    ap.add_argument("--sinkhorn-iter", type=int, default=200)
    ap.add_argument("--probe-alpha", type=float, default=8.0)
    ap.add_argument("--probe-frames", type=int, default=40)
    args = ap.parse_args()

    osc.INPUT_FORMAT = args.input_format

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device
    from lerobot.scripts.libero_eval_steering import (
        setup_dimas, clear_steering, enable_determinism)
    from libero.libero import benchmark

    enable_determinism()
    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    fname = suite_obj.get_task_bddl_files()[args.task_idx]
    bddl_path = osc.BDDL_ROOT / args.suite / fname
    print(f"Task {args.task_idx}: {fname}")
    demo_file = osc.find_demo_file(args.suite, fname)
    init_states = osc.load_demo_init_states(demo_file)
    print(f"Loaded {len(init_states)} init states; using {args.n_eps}")

    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.config.n_action_steps = 1          # run the model EVERY step -> dense reps
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy_path)
    clear_steering(policy)

    layers = policy.model.vlm_with_expert.lm_expert.layers
    n_layers = len(layers)
    num_steps = policy.config.num_steps
    print(f"{n_layers} expert layers, {num_steps} denoising steps/forward")

    # hook all layers' mlp -> per-forward buffer (one entry per denoising step)
    per_layer = [[] for _ in range(n_layers)]
    handles = []
    for i, lyr in enumerate(layers):
        def mk(idx):
            def hook(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                per_layer[idx].append(h.detach().mean(dim=1).squeeze(0).cpu().float())
            return hook
        handles.append(lyr.mlp.register_forward_hook(mk(i)))

    env = osc.make_env(bddl_path)

    R_layers = [[] for _ in range(n_layers)]
    F = {f: [] for f in args.features}
    probe_frames = []

    for ep in tqdm(range(min(args.n_eps, len(init_states))), desc="collect rollouts"):
        policy.reset(); pre.reset(); post.reset()
        env.reset()
        obs = env.set_init_state(init_states[ep])
        for _ in range(5):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
        for t in range(args.max_steps):
            agent = np.ascontiguousarray(obs["agentview_image"][::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
            state = osc.eef_state(obs)
            for buf in per_layer:
                buf.clear()
            with torch.no_grad(), silence():
                action = predict_action(
                    observation=osc.build_obs(agent, wrist, state),
                    policy=policy, device=device,
                    preprocessor=pre, postprocessor=post, use_amp=False, task=args.task)
            act = (action.detach().cpu().numpy() if torch.is_tensor(action)
                   else np.asarray(action)).reshape(-1)[:7].astype(np.float32)
            m = len(per_layer[0])
            if t % args.rep_stride == 0 and m > 0:
                for li in range(n_layers):
                    for r in per_layer[li]:
                        R_layers[li].append(r.numpy())
                for f in args.features:
                    F[f].extend([feature_from_action(act, f)] * m)
                if len(probe_frames) < args.probe_frames and t % 7 == 0:
                    probe_frames.append((agent, wrist, state))
            obs, _, done, _ = env.step(act)
            if done:
                break
    for h in handles:
        h.remove()

    n_reps = len(R_layers[0])
    print(f"\nCollected {n_reps} reps/layer from {args.n_eps} rollouts.")
    for f in args.features:
        arr = np.asarray(F[f])
        print(f"  feature[{f}] min/med/max = {arr.min():.4f}/{np.median(arr):.4f}/{arr.max():.4f}")

    out_root = Path(args.out_dir)
    R_np = [np.asarray(R_layers[li], dtype=np.float32) for li in range(n_layers)]

    for feat in args.features:
        Fv = np.asarray(F[feat], dtype=np.float64)
        fdir = out_root / f"layerscan_{feat}"
        fdir.mkdir(parents=True, exist_ok=True)
        arts = {}
        for li in range(n_layers):
            art = build_artifact_for_layer(
                R_np[li], Fv, li, num_steps, args.policy_path, feat,
                args.tau, args.max_samples, args.sinkhorn_reg, args.sinkhorn_iter, device)
            p = fdir / f"L{li:02d}.pt"
            torch.save(art, p); arts[li] = p

        # ── open-loop probe: feature shift high vs low at probe-alpha, per layer ──
        def meas():
            vals = []
            for a, w, s in probe_frames:
                policy.reset()
                with torch.no_grad(), silence():
                    act = predict_action(observation=osc.build_obs(a, w, s), policy=policy,
                                         device=device, preprocessor=pre, postprocessor=post,
                                         use_amp=False, task=args.task)
                act = (act.detach().cpu().numpy() if torch.is_tensor(act) else np.asarray(act)).reshape(-1)
                vals.append(float(np.linalg.norm(act[:3])) if feat == "speed" else float(act[2]))
            return float(np.mean(vals))

        clear_steering(policy); base = meas()
        print(f"\n=== feature={feat}  probe (alpha={args.probe_alpha}, base={base:+.4f}) ===")
        print(f"{'layer':>6} {'gate_acc':>9} {'high_d':>9} {'low_d':>9} {'sep':>9}")
        results = []
        for li in range(n_layers):
            with silence(): setup_dimas(policy, str(arts[li]), "high", alpha=args.probe_alpha, gate=False)
            hi = meas()
            with silence(): setup_dimas(policy, str(arts[li]), "low", alpha=args.probe_alpha, gate=False)
            lo = meas()
            with silence(): clear_steering(policy)
            art = torch.load(arts[li], map_location="cpu", weights_only=False)
            sep = hi - lo
            results.append((li, sep, hi - base, lo - base, art["gate_acc"]))
            print(f"{li:>6} {art['gate_acc']:>9.3f} {hi-base:>+9.4f} {lo-base:>+9.4f} {sep:>+9.4f}")
        results.sort(key=lambda r: -r[1])
        best = results[0][0]
        print(f"\n[feature={feat}] top layers by open-loop separation:")
        for li, sep, hd, ld, acc in results[:5]:
            print(f"  L{li}: sep={sep:+.4f} high_d={hd:+.4f} low_d={ld:+.4f} gate_acc={acc:.3f}")
        print(f"[✓] best {feat} layer = L{best}  -> {arts[best]}")

    env.close()
    print(f"\n[✓] artifacts under {out_root}/layerscan_<feature>/L*.pt")


if __name__ == "__main__":
    main()
