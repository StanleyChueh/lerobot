#!/usr/bin/env python3
"""
Compute CAA (Contrastive Activation Addition) steering vectors for an OSC-format
SmolVLA (dataset: agentview + robot0_eye_in_hand cameras, 8D eef state, 7D OSC
action).  Companion to libero_compute_caa.py (which is for the joint model).

  caa_vector[layer] = mean(lm_expert MLP output over HIGH carry frames)
                    - mean(lm_expert MLP output over LOW carry frames)

Captured at layer.mlp output. SmolVLA's fused VLM+expert forward decomposes each
layer manually (no layer.forward call), so a layer-level residual hook never fires;
layer.mlp is the reachable contrast point, and a delta injected there propagates
into the residual stream via the forward's `out_emb += after_first_residual`.

The expert-space contrast is MODEL-SPECIFIC (each model has its own action
expert), so run this separately per policy (natural, scripted-60k, ...).

Frames come from the high/low height-demo HDF5s OR a HF LeRobot dataset,
fed through the SAME 2-camera OSC obs the model sees at rollout.

Usage (HF dataset, recommended):
  conda run -n lerobot python src/lerobot/scripts/libero_compute_caa_osc.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_random_60k \\
    --dataset-repo-id ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_random \\
    --dataset-revision main \\
    --output outputs/caa_random_60k.pt --n-eps 50 --stride 10

Usage (local HDF5):
  conda run -n lerobot python src/lerobot/scripts/libero_compute_caa_osc.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_natural \\
    --high-hdf5 /home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5 \\
    --low-hdf5  /home/bruce/datasets/libero_height_demos/libero_spatial/low/task_00.hdf5 \\
    --output outputs/caa_osc_natural.pt --n-eps 25 --stride 10

Speed tips:
  --stride 20   : halves frame count, halves runtime (stride is the primary speed knob)
  --n-eps 25    : halves episode count vs default 50
  n_action_steps is forced to 1 internally (no multi-step decoding needed for activations)
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_EMPTY_256 = np.zeros((256, 256, 3), dtype=np.uint8)
_EMPTY_480 = np.zeros((480, 640, 3), dtype=np.uint8)

DEFAULT_TASK = ("pick the akita black bowl between the plate and the ramekin "
                "and place it on the plate")


def eef_state_from_hdf5(eef_pos, eef_quat, gripper_qpos):
    """8D OSC state = eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)."""
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([
        eef_pos.astype(np.float32),
        quat2axisangle(eef_quat).astype(np.float32),
        gripper_qpos.astype(np.float32),
    ])


def build_obs(agent_hwc, wrist_hwc, state8, front_hwc=None):
    """OSC 2-camera obs: camera1=agentview, camera2=wrist, camera3=blank."""
    return {
        "observation.images.camera1":        agent_hwc,
        "observation.images.camera2":        wrist_hwc,
        "observation.images.camera3":        front_hwc if front_hwc is not None else _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


def register_mlp_hooks(policy):
    """Capture each lm_expert layer's MLP OUTPUT (the contrast point).

    NOTE: SmolVLA's vlm_with_expert.forward() does NOT call layer.forward() — it
    manually decomposes each layer (self_attn.o_proj, post_attention_layernorm,
    mlp) in a fused VLM+expert attention loop (see smolvlm_with_expert.py L483+).
    So a forward hook on the whole `layer` NEVER fires; only submodule hooks
    (layer.mlp, layer.self_attn) do. We hook layer.mlp — the same point where a
    steering delta added here propagates into the residual stream, because the
    forward does `out_emb = layer.mlp(...)` then `out_emb += after_first_residual`.
    The injection in libero_eval_steering.setup_caa() MUST match this point."""
    layers = policy.model.vlm_with_expert.lm_expert.layers
    buffer = [[] for _ in range(len(layers))]
    handles = []
    for i, layer in enumerate(layers):
        def _make_hook(idx):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                buffer[idx].append(h.detach().mean(dim=1).squeeze(0).cpu().float())
            return hook
        handles.append(layer.mlp.register_forward_hook(_make_hook(i)))
    return handles, buffer


def iter_carry_frames(hdf5_path, n_eps, carry_lo=0.20, carry_hi=0.80, stride=10):
    """Yield (agentview, wrist, None, eef_state8) for carry-phase frames (HDF5 source)."""
    with h5py.File(hdf5_path, "r") as f:
        ep_keys = sorted(k for k in f.keys() if k.startswith("ep_"))
        if n_eps is not None:
            ep_keys = ep_keys[:n_eps]
        for key in ep_keys:
            g = f[key]
            agent = g["agentview_image"][()]
            wrist = g["eye_in_hand_image"][()]
            eef_p = g["eef_pos"][()]
            eef_q = g["eef_quat"][()]
            grip  = g["gripper_qpos"][()]
            T = len(agent)
            for t in range(int(carry_lo * T), int(carry_hi * T), stride):
                s8 = eef_state_from_hdf5(eef_p[t], eef_q[t], grip[t])
                yield agent[t], wrist[t], None, s8


def count_carry_frames_hdf5(hdf5_path, n_eps, carry_lo=0.20, carry_hi=0.80, stride=10):
    """Fast frame count for HDF5 (no image loading)."""
    total = 0
    with h5py.File(hdf5_path, "r") as f:
        ep_keys = sorted(k for k in f.keys() if k.startswith("ep_"))
        if n_eps is not None:
            ep_keys = ep_keys[:n_eps]
        for key in ep_keys:
            T = len(f[key]["agentview_image"])
            total += len(range(int(carry_lo * T), int(carry_hi * T), stride))
    return total


def _to_uint8_hwc(t):
    """LeRobot image tensor (CHW float [0,1]) → HWC uint8."""
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8) if a.max() <= 1.001 else a.astype(np.uint8)
    return a


def episode_bounds(ds):
    """dict episode_index → (from_frame, to_frame) via the episode_index column."""
    eidx = np.asarray(ds.hf_dataset["episode_index"])
    bounds = {}
    for e in np.unique(eidx):
        idxs = np.where(eidx == e)[0]
        bounds[int(e)] = (int(idxs[0]), int(idxs[-1]) + 1)
    return bounds


def episode_carry_heights(ds, bounds, carry_lo=0.20, carry_hi=0.80):
    """Mean carry-phase eef-z per episode from the state column (fast, no image decode)."""
    states = np.asarray([np.asarray(s, dtype=np.float32)
                         for s in ds.hf_dataset["observation.state"]])
    z = {}
    for ei, (f, t) in bounds.items():
        T = t - f
        lo, hi = f + int(carry_lo * T), f + int(carry_hi * T)
        if hi > lo:
            z[ei] = float(states[lo:hi, 2].mean())
    return z


def detect_camera_keys(ds):
    """Return (agent_key, wrist_key, front_key) — front_key is None if no frontview."""
    keys = [k for k in ds.meta.features if k.startswith("observation.images")]
    def is_wrist(k):
        return any(w in k for w in ("image2", "wrist", "eye_in_hand"))
    def is_front(k):
        return "sideview" in k or "frontview" in k
    wrist = next((k for k in keys if is_wrist(k)), None)
    front = next((k for k in keys if is_front(k)), None)
    agent = next((k for k in keys if k != wrist and k != front), None)
    if agent is None or wrist is None:
        raise RuntimeError(f"could not resolve agent/wrist cameras from {keys}")
    return agent, wrist, front


def iter_carry_frames_dataset(ds, ep_indices, bounds, agent_key, wrist_key,
                              front_key=None, carry_lo=0.20, carry_hi=0.80, stride=10):
    """Yield (agentview, wrist, front_or_None, eef_state) for carry-phase frames."""
    for ei in ep_indices:
        if ei not in bounds:
            continue
        f, t = bounds[ei]
        T = t - f
        for k in range(int(carry_lo * T), int(carry_hi * T), stride):
            row = ds[f + k]
            agent = _to_uint8_hwc(row[agent_key])
            wrist = _to_uint8_hwc(row[wrist_key])
            front = _to_uint8_hwc(row[front_key]) if front_key and front_key in row else None
            s8 = np.asarray(row["observation.state"], dtype=np.float32)
            if torch.is_tensor(row["observation.state"]):
                s8 = row["observation.state"].detach().cpu().numpy().astype(np.float32)
            # pad to 8D if dataset stores 6D state (eef_pos+axisangle without gripper)
            if s8.shape[0] == 6:
                s8 = np.concatenate([s8, np.zeros(2, dtype=np.float32)])
            yield agent, wrist, front, s8


def count_carry_frames_dataset(ep_indices, bounds, carry_lo=0.20, carry_hi=0.80, stride=10):
    """Fast frame count without loading images."""
    total = 0
    for ei in ep_indices:
        if ei not in bounds:
            continue
        f, t = bounds[ei]
        T = t - f
        total += len(range(int(carry_lo * T), int(carry_hi * T), stride))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--high-hdf5", default=None)
    ap.add_argument("--low-hdf5", default=None)
    ap.add_argument("--dataset-repo-id", default=None,
                    help="HF LeRobot dataset with BOTH high+low episodes; "
                         "auto-split by measured carry EEF height.")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--dataset-revision", default="main",
                    help="HF dataset revision ('main' avoids stale cached tags)")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-eps", type=int, default=25, help="episodes per arc (high, low)")
    ap.add_argument("--stride", type=int, default=10,
                    help="frame stride within carry phase — MAIN speed knob. "
                         "stride=10: ~100 frames/ep; stride=20: ~50 frames/ep (2× faster)")
    args = ap.parse_args()
    if not args.dataset_repo_id and not (args.high_hdf5 and args.low_hdf5):
        ap.error("provide either --dataset-repo-id OR both --high-hdf5 and --low-hdf5")

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.config.n_action_steps = 1   # only need activations, not multi-step actions → faster
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy_path)

    n_layers = len(policy.model.vlm_with_expert.lm_expert.layers)
    print(f"lm_expert: {n_layers} layers  (MLP-output CAA)")

    # ── Build per-condition frame iterators ────────────────────────────────────
    if args.dataset_repo_id:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        print(f"\nLoading dataset: {args.dataset_repo_id}  revision={args.dataset_revision}")
        ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root,
                            revision=args.dataset_revision)
        n_ep = ds.num_episodes
        print(f"  {n_ep} episodes, {len(ds)} frames total")

        agent_key, wrist_key, front_key = detect_camera_keys(ds)
        print(f"  cameras: agent={agent_key}, wrist={wrist_key}, front={front_key}")

        print("  Computing episode bounds ...", end="", flush=True)
        bounds = episode_bounds(ds)
        print(f" {len(bounds)} episodes found")

        if len(bounds) == 0:
            raise RuntimeError(
                "episode_bounds() returned empty — 'episode_index' column may be missing "
                "or the dataset has no episodes. Check ds.hf_dataset.column_names.")

        print("  Computing per-episode carry heights ...", end="", flush=True)
        zmap = episode_carry_heights(ds, bounds)
        print(f" done ({len(zmap)} episodes with valid carry phase)")

        ordered = sorted(zmap, key=zmap.get)   # low → high
        k = min(args.n_eps, len(ordered) // 2)
        if k == 0:
            raise RuntimeError(
                f"Not enough episodes for split: {len(ordered)} episodes, need at least 2. "
                f"Check that the dataset actually loaded (n_ep={n_ep}).")
        low_eps  = ordered[:k]
        high_eps = ordered[-k:]

        z_high = np.mean([zmap[e] for e in high_eps]) * 100
        z_low  = np.mean([zmap[e] for e in low_eps]) * 100
        print(f"\n  Episode split (top/bottom {k} by carry EEF-z):")
        print(f"    HIGH: {k} eps, mean z = {z_high:.1f} cm")
        print(f"    LOW:  {k} eps, mean z = {z_low:.1f} cm")
        print(f"    Separation: {z_high - z_low:.1f} cm", end="")
        if z_high - z_low < 3.0:
            print("  ← WARNING: very small gap, vectors may be weak!")
        else:
            print("  ← good separation")

        n_frames_high = count_carry_frames_dataset(high_eps, bounds, stride=args.stride)
        n_frames_low  = count_carry_frames_dataset(low_eps,  bounds, stride=args.stride)
        print(f"  Frames to process: HIGH={n_frames_high}, LOW={n_frames_low} "
              f"(stride={args.stride})")

        iters = [
            ("high", n_frames_high,
             lambda: iter_carry_frames_dataset(ds, high_eps, bounds, agent_key, wrist_key,
                                               front_key, stride=args.stride)),
            ("low",  n_frames_low,
             lambda: iter_carry_frames_dataset(ds, low_eps, bounds, agent_key, wrist_key,
                                               front_key, stride=args.stride)),
        ]
    else:
        n_high = count_carry_frames_hdf5(args.high_hdf5, args.n_eps, stride=args.stride)
        n_low  = count_carry_frames_hdf5(args.low_hdf5,  args.n_eps, stride=args.stride)
        print(f"\nFrames to process: HIGH={n_high}, LOW={n_low} (stride={args.stride})")
        iters = [
            ("high", n_high, lambda: iter_carry_frames(args.high_hdf5, args.n_eps, stride=args.stride)),
            ("low",  n_low,  lambda: iter_carry_frames(args.low_hdf5,  args.n_eps, stride=args.stride)),
        ]

    # ── Process each condition ─────────────────────────────────────────────────
    mean_acts = {}
    z_means   = {}
    t0_total  = time.time()

    for cond, n_frames_expected, make_iter in iters:
        print(f"\n{'─'*60}")
        print(f"Processing condition: {cond}  (expected ~{n_frames_expected} frames)")
        print(f"{'─'*60}")
        handles, buffer = register_mlp_hooks(policy)
        nf = 0
        zsum = 0.0
        t0 = time.time()

        with tqdm(total=n_frames_expected, desc=f"  {cond}",
                  unit="frame", dynamic_ncols=True) as pbar:
            for frame in make_iter():
                agent, wrist, front, s8 = frame
                zsum += float(s8[2])
                with torch.no_grad():
                    predict_action(
                        observation=build_obs(agent, wrist, s8, front),
                        policy=policy, device=device,
                        preprocessor=preprocessor, postprocessor=postprocessor,
                        use_amp=False, task=args.task,
                    )
                policy.reset()
                nf += 1
                pbar.update(1)
                if nf % 50 == 0:
                    elapsed = time.time() - t0
                    fps = nf / elapsed
                    pbar.set_postfix({"fps": f"{fps:.1f}", "z": f"{zsum/nf*100:.1f}cm"})

        for h in handles:
            h.remove()

        elapsed = time.time() - t0
        if nf == 0:
            raise RuntimeError(
                f"[{cond}] Zero frames were processed!\n"
                f"  This usually means the episode-index split produced empty episode lists,\n"
                f"  or all episodes had too few frames for the carry phase.\n"
                f"  Debug: high_eps={high_eps[:5] if 'high_eps' in dir() else 'N/A'}...\n"
                f"         low_eps={low_eps[:5] if 'low_eps' in dir() else 'N/A'}...\n"
                f"  Try running with a smaller --n-eps or check the dataset structure.")

        mean_acts[cond] = [torch.stack(b).mean(0).numpy() if b else None for b in buffer]
        z_means[cond]   = zsum / nf
        n_good = sum(1 for b in buffer if b)
        print(f"  Done: {nf} frames in {elapsed:.0f}s ({nf/elapsed:.1f} fps)")
        print(f"  Mean carry EEF-z = {z_means[cond]*100:.1f} cm")
        print(f"  Layers with activations: {n_good}/{n_layers}")

    # ── Build CAA vectors ──────────────────────────────────────────────────────
    total_time = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"Total processing time: {total_time:.0f}s")

    if z_means["high"] < z_means["low"]:
        print("[!] WARNING: 'high' mean z < 'low' mean z — episode split is reversed!\n"
              "    The CAA direction will be INVERTED. Either:\n"
              "    (a) swap --caa-alpha sign at eval  OR\n"
              "    (b) check that the dataset has clear high/low height separation.")

    caa = {}
    for i in range(n_layers):
        h = mean_acts["high"][i]
        l = mean_acts["low"][i]
        if h is not None and l is not None:
            caa[str(i)] = torch.from_numpy(h - l)

    if not caa:
        raise RuntimeError(
            "CAA dict is empty — all layer buffers were None. "
            "This means no activations were captured even though frames were processed. "
            "Check that register_mlp_hooks is hooking the correct module path:\n"
            "  policy.model.vlm_with_expert.lm_expert.layers[i].mlp")

    norms = [caa[k].norm().item() for k in caa]
    print(f"CAA vectors: {len(caa)}/{n_layers} layers")
    print(f"  norm min={min(norms):.4f}  mean={np.mean(norms):.4f}  max={max(norms):.4f}")
    print(f"  high mean z={z_means['high']*100:.1f}cm  vs  low mean z={z_means['low']*100:.1f}cm  "
          f"(Δ={( z_means['high']-z_means['low'])*100:.1f}cm)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vectors": caa,
        "policy_path": args.policy_path,
        "n_layers": n_layers,
        "format": "expert_mlp_output_high_minus_low",
        "z_high_cm": z_means["high"] * 100,
        "z_low_cm":  z_means["low"]  * 100,
    }, out)
    print(f"\n[✓] Saved → {out}")


if __name__ == "__main__":
    main()
