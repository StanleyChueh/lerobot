#!/usr/bin/env python3
"""
Compute CAA (Contrastive Activation Addition) steering vectors for an OSC-format
SmolVLA (dataset: agentview + robot0_eye_in_hand cameras, 8D eef state, 7D OSC
action).  Companion to libero_compute_caa.py (which is for the joint model).

  caa_vector[layer] = mean(lm_expert MLP output over HIGH carry frames)
                    - mean(lm_expert MLP output over LOW carry frames)

The expert-space contrast is MODEL-SPECIFIC (each model has its own action
expert), so run this separately per policy (natural, scripted-60k, ...).

Frames come from the high/low height-demo HDF5s, fed through the SAME 2-camera
OSC obs the model sees at rollout (camera1=agentview, camera2=wrist).

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_compute_caa_osc.py \
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_natural \
    --high-hdf5 /home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5 \
    --low-hdf5  /home/bruce/datasets/libero_height_demos/libero_spatial/low/task_00.hdf5 \
    --output outputs/caa_osc_natural.pt --n-eps 25 --stride 10
"""

import argparse
import sys
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


def build_obs(agent_hwc, wrist_hwc, state8):
    """OSC 2-camera obs: camera1=agentview, camera2=wrist, rest blank."""
    return {
        "observation.images.camera1":        agent_hwc,
        "observation.images.camera2":        wrist_hwc,
        "observation.images.camera3":        _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


def register_mlp_hooks(policy):
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
    """Yield (agentview, wrist, eef_state8) for carry-phase frames (HDF5 source)."""
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
                yield agent[t], wrist[t], s8


def _to_uint8_hwc(t):
    """LeRobot image tensor (CHW float [0,1]) → HWC uint8, matching env obs."""
    a = t.detach().cpu().numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8) if a.max() <= 1.001 else a.astype(np.uint8)
    return a


def episode_bounds(ds):
    """dict episode_index → (from_frame, to_frame) via the episode_index column."""
    eidx = np.asarray(ds.hf_dataset["episode_index"])
    return {int(e): (int(np.where(eidx == e)[0][0]), int(np.where(eidx == e)[0][-1]) + 1)
            for e in np.unique(eidx)}


def episode_carry_heights(ds, bounds, carry_lo=0.20, carry_hi=0.80):
    """Mean carry-phase eef-z (observation.state[2]) per episode — read from the
    state column only (no image decode, fast). This is the SELF-LABEL used to split
    high vs low episodes by their ACTUAL end-effector height."""
    states = np.asarray([np.asarray(s, dtype=np.float32)
                         for s in ds.hf_dataset["observation.state"]])  # (N, 8)
    z = {}
    for ei, (f, t) in bounds.items():
        T = t - f
        lo, hi = f + int(carry_lo * T), f + int(carry_hi * T)
        z[ei] = float(states[lo:hi, 2].mean())
    return z


def detect_camera_keys(ds):
    """Return (agent_key, wrist_key) for either naming convention
    (image/image2 or agentview/robot0_eye_in_hand). Wrist = camera2."""
    keys = [k for k in ds.meta.features if k.startswith("observation.images")]
    def is_wrist(k):
        return any(w in k for w in ("image2", "wrist", "eye_in_hand"))
    wrist = next((k for k in keys if is_wrist(k)), None)
    agent = next((k for k in keys if k != wrist), None)
    if agent is None or wrist is None:
        raise RuntimeError(f"could not resolve agent/wrist cameras from {keys}")
    return agent, wrist


def iter_carry_frames_dataset(ds, ep_indices, bounds, agent_key, wrist_key,
                              carry_lo=0.20, carry_hi=0.80, stride=10):
    """Yield (agentview, wrist, eef_state8) for carry-phase frames from a LeRobot
    dataset (HF). eef state is observation.state (already 8D eef in OSC format)."""
    for ei in ep_indices:
        f, t = bounds[ei]
        T = t - f
        for k in range(int(carry_lo * T), int(carry_hi * T), stride):
            row = ds[f + k]
            agent = _to_uint8_hwc(row[agent_key])
            wrist = _to_uint8_hwc(row[wrist_key])
            s8 = row["observation.state"].detach().cpu().numpy().astype(np.float32)
            yield agent, wrist, s8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--high-hdf5", default=None)
    ap.add_argument("--low-hdf5", default=None)
    ap.add_argument("--dataset-repo-id", default=None,
                    help="LeRobot dataset (HF) holding BOTH high+low episodes; "
                         "high=first half, low=second half. Alternative to --high/low-hdf5.")
    ap.add_argument("--dataset-root", default=None, help="local dataset root (default HF cache)")
    ap.add_argument("--dataset-revision", default="main",
                    help="HF dataset revision (default 'main' = freshest; avoids stale version-tagged cache)")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-eps", type=int, default=25, help="episodes per arc (high, low)")
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()
    if not args.dataset_repo_id and not (args.high_hdf5 and args.low_hdf5):
        ap.error("provide either --dataset-repo-id OR both --high-hdf5 and --low-hdf5")

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nLoading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy_path)

    n_layers = len(policy.model.vlm_with_expert.lm_expert.layers)
    print(f"lm_expert: {n_layers} layers")

    # Build per-condition frame iterators (HF dataset OR HDF5)
    if args.dataset_repo_id:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root,
                            revision=args.dataset_revision)
        n_ep = ds.num_episodes
        agent_key, wrist_key = detect_camera_keys(ds)
        bounds = episode_bounds(ds)
        # SELF-LABEL by measured EEF height: top-N carry-height eps = high, bottom-N = low.
        zmap = episode_carry_heights(ds, bounds)
        ordered = sorted(zmap, key=zmap.get)          # low → high
        k = min(args.n_eps, n_ep // 2)
        low_eps  = ordered[:k]
        high_eps = ordered[-k:]
        print(f"Dataset {args.dataset_repo_id}: {n_ep} eps, split by carry eef-z (top/bottom {k})")
        print(f"  HIGH eps {sorted(high_eps)}  z={np.mean([zmap[e] for e in high_eps])*100:.1f}cm")
        print(f"  LOW  eps {sorted(low_eps)}  z={np.mean([zmap[e] for e in low_eps])*100:.1f}cm")
        print(f"  cameras: camera1={agent_key}, camera2={wrist_key}")
        iters = [("high", lambda: iter_carry_frames_dataset(ds, high_eps, bounds, agent_key, wrist_key, stride=args.stride)),
                 ("low",  lambda: iter_carry_frames_dataset(ds, low_eps,  bounds, agent_key, wrist_key, stride=args.stride))]
    else:
        iters = [("high", lambda: iter_carry_frames(args.high_hdf5, args.n_eps, stride=args.stride)),
                 ("low",  lambda: iter_carry_frames(args.low_hdf5,  args.n_eps, stride=args.stride))]

    mean_acts = {}
    z_means = {}
    for cond, make_iter in iters:
        print(f"\nProcessing {cond}")
        handles, buffer = register_mlp_hooks(policy)
        nf = 0; zsum = 0.0
        for agent, wrist, s8 in tqdm(make_iter(), desc=f"  {cond}"):
            zsum += float(s8[2])
            with torch.no_grad():
                predict_action(observation=build_obs(agent, wrist, s8), policy=policy,
                               device=device, preprocessor=preprocessor,
                               postprocessor=postprocessor, use_amp=False, task=args.task)
            policy.reset(); nf += 1
        for h in handles:
            h.remove()
        mean_acts[cond] = [torch.stack(b).mean(0).numpy() if b else None for b in buffer]
        z_means[cond] = zsum / max(nf, 1)
        print(f"  {nf} carry frames, mean carry eef-z = {z_means[cond]*100:.1f} cm")

    if z_means["high"] < z_means["low"]:
        print("  [!] WARNING: 'high' group mean height < 'low' — episode split may be reversed; "
              "the CAA sign will be inverted (swap caa_high/caa_low or flip alpha).")

    caa = {str(i): torch.from_numpy(mean_acts["high"][i] - mean_acts["low"][i])
           for i in range(n_layers) if mean_acts["high"][i] is not None}
    norms = [caa[k].norm().item() for k in caa]
    print(f"\nCAA norms min/mean/max: {min(norms):.4f}/{np.mean(norms):.4f}/{max(norms):.4f}")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": caa, "policy_path": args.policy_path,
                "n_layers": n_layers, "format": "expert_mlp_output_high_minus_low"}, out)
    print(f"[✓] Saved → {out}")


if __name__ == "__main__":
    main()
