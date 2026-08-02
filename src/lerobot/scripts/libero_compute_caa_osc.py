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
    """Yield (agentview, wrist, eef_state8) for carry-phase frames."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--high-hdf5", required=True)
    ap.add_argument("--low-hdf5", required=True)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-eps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

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

    mean_acts = {}
    for cond, hdf5_path in [("high", args.high_hdf5), ("low", args.low_hdf5)]:
        print(f"\nProcessing {cond}: {hdf5_path}")
        handles, buffer = register_mlp_hooks(policy)
        nf = 0
        for agent, wrist, s8 in tqdm(iter_carry_frames(hdf5_path, args.n_eps, stride=args.stride),
                                     desc=f"  {cond}"):
            with torch.no_grad():
                predict_action(observation=build_obs(agent, wrist, s8), policy=policy,
                               device=device, preprocessor=preprocessor,
                               postprocessor=postprocessor, use_amp=False, task=args.task)
            policy.reset(); nf += 1
        for h in handles:
            h.remove()
        mean_acts[cond] = [torch.stack(b).mean(0).numpy() if b else None for b in buffer]
        print(f"  {nf} carry frames")

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
