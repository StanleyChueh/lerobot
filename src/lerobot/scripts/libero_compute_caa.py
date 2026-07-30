#!/usr/bin/env python3
"""
Compute CAA (Contrastive Activation Addition) steering vectors for SmolVLA
fine-tuned on LIBERO height demos.

  caa_vector[layer] = mean(lm_expert MLP output over HIGH carry frames)
                    - mean(lm_expert MLP output over LOW carry frames)

The carry phase = middle 20–80% of each episode (when the bowl is being transported).

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_compute_caa.py \
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero \
    --high-hdf5 /home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5 \
    --low-hdf5  /home/bruce/datasets/libero_height_demos/libero_spatial/low/task_00.hdf5 \
    --task "Pick up the black bowl and place it on the plate." \
    --output outputs/libero_caa_vectors.pt \
    --n-eps 30
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── Franka DH forward kinematics (pure numpy, no deps) ────────────────────────

_PI = np.pi
# [a (m), d (m), alpha (rad)]  — modified DH, Franka Panda Research 3
_FRANKA_DH = [
    [0,        0.333,   0      ],
    [0,        0,      -_PI/2  ],
    [0,        0.316,   _PI/2  ],
    [0.0825,   0,       _PI/2  ],
    [-0.0825,  0.384,  -_PI/2  ],
    [0,        0,       _PI/2  ],
    [0.088,    0.107,   _PI/2  ],
]
_FRANKA_EEF_Z = 0.1034  # flange → grip site offset along z


def franka_fk_z(q: np.ndarray) -> float:
    """Return EEF z-height (metres) for 7-DoF Franka joint angles (radians)."""
    T = np.eye(4)
    for i, (a, d, alpha) in enumerate(_FRANKA_DH):
        theta = float(q[i])
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T = T @ np.array([
            [ct, -st*ca,  st*sa,  a*ct],
            [st,  ct*ca, -ct*sa,  a*st],
            [0,   sa,     ca,     d   ],
            [0,   0,      0,      1   ],
        ])
    # EEF offset along joint-7 z-axis
    return float(T[2, 3] + T[2, 2] * _FRANKA_EEF_Z)


def validate_fk(hdf5_path: str, n: int = 20):
    """Validate calibrated FK (per-episode z-offset) against stored eef_pos."""
    with h5py.File(hdf5_path, "r") as f:
        g = f["ep_000"]
        jp = g["joint_pos"][()]
        ep = g["eef_pos"][()]
    offset = ep[0, 2] - franka_fk_z(jp[0])
    errs = [abs(franka_fk_z(jp[i]) + offset - ep[i, 2]) * 100 for i in range(min(n, len(jp)))]
    print(f"  FK calibration offset: {offset*100:.1f} cm  | "
          f"residual mean Δz={np.mean(errs):.2f} cm  max={np.max(errs):.2f} cm")


# ── Observation builder ────────────────────────────────────────────────────────

_EMPTY_256 = np.zeros((256, 256, 3), dtype=np.uint8)
_EMPTY_480 = np.zeros((480, 640, 3), dtype=np.uint8)


def build_obs(img_hwc: np.ndarray, state8: np.ndarray) -> dict:
    """Build the observation dict that matches what the trained model expects.
    state8: (8,) = [panda_joint_1..7, gripper] — normalizer stats are 8D.
    """
    return {
        "observation.images.agentview":   img_hwc,         # renamed → camera1
        "observation.images.camera2":     _EMPTY_256,
        "observation.images.camera3":     _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.images.empty_camera_1": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


# ── Hook infrastructure ────────────────────────────────────────────────────────

def register_mlp_hooks(policy) -> tuple[list, list]:
    """
    Register forward hooks on every lm_expert MLP layer.
    Returns (handles, buffer) where buffer[i] accumulates (hidden_dim,) tensors.
    """
    layers = policy.model.vlm_with_expert.lm_expert.layers
    buffer: list[list] = [[] for _ in range(len(layers))]
    handles = []

    for i, layer in enumerate(layers):
        def _make_hook(idx):
            def hook(module, inputs, output):
                # output: (batch=1, seq_len, hidden_dim) — average over tokens
                h = output[0] if isinstance(output, tuple) else output
                buffer[idx].append(h.detach().mean(dim=1).squeeze(0).cpu().float())
            return hook
        handles.append(layer.mlp.register_forward_hook(_make_hook(i)))

    return handles, buffer


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── Episode loader ─────────────────────────────────────────────────────────────

def iter_carry_frames(hdf5_path: str, n_eps: int | None, carry_lo=0.20, carry_hi=0.80, stride: int = 10):
    """
    Yield (img_hwc, state8) for carry-phase frames across all episodes.
    state8 = [joint_pos(7), gripper(1)] — 8D to match normalizer stats.
    Carry phase = [20%, 80%] of episode length.
    stride: sample every N-th frame (default 10). CAA vectors are stable across
    nearby frames so dense sampling is unnecessary and very slow (~1s/frame).
    """
    with h5py.File(hdf5_path, "r") as f:
        ep_keys = sorted(k for k in f.keys() if k.startswith("ep_"))
        if n_eps is not None:
            ep_keys = ep_keys[:n_eps]
        for key in ep_keys:
            g = f[key]
            imgs    = g["agentview_image"][()]  # (T, 256, 256, 3)
            joints  = g["joint_pos"][()]         # (T, 7)
            gripper = g["actions"][:, 6:7]       # (T, 1) gripper cmd
            T = len(imgs)
            lo = int(carry_lo * T)
            hi = int(carry_hi * T)
            for t in range(lo, hi, stride):
                state8 = np.concatenate([joints[t], gripper[t]], axis=0)  # (8,)
                yield imgs[t], state8


# ── Main ──────────────────────────────────────────────────────────────────────

def compute_caa(args):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # FK validation
    print("Validating Franka FK ...")
    validate_fk(args.high_hdf5)

    # Load policy via from_pretrained (make_policy requires dataset meta which we don't have)
    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy_cfg = policy.config
    policy_cfg.device = str(device)
    policy.eval()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
    )

    n_layers = len(policy.model.vlm_with_expert.lm_expert.layers)
    print(f"lm_expert: {n_layers} layers")

    # ── Collect activations per condition ──────────────────────────────────────
    mean_acts: dict[str, list[np.ndarray]] = {}

    for condition, hdf5_path in [("high", args.high_hdf5), ("low", args.low_hdf5)]:
        print(f"\nProcessing {condition} episodes: {hdf5_path}")
        handles, buffer = register_mlp_hooks(policy)

        frame_count = 0
        for img, state8 in tqdm(
            iter_carry_frames(hdf5_path, args.n_eps, stride=args.stride),
            desc=f"  {condition} carry frames",
        ):
            obs = build_obs(img, state8)
            with torch.no_grad():
                predict_action(
                    observation=obs,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=False,
                    task=args.task,
                )
            frame_count += 1
            # Reset policy hidden state to avoid temporal contamination
            policy.reset()

        remove_hooks(handles)

        # Mean per layer
        layer_means = []
        for layer_buf in buffer:
            if layer_buf:
                layer_means.append(torch.stack(layer_buf).mean(0).numpy())
            else:
                n_layers_dim = policy.model.vlm_with_expert.lm_expert.layers[0].mlp.down_proj.weight.shape[0]
                layer_means.append(np.zeros(n_layers_dim, dtype=np.float32))
        mean_acts[condition] = layer_means
        print(f"  Processed {frame_count} carry frames, {n_layers} layers")

    # ── Compute CAA vectors ────────────────────────────────────────────────────
    caa_vectors = {}
    for i in range(n_layers):
        high_vec = mean_acts["high"][i]
        low_vec  = mean_acts["low"][i]
        caa_vectors[str(i)] = torch.from_numpy(high_vec - low_vec)

    # Norm summary
    norms = [caa_vectors[str(i)].norm().item() for i in range(n_layers)]
    print(f"\nCAA vector norms (min/mean/max): "
          f"{min(norms):.4f} / {np.mean(norms):.4f} / {max(norms):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vectors":       caa_vectors,
        "policy_path":   args.policy_path,
        "high_hdf5":     args.high_hdf5,
        "low_hdf5":      args.low_hdf5,
        "n_layers":      n_layers,
        "n_high_eps":    args.n_eps,
        "n_low_eps":     args.n_eps,
        "carry_lo":      0.20,
        "carry_hi":      0.80,
    }, out)
    print(f"\n[✓] Saved CAA vectors → {out}")
    print(f"    {n_layers} layers, format: {{\"vectors\": {{\"0\": tensor, ...}}}}")
    print(f"\nUse with libero_eval_steering.py:  --steering caa --caa-path {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", default="ethanCSL/svla_franka_pick_n_place_vla_steering_libero")
    ap.add_argument("--high-hdf5",   required=True)
    ap.add_argument("--low-hdf5",    required=True)
    ap.add_argument("--task",        default="Pick up the black bowl and place it on the plate.")
    ap.add_argument("--output",      default="outputs/libero_caa_vectors.pt")
    ap.add_argument("--n-eps",       type=int, default=None, help="Episodes per condition (default: all)")
    ap.add_argument("--stride",      type=int, default=10,
                    help="Sample every N-th carry-phase frame (default 10). "
                         "Reduces ~5400 frames → ~540, cutting runtime from ~90 min to ~9 min.")
    args = ap.parse_args()
    compute_caa(args)
