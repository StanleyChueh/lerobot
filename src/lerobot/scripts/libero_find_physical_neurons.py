#!/usr/bin/env python3
"""
Find LIBERO physical height neurons (behavioral contrast method).

Unlike keyword neurons (selected by semantic token prediction),
physical neurons are selected by ACTUAL activation difference:

    score[layer, neuron] = mean_activation(high_carry_frames)[neuron]
                         - mean_activation(low_carry_frames)[neuron]

Top +score neurons = physical "high" neurons (more active when carrying high)
Top -score neurons = physical "low"  neurons (more active when carrying low)

Target: VLM text_model.layers[i].mlp.down_proj  PRE-activation
        (intermediate_dim=2560 — same neuron space as keyword neurons)
Steering: same SET mechanism (activation = alpha) as keyword neurons.

Two-step workflow
─────────────────
Step 1 — Find neurons (this script):
  conda run -n lerobot python src/lerobot/scripts/libero_find_physical_neurons.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero \\
    --high-hdf5 /home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5 \\
    --low-hdf5  /home/bruce/datasets/libero_height_demos/libero_spatial/low/task_00.hdf5 \\
    --task "Pick up the black bowl and place it on the plate." \\
    --top-n 10 \\
    --output outputs/libero_physical_neurons.json

Step 2 — Evaluate (same as keyword method, just different --neurons-json):
  conda run -n lerobot python src/lerobot/scripts/libero_eval_steering.py \\
    --neurons-json outputs/libero_physical_neurons.json \\
    --steering-mode set --keyword-alpha 6.0 \\
    --conditions none keyword_high keyword_low \\
    ...
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_EMPTY_256 = np.zeros((256, 256, 3), dtype=np.uint8)
_EMPTY_480 = np.zeros((480, 640, 3), dtype=np.uint8)


def build_obs(img_hwc: np.ndarray, state8: np.ndarray) -> dict:
    """Build observation dict matching trained model input format."""
    return {
        "observation.images.agentview":      img_hwc,
        "observation.images.camera2":        _EMPTY_256,
        "observation.images.camera3":        _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.images.empty_camera_1": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


def iter_carry_frames(hdf5_path: str, carry_lo: float = 0.20, carry_hi: float = 0.80, stride: int = 5):
    """Yield (img_hwc, state8) for carry-phase frames across all episodes.

    stride: sample every N-th frame within each episode's carry phase
    (stride=5 keeps ~12 frames/ep for a 300-step episode; faster than every frame).
    """
    with h5py.File(hdf5_path, "r") as f:
        ep_keys = sorted(k for k in f.keys() if k.startswith("ep_"))
        for key in ep_keys:
            g = f[key]
            imgs    = g["agentview_image"][()]   # (T, 256, 256, 3)
            joints  = g["joint_pos"][()]          # (T, 7) radians
            gripper = g["actions"][:, 6:7]        # (T, 1) gripper cmd
            T = len(imgs)
            lo = int(carry_lo * T)
            hi = int(carry_hi * T)
            for t in range(lo, hi, stride):
                state8 = np.concatenate([joints[t], gripper[t]], axis=0)  # (8,)
                yield imgs[t], state8


def collect_neuron_means(
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    hdf5_path: str,
    stride: int,
) -> list[np.ndarray]:
    """
    Run inference on carry-phase frames and capture VLM text_model
    pre-down_proj activations. Returns mean across all frames:
    list of n_layers arrays, each shape (intermediate_dim=2560,).
    """
    from lerobot.utils.control_utils import predict_action

    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    n_layers = len(text_model.layers)

    # Accumulate sum and count per layer for computing mean
    layer_sums: list[np.ndarray | None] = [None] * n_layers
    layer_counts = [0] * n_layers

    handles = []
    for i in range(n_layers):
        def _hook(module, inputs, _i=i):
            act = inputs[0]  # (batch, seq_len, intermediate_dim)
            arr = act.detach().float().cpu().mean(dim=(0, 1)).numpy()  # (intermediate_dim,)
            if layer_sums[_i] is None:
                layer_sums[_i] = arr.copy()
            else:
                layer_sums[_i] += arr
            layer_counts[_i] += 1
        handles.append(text_model.layers[i].mlp.down_proj.register_forward_pre_hook(_hook))

    frame_count = 0
    for img, state8 in tqdm(iter_carry_frames(hdf5_path, stride=stride),
                             desc=f"  {Path(hdf5_path).parent.name}"):
        obs = build_obs(img, state8)
        with torch.no_grad():
            predict_action(
                observation=obs,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
                task=task,
            )
        policy.reset()
        frame_count += 1

    for h in handles:
        h.remove()

    print(f"    {frame_count} carry frames, {n_layers} VLM layers captured")

    means = []
    for i in range(n_layers):
        if layer_sums[i] is not None and layer_counts[i] > 0:
            means.append(layer_sums[i] / layer_counts[i])
        else:
            d_ff = text_model.layers[i].mlp.down_proj.weight.shape[1]
            means.append(np.zeros(d_ff, dtype=np.float32))

    return means


def main():
    ap = argparse.ArgumentParser(
        description="Find LIBERO physical height neurons by behavioral activation contrast."
    )
    ap.add_argument("--policy-path", default="ethanCSL/svla_franka_pick_n_place_vla_steering_libero")
    ap.add_argument("--high-hdf5",   required=True, help="HDF5 file with high-carry episodes")
    ap.add_argument("--low-hdf5",    required=True, help="HDF5 file with low-carry episodes")
    ap.add_argument("--task",        default="Pick up the black bowl and place it on the plate.")
    ap.add_argument("--top-n",       type=int, default=10, help="Top neurons per concept to output")
    ap.add_argument("--stride",      type=int, default=5,  help="Frame stride within carry phase")
    ap.add_argument("--output",      default="outputs/libero_physical_neurons.json")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy_cfg = policy.config
    policy_cfg.device = str(device)
    policy.eval().to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
    )

    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    n_layers = len(text_model.layers)
    d_ff = text_model.layers[0].mlp.down_proj.weight.shape[1]
    print(f"VLM text_model: {n_layers} layers, intermediate_dim={d_ff}")

    # ── Collect mean activations per condition ────────────────────────────────
    print("\nCollecting carry-phase activations — HIGH episodes ...")
    high_means = collect_neuron_means(
        policy, preprocessor, postprocessor, device, args.task, args.high_hdf5, args.stride
    )

    print("\nCollecting carry-phase activations — LOW episodes ...")
    low_means = collect_neuron_means(
        policy, preprocessor, postprocessor, device, args.task, args.low_hdf5, args.stride
    )

    # ── Score: mean_high − mean_low per neuron ────────────────────────────────
    all_neurons = []
    for layer_idx in range(n_layers):
        diff = high_means[layer_idx].astype(np.float64) - low_means[layer_idx].astype(np.float64)
        for neuron_idx, score in enumerate(diff):
            all_neurons.append({"layer": layer_idx, "neuron": neuron_idx, "score": float(score)})

    top_high = sorted(all_neurons, key=lambda x: x["score"], reverse=True)[:args.top_n]
    top_low  = sorted(all_neurons, key=lambda x: x["score"])[:args.top_n]

    print("\n" + "="*65)
    print("TOP PHYSICAL 'HIGH' NEURONS  (mean_high − mean_low, most positive)")
    print("="*65)
    for n in top_high:
        print(f"  Layer {n['layer']:>2} | Neuron {n['neuron']:>5} | Score: {n['score']:+.6f}")

    print("\n" + "="*65)
    print("TOP PHYSICAL 'LOW' NEURONS   (mean_low − mean_high, most positive)")
    print("="*65)
    for n in top_low:
        print(f"  Layer {n['layer']:>2} | Neuron {n['neuron']:>5} | Score: {abs(n['score']):+.6f}")

    # ── Save JSON (same format as keyword neurons) ────────────────────────────
    neurons_out: dict[str, dict[str, list[int]]] = {"high": {}, "low": {}}
    for n in top_high:
        neurons_out["high"].setdefault(str(n["layer"]), []).append(n["neuron"])
    for n in top_low:
        neurons_out["low"].setdefault(str(n["layer"]), []).append(n["neuron"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(neurons_out, f, indent=2)

    print(f"\n[✓] Saved → {out}")
    for concept, lm in neurons_out.items():
        total = sum(len(v) for v in lm.values())
        print(f"    {concept}: {len(lm)} layers, {total} neurons")
    print(f"\nEvaluate with (same steering mechanism as keyword method):")
    print(f"  libero_eval_steering.py \\")
    print(f"    --neurons-json {out} \\")
    print(f"    --steering-mode set --keyword-alpha 6.0 \\")
    print(f"    --conditions none keyword_high keyword_low")


if __name__ == "__main__":
    main()
