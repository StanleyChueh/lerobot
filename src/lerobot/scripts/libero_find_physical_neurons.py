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


def _capture_vlm_layer_means(policy, text_model):
    """Register pre-hooks on each VLM text_model down_proj; return (handles, sums, counts)."""
    n_layers = len(text_model.layers)
    layer_sums: list[np.ndarray | None] = [None] * n_layers
    layer_counts = [0] * n_layers
    handles = []
    for i in range(n_layers):
        def _hook(module, inputs, _i=i):
            arr = inputs[0].detach().float().cpu().mean(dim=(0, 1)).numpy()
            if layer_sums[_i] is None:
                layer_sums[_i] = arr.copy()
            else:
                layer_sums[_i] += arr
            layer_counts[_i] += 1
        handles.append(text_model.layers[i].mlp.down_proj.register_forward_pre_hook(_hook))
    return handles, layer_sums, layer_counts


def _finish_means(text_model, layer_sums, layer_counts, frame_count, desc):
    n_layers = len(text_model.layers)
    print(f"    {frame_count} carry frames, {n_layers} VLM layers captured  [{desc}]")
    means = []
    for i in range(n_layers):
        if layer_sums[i] is not None and layer_counts[i] > 0:
            means.append(layer_sums[i] / layer_counts[i])
        else:
            means.append(np.zeros(text_model.layers[i].mlp.down_proj.weight.shape[1], dtype=np.float32))
    return means


def collect_neuron_means(
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    hdf5_path: str,
    stride: int,
) -> list[np.ndarray]:
    """Capture VLM text_model pre-down_proj mean activations from HDF5 carry frames."""
    from lerobot.utils.control_utils import predict_action

    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    handles, layer_sums, layer_counts = _capture_vlm_layer_means(policy, text_model)

    frame_count = 0
    for img, state8 in tqdm(iter_carry_frames(hdf5_path, stride=stride),
                             desc=f"  {Path(hdf5_path).parent.name}"):
        obs = build_obs(img, state8)
        with torch.no_grad():
            predict_action(observation=obs, policy=policy, device=device,
                           preprocessor=preprocessor, postprocessor=postprocessor,
                           use_amp=False, task=task)
        policy.reset()
        frame_count += 1

    for h in handles:
        h.remove()
    return _finish_means(text_model, layer_sums, layer_counts, frame_count,
                         Path(hdf5_path).parent.name)


def collect_neuron_means_from_iter(
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    frame_iter,
    n_frames: int,
    desc: str = "frames",
) -> list[np.ndarray]:
    """Capture VLM text_model pre-down_proj mean activations from a (agent,wrist,front,state8) iterator.

    Used with HF LeRobot datasets; frame format matches iter_carry_frames_dataset()
    from libero_compute_caa_osc.py.
    """
    from lerobot.utils.control_utils import predict_action
    from lerobot.scripts.libero_compute_caa_osc import build_obs as build_obs_cams

    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    handles, layer_sums, layer_counts = _capture_vlm_layer_means(policy, text_model)

    frame_count = 0
    for agent, wrist, front, s8 in tqdm(frame_iter, total=n_frames, desc=f"  {desc}"):
        obs = build_obs_cams(agent, wrist, s8, front)
        with torch.no_grad():
            predict_action(observation=obs, policy=policy, device=device,
                           preprocessor=preprocessor, postprocessor=postprocessor,
                           use_amp=False, task=task)
        policy.reset()
        frame_count += 1

    for h in handles:
        h.remove()
    return _finish_means(text_model, layer_sums, layer_counts, frame_count, desc)


def main():
    ap = argparse.ArgumentParser(
        description="Find LIBERO physical height neurons by behavioral activation contrast."
    )
    ap.add_argument("--policy-path", default="ethanCSL/svla_franka_pick_n_place_vla_steering_libero")
    # HDF5 mode (original)
    ap.add_argument("--high-hdf5",   default=None, help="HDF5 file with high-carry episodes")
    ap.add_argument("--low-hdf5",    default=None, help="HDF5 file with low-carry episodes")
    # HF dataset mode (for models trained on HF LeRobot datasets)
    ap.add_argument("--dataset-repo-id",  default=None,
                    help="HF LeRobot dataset; episodes auto-split by carry EEF-z height")
    ap.add_argument("--dataset-revision", default="main",
                    help="HF dataset revision (default: 'main' — avoids stale cached tags)")
    ap.add_argument("--dataset-root",     default=None)
    ap.add_argument("--n-eps", type=int, default=25,
                    help="Episodes per condition (high, low) when using HF dataset")
    ap.add_argument("--task",        default="pick the akita black bowl between the plate and the ramekin "
                    "and place it on the plate")
    ap.add_argument("--top-n",       type=int, default=10, help="Top neurons per concept to output")
    ap.add_argument("--stride",      type=int, default=5,  help="Frame stride within carry phase")
    ap.add_argument("--output",      default="outputs/libero_physical_neurons.json")
    args = ap.parse_args()
    if not args.dataset_repo_id and not (args.high_hdf5 and args.low_hdf5):
        ap.error("provide either --dataset-repo-id OR both --high-hdf5 and --low-hdf5")

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
    if args.dataset_repo_id:
        import numpy as _np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.scripts.libero_compute_caa_osc import (
            detect_camera_keys, episode_bounds, episode_carry_heights,
            iter_carry_frames_dataset, count_carry_frames_dataset,
        )
        print(f"\nLoading HF dataset: {args.dataset_repo_id}  revision={args.dataset_revision}")
        ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root,
                            revision=args.dataset_revision)
        agent_key, wrist_key, front_key = detect_camera_keys(ds)
        print(f"  cameras: agent={agent_key}, wrist={wrist_key}, front={front_key}")
        bounds = episode_bounds(ds)
        zmap   = episode_carry_heights(ds, bounds)
        ordered = sorted(zmap, key=zmap.get)
        k = min(args.n_eps, len(ordered) // 2)
        if k == 0:
            raise RuntimeError(f"Not enough episodes ({len(ordered)}) to split high/low.")
        low_eps, high_eps = ordered[:k], ordered[-k:]
        z_high = _np.mean([zmap[e] for e in high_eps]) * 100
        z_low  = _np.mean([zmap[e] for e in low_eps])  * 100
        print(f"  HIGH: {k} eps, mean z={z_high:.1f}cm  |  LOW: {k} eps, mean z={z_low:.1f}cm  "
              f"(Δ={z_high-z_low:.1f}cm)")
        if z_high - z_low < 3.0:
            print("  WARNING: very small height gap — vectors may be weak!")
        n_high = count_carry_frames_dataset(high_eps, bounds, stride=args.stride)
        n_low  = count_carry_frames_dataset(low_eps,  bounds, stride=args.stride)
        print(f"\nCollecting carry-phase activations — HIGH episodes ({n_high} frames) ...")
        high_means = collect_neuron_means_from_iter(
            policy, preprocessor, postprocessor, device, args.task,
            iter_carry_frames_dataset(ds, high_eps, bounds, agent_key, wrist_key,
                                      front_key, stride=args.stride),
            n_high, desc="HIGH")
        print(f"\nCollecting carry-phase activations — LOW episodes ({n_low} frames) ...")
        low_means = collect_neuron_means_from_iter(
            policy, preprocessor, postprocessor, device, args.task,
            iter_carry_frames_dataset(ds, low_eps, bounds, agent_key, wrist_key,
                                      front_key, stride=args.stride),
            n_low, desc="LOW")
    else:
        print("\nCollecting carry-phase activations — HIGH episodes ...")
        high_means = collect_neuron_means(
            policy, preprocessor, postprocessor, device, args.task, args.high_hdf5, args.stride)
        print("\nCollecting carry-phase activations — LOW episodes ...")
        low_means = collect_neuron_means(
            policy, preprocessor, postprocessor, device, args.task, args.low_hdf5, args.stride)

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

    # ── Save per-layer contrast VECTORS (for sparse-CAA steering) ─────────────
    # vectors[layer] = mean_high − mean_low  (dim = intermediate_dim, 2560).
    # This is the VLM-space CAA vector; the eval can ADD ±alpha × (top-k masked)
    # to down_proj input for principled, per-neuron-weighted physical steering.
    vec_out = {}
    for layer_idx in range(n_layers):
        diff = (high_means[layer_idx].astype(np.float32)
                - low_means[layer_idx].astype(np.float32))
        vec_out[str(layer_idx)] = torch.from_numpy(diff)
    vec_path = out.with_name(out.stem + "_vectors.pt")
    torch.save({"vectors": vec_out, "n_layers": n_layers,
                "module": "vlm.text_model.layers[i].mlp.down_proj(input)",
                "direction": "high_minus_low"}, vec_path)
    print(f"[✓] Saved contrast vectors → {vec_path}")

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
