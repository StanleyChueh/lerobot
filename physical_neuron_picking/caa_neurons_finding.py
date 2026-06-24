# Copyright 2026
# CAA dense layer-group steering vector extraction for SmolVLA / LeRobot.
#
# This script extracts dense CAA steering vectors:
#
#   v_l = mean_over_samples(activation_l(high_prompt) - activation_l(low_prompt))
#
# Supported layer groups for SmolVLA lm_expert:
#   early = first half of layers
#   late  = second half of layers
#   full  = all layers
#   all   = one extraction pass, then output early / late / full .pt files
#
# For a 16-layer SmolVLA lm_expert:
#   early = 0-7
#   late  = 8-15
#   full  = 0-15

'''
python src/lerobot/scripts/caa_neurons_finding.py \
  --policy-path ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
  --dataset-repo-id ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
  --frames-per-episode 3 \
  --max-episodes 60 \
  --layer-group all \
  --output-dir /home/bruce/CSL/lerobot_nn
'''
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


# ==============================================================================
# 1. Dataset utilities
# ==============================================================================

def get_episode_indices(dataset):
    for obj in [dataset, dataset.meta]:
        if hasattr(obj, "episode_data_index"):
            return getattr(obj, "episode_data_index")

    if hasattr(dataset.meta, "episodes"):
        return dataset.meta.episodes

    return None


def _to_int(x):
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def get_episode_frame_range(ep_index, ep_idx):
    """Robustly parse LeRobot episode frame ranges across schema formats."""
    if hasattr(ep_index, "column_names"):
        columns = set(ep_index.column_names)

        if "dataset_from_index" in columns and "dataset_to_index" in columns:
            row = ep_index[ep_idx]
            return _to_int(row["dataset_from_index"]), _to_int(row["dataset_to_index"])

        if "from" in columns and "to" in columns:
            row = ep_index[ep_idx]
            return _to_int(row["from"]), _to_int(row["to"])

        if "length" in columns:
            start_frame = 0
            for i in range(ep_idx):
                start_frame += _to_int(ep_index[i]["length"])
            end_frame = start_frame + _to_int(ep_index[ep_idx]["length"])
            return start_frame, end_frame

    if isinstance(ep_index, dict) and "from" in ep_index and "to" in ep_index:
        return _to_int(ep_index["from"][ep_idx]), _to_int(ep_index["to"][ep_idx])

    if isinstance(ep_index, list) and isinstance(ep_index[ep_idx], dict):
        ep = ep_index[ep_idx]

        if "from" in ep and "to" in ep:
            return _to_int(ep["from"]), _to_int(ep["to"])

        if "start" in ep and "end" in ep:
            return _to_int(ep["start"]), _to_int(ep["end"])

        if "dataset_from_index" in ep and "dataset_to_index" in ep:
            return _to_int(ep["dataset_from_index"]), _to_int(ep["dataset_to_index"])

    raise TypeError("Unrecognized episode index layout.")


def sample_frames_across_episodes(
    dataset,
    frames_per_episode=3,
    max_episodes=None,
    frame_start_frac=0.20,
    frame_end_frac=0.80,
):
    """
    Sample frames across episodes.

    Default:
      sample middle 20%-80% of each episode.

    This avoids using only the first N frames, which would overrepresent
    static / initial grasping frames.
    """
    ep_index = get_episode_indices(dataset)
    if ep_index is None:
        raise AttributeError("Could not recover episode frame ranges from dataset.")

    num_episodes = dataset.num_episodes
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    sampled = []

    for ep_idx in range(num_episodes):
        start, end = get_episode_frame_range(ep_index, ep_idx)
        length = end - start

        if length <= 0:
            continue

        lo = start + int(frame_start_frac * length)
        hi = start + int(frame_end_frac * length)

        if hi <= lo:
            lo, hi = start, end

        frame_ids = np.linspace(lo, max(lo, hi - 1), frames_per_episode)
        frame_ids = [int(x) for x in frame_ids]

        for frame_idx in frame_ids:
            sampled.append((ep_idx, frame_idx))

    return sampled


def build_observation_frame(sample_frame, rename_map):
    """
    Convert a LeRobot dataset frame into predict_action-compatible observation.
    """
    observation_frame = {}

    for key, value in sample_frame.items():
        if isinstance(value, str):
            continue

        target_key = rename_map.get(key, key)

        if isinstance(value, torch.Tensor):
            if "image" in key:
                img = value.detach().cpu().numpy()

                # CHW -> HWC
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))

                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = img * 255.0
                    img = np.clip(img, 0, 255).astype(np.uint8)

                observation_frame[target_key] = np.ascontiguousarray(img)
            else:
                observation_frame[target_key] = value.detach().cpu().numpy()
        else:
            observation_frame[target_key] = value

    return observation_frame


# ==============================================================================
# 2. Model utilities
# ==============================================================================

def get_lm_expert_layers(policy):
    try:
        return policy.model.vlm_with_expert.lm_expert.layers
    except Exception as exc:
        raise AttributeError(
            "Could not find policy.model.vlm_with_expert.lm_expert.layers. "
            "This script assumes SmolVLA with an lm_expert pathway."
        ) from exc


def patch_dataset_meta_for_policy(dataset, rename_map):
    patched_meta = copy.deepcopy(dataset.meta)

    for old_key, new_key in rename_map.items():
        if old_key in patched_meta.features:
            patched_meta.features[new_key] = patched_meta.features.pop(old_key)

        if hasattr(patched_meta, "stats") and patched_meta.stats and old_key in patched_meta.stats:
            patched_meta.stats[new_key] = patched_meta.stats.pop(old_key)

    return patched_meta


def select_layer_indices(num_layers, layer_group):
    """
    Select lm_expert layers.

    For 16 layers:
      early = 0-7
      late  = 8-15
      full  = 0-15
    """
    if layer_group == "full":
        return list(range(num_layers))

    split = num_layers // 2

    if layer_group == "early":
        return list(range(0, split))

    if layer_group == "late":
        return list(range(split, num_layers))

    raise ValueError(f"Unknown layer_group: {layer_group}")


def layer_range_tag(layer_indices):
    if not layer_indices:
        return "Lnone"

    if len(layer_indices) == 1:
        return f"L{layer_indices[0]}"

    return f"L{min(layer_indices)}_{max(layer_indices)}"


def default_output_name(layer_group, layer_indices, max_episodes, frames_per_episode):
    tag = layer_range_tag(layer_indices)
    return f"caa_dense_{layer_group}_{tag}_high_low_{max_episodes}eps_f{frames_per_episode}.pt"


# ==============================================================================
# 3. CAA extraction
# ==============================================================================

@torch.no_grad()
def collect_dense_caa_diffs(
    policy,
    preprocessor,
    postprocessor,
    dataset,
    prompt_pairs,
    rename_map,
    capture_layers,
    frames_per_episode=3,
    max_episodes=60,
    device="cuda",
    frame_start_frac=0.20,
    frame_end_frac=0.80,
):
    """
    Run high/low prompt pairs and collect activation differences.

    Returns:
      diff_accumulated[layer_idx] = list of tensors [1, hidden_dim]
    """
    layers = get_lm_expert_layers(policy)
    num_layers = len(layers)

    print(f"[*] Found {num_layers} lm_expert layers.")
    print(f"[*] Capture layers: {capture_layers}")
    print("[*] Registering CAA capture hooks on selected layer.mlp modules...")

    captured = {layer_idx: None for layer_idx in capture_layers}
    diff_accumulated = {layer_idx: [] for layer_idx in capture_layers}

    def make_hook(layer_idx):
        def hook_fn(module, inputs, outputs):
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

            # Expected shape: [B, seq_len, hidden_dim]
            if hidden_states.ndim == 3:
                layer_vec = hidden_states.detach().mean(dim=1).float().cpu()
            elif hidden_states.ndim == 2:
                layer_vec = hidden_states.detach().float().cpu()
            else:
                raise RuntimeError(
                    f"Unexpected activation shape at layer {layer_idx}: {tuple(hidden_states.shape)}"
                )

            captured[layer_idx] = layer_vec

        return hook_fn

    handles = []
    for layer_idx in capture_layers:
        layer = layers[layer_idx]

        if not hasattr(layer, "mlp"):
            raise AttributeError(f"Layer {layer_idx} has no .mlp module.")

        handles.append(layer.mlp.register_forward_hook(make_hook(layer_idx)))

    sampled_frames = sample_frames_across_episodes(
        dataset,
        frames_per_episode=frames_per_episode,
        max_episodes=max_episodes,
        frame_start_frac=frame_start_frac,
        frame_end_frac=frame_end_frac,
    )

    print(f"[*] Sampled {len(sampled_frames)} frames across episodes.")
    print(f"[*] Prompt pairs: {len(prompt_pairs)}")
    print("[*] Starting CAA extraction: high prompt - low prompt")

    try:
        for sample_idx, (ep_idx, frame_idx) in enumerate(tqdm(sampled_frames, desc="CAA samples")):
            sample_frame = dataset[frame_idx]
            observation_frame = build_observation_frame(sample_frame, rename_map)

            for pair_idx, (prompt_high, prompt_low) in enumerate(prompt_pairs):
                # ------------------------------------------------------
                # High prompt forward pass
                # ------------------------------------------------------
                for layer_idx in capture_layers:
                    captured[layer_idx] = None

                policy.reset()

                _ = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=getattr(policy.config, "use_amp", False),
                    device=device,
                    task=prompt_high,
                )

                high_cache = {}
                missing_high = []

                for layer_idx in capture_layers:
                    if captured[layer_idx] is None:
                        missing_high.append(layer_idx)
                    else:
                        high_cache[layer_idx] = captured[layer_idx].clone()

                if missing_high:
                    print(f"[!] Missing high activations at layers: {missing_high}")
                    continue

                # ------------------------------------------------------
                # Low prompt forward pass
                # ------------------------------------------------------
                for layer_idx in capture_layers:
                    captured[layer_idx] = None

                policy.reset()

                _ = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=getattr(policy.config, "use_amp", False),
                    device=device,
                    task=prompt_low,
                )

                missing_low = []

                for layer_idx in capture_layers:
                    if captured[layer_idx] is None:
                        missing_low.append(layer_idx)

                if missing_low:
                    print(f"[!] Missing low activations at layers: {missing_low}")
                    continue

                # ------------------------------------------------------
                # Accumulate high - low
                # ------------------------------------------------------
                for layer_idx in capture_layers:
                    diff = high_cache[layer_idx] - captured[layer_idx]
                    diff_accumulated[layer_idx].append(diff)

                if sample_idx == 0 and pair_idx == 0:
                    print("[DIAG] Example captured activation shapes:")
                    for layer_idx in capture_layers:
                        print(f"  Layer {layer_idx:02d}: {tuple(high_cache[layer_idx].shape)}")

    finally:
        for handle in handles:
            handle.remove()
        print("[*] Removed all CAA capture hooks.")

    return diff_accumulated


def build_vectors_from_diffs(diff_accumulated, selected_layers):
    vectors = {}
    raw_vectors = {}
    layer_stats = {}

    for layer_idx in selected_layers:
        diffs = diff_accumulated.get(layer_idx, [])

        if not diffs:
            print(f"[!] No diffs collected for layer {layer_idx}; skipping.")
            continue

        mean_diff = torch.cat(diffs, dim=0).mean(dim=0, keepdim=True)
        norm = torch.linalg.vector_norm(mean_diff, ord=2, dim=-1, keepdim=True).clamp_min(1e-8)
        v_steer = mean_diff / norm

        vectors[layer_idx] = v_steer.cpu()
        raw_vectors[layer_idx] = mean_diff.cpu()

        layer_stats[layer_idx] = {
            "num_diffs": len(diffs),
            "raw_norm": float(norm.item()),
            "mean_abs": float(mean_diff.abs().mean().item()),
            "max_abs": float(mean_diff.abs().max().item()),
            "shape": list(v_steer.shape),
        }

    return vectors, raw_vectors, layer_stats


def save_caa_file(
    output_path,
    layer_group,
    selected_layers,
    vectors,
    raw_vectors,
    layer_stats,
    prompt_pairs,
    frames_per_episode,
    max_episodes,
    frame_start_frac,
    frame_end_frac,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_obj = {
        "method": "dense_caa_layer_group",
        "module_path": "policy.model.vlm_with_expert.lm_expert.layers[*].mlp",
        "direction": "high_minus_low",
        "layer_group": layer_group,
        "selected_layers": selected_layers,
        "vectors": vectors,
        "raw_vectors": raw_vectors,
        "layer_stats": layer_stats,
        "prompt_pairs": prompt_pairs,
        "frames_per_episode": frames_per_episode,
        "max_episodes": max_episodes,
        "frame_start_frac": frame_start_frac,
        "frame_end_frac": frame_end_frac,
    }

    torch.save(save_obj, output_path)

    json_path = output_path.with_suffix(".summary.json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": save_obj["method"],
                "module_path": save_obj["module_path"],
                "direction": save_obj["direction"],
                "layer_group": layer_group,
                "selected_layers": selected_layers,
                "num_layers_saved": len(vectors),
                "layer_stats": layer_stats,
                "prompt_pairs": prompt_pairs,
                "frames_per_episode": frames_per_episode,
                "max_episodes": max_episodes,
                "frame_start_frac": frame_start_frac,
                "frame_end_frac": frame_end_frac,
            },
            f,
            indent=2,
        )

    print(f"\n[✓] Saved CAA vectors: {output_path}")
    print(f"[✓] Saved summary:     {json_path}")
    print(f"[✓] Layers saved:     {sorted(vectors.keys())}")


# ==============================================================================
# 4. Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy-path",
        type=str,
        default="ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2",
    )

    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=60)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--layer-group",
        type=str,
        default="full",
        choices=["early", "late", "full", "all"],
        help=(
            "Which lm_expert layer group to save. "
            "For 16 layers: early=0-7, late=8-15, full=0-15. "
            "Use all to output early, late, and full from one extraction pass."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .pt path for a single layer group. "
            "Ignored when --layer-group all."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory used for default output names and for --layer-group all.",
    )

    parser.add_argument(
        "--frame-start-frac",
        type=float,
        default=0.20,
        help="Start fraction of each episode used for frame sampling.",
    )
    parser.add_argument(
        "--frame-end-frac",
        type=float,
        default=0.80,
        help="End fraction of each episode used for frame sampling.",
    )

    args = parser.parse_args()

    if args.device is None:
        device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[*] Device: {device}")

    rename_map = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
        "observation.images.wrist": "observation.images.camera3",
    }

    # Minimal-pair prompts.
    # Keep the task and object constant; only change high/low wording.
    prompt_pairs = [
        (
            "Put the red cube in the box using a high trajectory.",
            "Put the red cube in the box using a low trajectory.",
        ),
        (
            "Move the red cube to the box with high clearance.",
            "Move the red cube to the box with low clearance.",
        ),
        (
            "Transport the red cube high above the table.",
            "Transport the red cube low above the table.",
        ),
        (
            "Use a higher transport path to put the red cube in the box.",
            "Use a lower transport path to put the red cube in the box.",
        ),
    ]

    print(f"[*] Loading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id)
    print(f"[*] Dataset frames:   {len(dataset)}")
    print(f"[*] Dataset episodes: {dataset.num_episodes}")

    patched_meta = patch_dataset_meta_for_policy(dataset, rename_map)

    print(f"[*] Loading policy: {args.policy_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.device = str(device)

    policy = make_policy(policy_cfg, ds_meta=patched_meta)
    policy.eval()

    if hasattr(policy, "to"):
        policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        dataset_stats=patched_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    layers = get_lm_expert_layers(policy)
    num_layers = len(layers)

    early_layers = select_layer_indices(num_layers, "early")
    late_layers = select_layer_indices(num_layers, "late")
    full_layers = select_layer_indices(num_layers, "full")

    print("\n" + "=" * 80)
    print("Layer split")
    print("=" * 80)
    print(f"num_layers = {num_layers}")
    print(f"early      = {early_layers}")
    print(f"late       = {late_layers}")
    print(f"full       = {full_layers}")
    print("=" * 80 + "\n")

    if args.layer_group == "all":
        capture_layers = full_layers
        groups_to_save = ["early", "late", "full"]
    else:
        capture_layers = select_layer_indices(num_layers, args.layer_group)
        groups_to_save = [args.layer_group]

    diff_accumulated = collect_dense_caa_diffs(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        prompt_pairs=prompt_pairs,
        rename_map=rename_map,
        capture_layers=capture_layers,
        frames_per_episode=args.frames_per_episode,
        max_episodes=args.max_episodes,
        device=device,
        frame_start_frac=args.frame_start_frac,
        frame_end_frac=args.frame_end_frac,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in groups_to_save:
        selected_layers = select_layer_indices(num_layers, group)

        vectors, raw_vectors, layer_stats = build_vectors_from_diffs(
            diff_accumulated=diff_accumulated,
            selected_layers=selected_layers,
        )

        if args.output is not None and args.layer_group != "all":
            output_path = Path(args.output)
        else:
            output_path = output_dir / default_output_name(
                layer_group=group,
                layer_indices=selected_layers,
                max_episodes=args.max_episodes,
                frames_per_episode=args.frames_per_episode,
            )

        save_caa_file(
            output_path=output_path,
            layer_group=group,
            selected_layers=selected_layers,
            vectors=vectors,
            raw_vectors=raw_vectors,
            layer_stats=layer_stats,
            prompt_pairs=prompt_pairs,
            frames_per_episode=args.frames_per_episode,
            max_episodes=args.max_episodes,
            frame_start_frac=args.frame_start_frac,
            frame_end_frac=args.frame_end_frac,
        )


if __name__ == "__main__":
    main()