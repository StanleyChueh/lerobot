#!/usr/bin/env python3
"""
physical_color_neuron_finding_v2.py

Improved red/green physical neuron finder for SmolVLA / LeRobot.

Why this version is stricter than the previous one:
  1) Forces a fresh policy forward for every sampled frame by resetting the policy
     action queue before each sample. This avoids silently collecting only a few
     activation samples when select_action reuses cached action chunks.
  2) Computes episode-level means first, then compares red vs green episodes.
     This avoids overweighting episodes that produce more sampled frames.
  3) Ranks neurons by a standardized red-vs-green effect:
        effect = (mean_red - mean_green) / pooled_std
     The raw signed difference is still saved.
  4) Adds split-half stability and color-token diagnostics.
  5) Optionally removes neurons that overlap with known height/transport neurons.

Default dataset assumption:
  - Episodes 0..29  : red cube trajectories
  - Episodes 30..59 : green cube trajectories
  - Prompt constant: "put the cube in the box"

Example:
  python physical_color_neuron_finding_v2.py \
    --repo-id ethanCSL/svla_koch_pick_n_place_vla_steering_color \
    --red-start 0 --red-end 29 \
    --green-start 30 --green-end 59 \
    --frame-stride 5 \
    --top-n 20 \
    --top-k-tokens 20 \
    --force-fresh-forward \
    --exclude-height-overlap

Recommended first run:
  python physical_color_neuron_finding_v2.py --frame-stride 5 --top-n 20 --top-k-tokens 20 --force-fresh-forward
"""

import argparse
import copy
import csv
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.utils import get_safe_torch_device


# ----------------------------
# Episode indexing utilities
# ----------------------------

def _to_int(x: Any) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def get_episode_indices(dataset: Any) -> Any:
    for obj in [dataset, dataset.meta]:
        if hasattr(obj, "episode_data_index"):
            return getattr(obj, "episode_data_index")
    if hasattr(dataset.meta, "episodes"):
        return dataset.meta.episodes
    return None


def get_episode_frame_range(ep_index: Any, ep_idx: int) -> Tuple[int, int]:
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

        raise KeyError(f"Unsupported episode index columns: {ep_index.column_names}")

    if isinstance(ep_index, dict) and "from" in ep_index and "to" in ep_index:
        return _to_int(ep_index["from"][ep_idx]), _to_int(ep_index["to"][ep_idx])

    if isinstance(ep_index, list) and isinstance(ep_index[ep_idx], dict):
        ep = ep_index[ep_idx]
        for a, b in [("from", "to"), ("start", "end"), ("dataset_from_index", "dataset_to_index")]:
            if a in ep and b in ep:
                return _to_int(ep[a]), _to_int(ep[b])
        if "length" in ep:
            start_frame = sum(_to_int(ep_index[i]["length"]) for i in range(ep_idx))
            return start_frame, start_frame + _to_int(ep["length"])
        raise KeyError(f"Unsupported episode dict keys: {ep.keys()}")

    if isinstance(ep_index, dict):
        if ep_idx in ep_index:
            ep = ep_index[ep_idx]
        elif str(ep_idx) in ep_index:
            ep = ep_index[str(ep_idx)]
        else:
            raise KeyError(f"Cannot find episode {ep_idx}")

        if isinstance(ep, dict):
            for a, b in [("from", "to"), ("start", "end"), ("dataset_from_index", "dataset_to_index")]:
                if a in ep and b in ep:
                    return _to_int(ep[a]), _to_int(ep[b])
            if "length" in ep:
                start_frame = 0
                for i in range(ep_idx):
                    prev = ep_index[i] if i in ep_index else ep_index[str(i)]
                    start_frame += _to_int(prev["length"])
                return start_frame, start_frame + _to_int(ep["length"])
            raise KeyError(f"Unsupported episode dict keys: {ep.keys()}")

        start_frame = 0 if ep_idx == 0 else _to_int(ep_index[ep_idx - 1])
        end_frame = _to_int(ep)
        return start_frame, end_frame

    start_frame = 0 if ep_idx == 0 else _to_int(ep_index[ep_idx - 1])
    end_frame = _to_int(ep_index[ep_idx])
    return start_frame, end_frame


# ----------------------------
# Model component discovery
# ----------------------------

def _add_candidate(candidates, seen, name, obj) -> None:
    if obj is None:
        return
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)
    candidates.append((name, obj))


def _object_has_lm_head(obj: Any) -> bool:
    return obj is not None and hasattr(obj, "lm_head") and hasattr(obj.lm_head, "weight")


def _object_has_text_model(obj: Any) -> bool:
    return obj is not None and hasattr(obj, "text_model") and hasattr(obj.text_model, "layers")


def get_vlm_text_model_tokenizer(policy: Any) -> Tuple[Any, Any, Any]:
    if not hasattr(policy, "model") or not hasattr(policy.model, "vlm_with_expert"):
        raise AttributeError("policy.model.vlm_with_expert not found")

    vlm_with_expert = policy.model.vlm_with_expert
    candidates = []
    seen = set()

    _add_candidate(candidates, seen, "policy", policy)
    _add_candidate(candidates, seen, "policy.model", getattr(policy, "model", None))
    _add_candidate(candidates, seen, "policy.model.vlm_with_expert", vlm_with_expert)

    if hasattr(vlm_with_expert, "vlm"):
        _add_candidate(candidates, seen, "vlm_with_expert.vlm", vlm_with_expert.vlm)

    if hasattr(vlm_with_expert, "get_vlm_model"):
        try:
            _add_candidate(candidates, seen, "vlm_with_expert.get_vlm_model()", vlm_with_expert.get_vlm_model())
        except Exception as exc:
            print(f"[WARN] Could not call get_vlm_model(): {exc}")

    base_candidates = list(candidates)
    for base_name, base_obj in base_candidates:
        for attr in ("model", "language_model", "text_model"):
            if hasattr(base_obj, attr):
                _add_candidate(candidates, seen, f"{base_name}.{attr}", getattr(base_obj, attr))

    lm_head_owner = None
    lm_head_owner_name = None
    for name, obj in candidates:
        if _object_has_lm_head(obj):
            lm_head_owner = obj
            lm_head_owner_name = name
            break

    text_model = None
    text_model_owner_name = None
    for name, obj in candidates:
        if _object_has_text_model(obj):
            text_model = obj.text_model
            text_model_owner_name = name
            break
        if hasattr(obj, "layers"):
            try:
                _ = len(obj.layers)
                text_model = obj
                text_model_owner_name = name
                break
            except Exception:
                pass

    tokenizer_candidates = [
        ("vlm_with_expert.processor.tokenizer", getattr(getattr(vlm_with_expert, "processor", None), "tokenizer", None)),
        ("policy.processor.tokenizer", getattr(getattr(policy, "processor", None), "tokenizer", None)),
        ("policy.model.processor.tokenizer", getattr(getattr(getattr(policy, "model", None), "processor", None), "tokenizer", None)),
    ]

    tokenizer = None
    tokenizer_source = None
    for name, obj in tokenizer_candidates:
        if obj is not None:
            tokenizer = obj
            tokenizer_source = name
            break

    if lm_head_owner is None:
        raise AttributeError("Cannot find lm_head.weight. Checked: " + ", ".join(n for n, _ in candidates))
    if text_model is None:
        raise AttributeError("Cannot find text_model.layers. Checked: " + ", ".join(n for n, _ in candidates))
    if tokenizer is None:
        raise AttributeError("Cannot find tokenizer")

    print(f"[*] lm_head source   : {lm_head_owner_name}")
    print(f"[*] text_model source: {text_model_owner_name}")
    print(f"[*] tokenizer source : {tokenizer_source}")

    return lm_head_owner, text_model, tokenizer


def build_policy_and_preprocessor(repo_id: str, device: torch.device):
    print(f"[*] Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, video_backend="pyav")

    print("[*] Loading policy config...")
    policy_cfg = PreTrainedConfig.from_pretrained(repo_id)

    print("[*] Adjusting metadata to match policy expected features...")
    policy_meta = copy.deepcopy(dataset.meta)

    rename_map = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
        "observation.images.wrist": "observation.images.camera3",
    }

    for actual, expected in rename_map.items():
        if actual in policy_meta.features:
            policy_meta.features[expected] = policy_meta.features.pop(actual)

    print("[*] Initializing policy...")
    policy = make_policy(policy_cfg, ds_meta=policy_meta).to(device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=repo_id,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    _, text_model, _ = get_vlm_text_model_tokenizer(policy)
    return dataset, policy, preprocessor, text_model


# ----------------------------
# Top-token extraction
# ----------------------------

@torch.no_grad()
def extract_top_tokens_for_all_neurons(policy, text_model, top_k_tokens: int, device: torch.device):
    lm_head_owner, _, tokenizer = get_vlm_text_model_tokenizer(policy)
    W_out = lm_head_owner.lm_head.weight.detach().to(device=device, dtype=torch.float32)

    metadata = []
    metadata_by_key = {}
    global_id = 0

    print(f"[*] Extracting top-{top_k_tokens} tokens for all FFN neurons...")

    for layer_idx in tqdm(range(len(text_model.layers)), desc="Extracting top tokens"):
        W_value = text_model.layers[layer_idx].mlp.down_proj.weight.detach().to(device=device, dtype=torch.float32)

        if W_out.shape[1] != W_value.shape[0]:
            raise RuntimeError(
                f"Shape mismatch at layer {layer_idx}: W_out={tuple(W_out.shape)}, W_value={tuple(W_value.shape)}"
            )

        token_logits = W_out @ W_value
        k = min(top_k_tokens, token_logits.shape[0])
        top_logits, top_token_ids = torch.topk(token_logits, k=k, dim=0)

        top_logits_t = top_logits.transpose(0, 1).contiguous().cpu().numpy()
        top_token_ids_t = top_token_ids.transpose(0, 1).contiguous().cpu().numpy()

        for neuron_idx in range(W_value.shape[1]):
            token_ids = [int(x) for x in top_token_ids_t[neuron_idx].tolist()]
            tokens = [
                tokenizer.decode([tok_id]).replace("\n", " ").strip()
                for tok_id in token_ids
            ]

            item = {
                "global_id": global_id,
                "layer": int(layer_idx),
                "neuron": int(neuron_idx),
                "top_token_ids": token_ids,
                "top_tokens": tokens,
                "top_logits": [float(x) for x in top_logits_t[neuron_idx].tolist()],
                "top1_token": tokens[0] if tokens else "",
                "max_logit": float(top_logits_t[neuron_idx][0]),
            }
            metadata.append(item)
            metadata_by_key[(layer_idx, neuron_idx)] = item
            global_id += 1

    print(f"[*] Extracted top-token metadata for {len(metadata)} neurons.")
    return metadata, metadata_by_key


# ----------------------------
# Activation capture
# ----------------------------

def reduce_activation_tensor_to_neuron_vector(x: torch.Tensor, reduction: str) -> np.ndarray:
    x = x.detach()

    if x.ndim == 3:
        if reduction == "last_token":
            x = x[:, -1, :].mean(dim=0)
        elif reduction == "max":
            x = x.amax(dim=(0, 1))
        elif reduction == "mean_abs":
            x = x.abs().mean(dim=(0, 1))
        else:
            x = x.mean(dim=(0, 1))
    elif x.ndim == 2:
        if reduction == "max":
            x = x.amax(dim=0)
        elif reduction == "mean_abs":
            x = x.abs().mean(dim=0)
        else:
            x = x.mean(dim=0)
    elif x.ndim == 1:
        pass
    else:
        dims = tuple(range(x.ndim - 1))
        if reduction == "max":
            x = x.amax(dim=dims)
        elif reduction == "mean_abs":
            x = x.abs().mean(dim=dims)
        else:
            x = x.mean(dim=dims)

    return x.float().cpu().numpy()


def make_capture_hook(layer_idx: int, captured_act: Dict[int, np.ndarray], reduction: str):
    def hook(module, inputs):
        captured_act[layer_idx] = reduce_activation_tensor_to_neuron_vector(inputs[0], reduction=reduction)
    return hook


def reset_for_fresh_forward(policy, preprocessor=None):
    """
    Force select_action() to run the model instead of reusing a cached action chunk.
    """
    if hasattr(policy, "reset"):
        policy.reset()
    # In most LeRobot processors, reset is safe. It should not change normalization stats.
    if preprocessor is not None and hasattr(preprocessor, "reset"):
        preprocessor.reset()


@torch.no_grad()
def collect_episode_mean_activations(
    dataset,
    policy,
    preprocessor,
    text_model,
    ep_index,
    episode_indices: List[int],
    frame_stride: int,
    max_frames_per_episode: Optional[int],
    force_fresh_forward: bool,
    reduction: str,
    group_name: str,
):
    """
    Returns:
      episode_vectors_by_layer[layer_idx] = np.ndarray [num_episodes_with_samples, d_ff]
      frame_counts_by_episode = list[int]
    """
    captured_act: Dict[int, np.ndarray] = {}
    handles = []

    for layer_idx in range(len(text_model.layers)):
        h = text_model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
            make_capture_hook(layer_idx, captured_act, reduction=reduction)
        )
        handles.append(h)

    episode_vectors_by_layer: Dict[int, List[np.ndarray]] = defaultdict(list)
    frame_counts_by_episode: List[int] = []

    try:
        for ep_idx in tqdm(episode_indices, desc=f"Collecting {group_name} episodes"):
            start_frame, end_frame = get_episode_frame_range(ep_index, ep_idx)
            frame_indices = list(range(start_frame, end_frame, frame_stride))

            if max_frames_per_episode is not None and max_frames_per_episode > 0:
                frame_indices = frame_indices[:max_frames_per_episode]

            layer_sums: Dict[int, np.ndarray] = {}
            layer_counts: Dict[int, int] = defaultdict(int)
            good_frames = 0

            for frame_idx in frame_indices:
                frame = dataset[frame_idx]
                captured_act.clear()

                if force_fresh_forward:
                    reset_for_fresh_forward(policy, preprocessor)

                obs_processed = preprocessor(frame)
                policy.select_action(obs_processed)

                if len(captured_act) == 0:
                    continue

                good_frames += 1

                for layer_idx, act in captured_act.items():
                    if layer_idx not in layer_sums:
                        layer_sums[layer_idx] = np.zeros_like(act, dtype=np.float64)
                    layer_sums[layer_idx] += act.astype(np.float64)
                    layer_counts[layer_idx] += 1

            if good_frames == 0:
                print(f"[WARN] {group_name} episode {ep_idx} produced 0 captured forward passes.")
                continue

            frame_counts_by_episode.append(good_frames)

            for layer_idx, summed in layer_sums.items():
                episode_vectors_by_layer[layer_idx].append(summed / float(layer_counts[layer_idx]))

    finally:
        for h in handles:
            h.remove()

    stacked = {}
    for layer_idx, vectors in episode_vectors_by_layer.items():
        if len(vectors) > 0:
            stacked[layer_idx] = np.stack(vectors, axis=0)

    return stacked, frame_counts_by_episode


# ----------------------------
# Ranking
# ----------------------------

def token_color_score(tokens: List[str], positive_color: str) -> int:
    """
    Diagnostic only. Do not rely on this as the main score.
    """
    t = " ".join(tok.lower() for tok in tokens)
    red_terms = ["red", "crimson", "scarlet"]
    green_terms = ["green", "lime", "emerald"]

    if positive_color == "red":
        return sum(term in t for term in red_terms) - sum(term in t for term in green_terms)
    if positive_color == "green":
        return sum(term in t for term in green_terms) - sum(term in t for term in red_terms)
    return 0


def split_half_effect(red_X: np.ndarray, green_X: np.ndarray, neuron_idx: int, seed: int) -> Tuple[float, float, float]:
    """
    Compute effect on two random half splits of episodes.
    Returns effect_a, effect_b, same_sign_flag.
    """
    rng = np.random.default_rng(seed)
    red_indices = np.arange(red_X.shape[0])
    green_indices = np.arange(green_X.shape[0])
    rng.shuffle(red_indices)
    rng.shuffle(green_indices)

    def half_effect(ridx, gidx):
        if len(ridx) == 0 or len(gidx) == 0:
            return 0.0
        r = red_X[ridx, neuron_idx]
        g = green_X[gidx, neuron_idx]
        raw = float(r.mean() - g.mean())
        pooled = float(np.sqrt(0.5 * (r.var(ddof=1) + g.var(ddof=1)) + 1e-8)) if len(r) > 1 and len(g) > 1 else 1.0
        return raw / pooled

    r_mid = max(1, len(red_indices) // 2)
    g_mid = max(1, len(green_indices) // 2)

    eff_a = half_effect(red_indices[:r_mid], green_indices[:g_mid])
    eff_b = half_effect(red_indices[r_mid:], green_indices[g_mid:])

    same_sign = 1.0 if eff_a * eff_b > 0 else 0.0
    return float(eff_a), float(eff_b), same_sign


def build_height_overlap_set() -> set:
    """
    Known height/transport neurons from your previous run.
    Excluding these can reduce reuse of generic trajectory neurons.
    """
    high_transport = {
        0: [1293],
        1: [1050],
        3: [2259],
        4: [1183],
        7: [295],
        11: [1115, 1595],
        13: [431],
        14: [736, 805],
    }
    low_transport = {
        3: [962],
        4: [1627],
        6: [587],
        7: [1007],
        9: [149],
        11: [1066],
        12: [629, 1164],
        14: [423],
        15: [1886],
    }

    s = set()
    for d in [high_transport, low_transport]:
        for layer, neurons in d.items():
            for neuron in neurons:
                s.add((int(layer), int(neuron)))
    return s


def rank_neurons(
    red_by_layer: Dict[int, np.ndarray],
    green_by_layer: Dict[int, np.ndarray],
    metadata_by_key: Dict[Tuple[int, int], Dict[str, Any]],
    top_n: int,
    exclude_height_overlap: bool,
    min_abs_effect: float,
    seed: int,
):
    overlap_set = build_height_overlap_set() if exclude_height_overlap else set()
    all_items = []

    common_layers = sorted(set(red_by_layer.keys()) & set(green_by_layer.keys()))

    for layer_idx in common_layers:
        red_X = red_by_layer[layer_idx]
        green_X = green_by_layer[layer_idx]

        if red_X.ndim != 2 or green_X.ndim != 2:
            continue

        red_mean = red_X.mean(axis=0)
        green_mean = green_X.mean(axis=0)
        red_var = red_X.var(axis=0, ddof=1) if red_X.shape[0] > 1 else np.zeros_like(red_mean)
        green_var = green_X.var(axis=0, ddof=1) if green_X.shape[0] > 1 else np.zeros_like(green_mean)

        raw_diff = red_mean - green_mean
        pooled_std = np.sqrt(0.5 * (red_var + green_var) + 1e-8)
        effect = raw_diff / pooled_std

        # Welch-style standard error and t-like diagnostic.
        se = np.sqrt((red_var / max(red_X.shape[0], 1)) + (green_var / max(green_X.shape[0], 1)) + 1e-8)
        t_score = raw_diff / se

        for neuron_idx in range(red_X.shape[1]):
            key = (int(layer_idx), int(neuron_idx))
            if key in overlap_set:
                continue

            tok = metadata_by_key.get(key, {})
            tokens = tok.get("top_tokens", [])
            logits = tok.get("top_logits", [])

            e = float(effect[neuron_idx])
            if abs(e) < min_abs_effect:
                continue

            sh_a, sh_b, same_sign = split_half_effect(red_X, green_X, neuron_idx, seed + layer_idx * 100000 + neuron_idx)

            item = {
                "layer": int(layer_idx),
                "neuron": int(neuron_idx),
                "raw_diff": float(raw_diff[neuron_idx]),
                "abs_raw_diff": float(abs(raw_diff[neuron_idx])),
                "effect": e,
                "abs_effect": float(abs(e)),
                "t_score": float(t_score[neuron_idx]),
                "abs_t_score": float(abs(t_score[neuron_idx])),
                "red_mean": float(red_mean[neuron_idx]),
                "green_mean": float(green_mean[neuron_idx]),
                "red_std": float(np.sqrt(red_var[neuron_idx] + 1e-8)),
                "green_std": float(np.sqrt(green_var[neuron_idx] + 1e-8)),
                "split_half_a": sh_a,
                "split_half_b": sh_b,
                "split_half_same_sign": same_sign,
                "top1_token": tokens[0] if tokens else "",
                "top_tokens": tokens,
                "top_logits": logits,
                "red_token_score": token_color_score(tokens, "red"),
                "green_token_score": token_color_score(tokens, "green"),
                "excluded_height_overlap": False,
            }
            all_items.append(item)

    # Main ranking:
    #  - standardized effect is primary
    #  - split-half same sign encourages stable neurons
    #  - abs raw diff breaks ties
    red_candidates = [x for x in all_items if x["effect"] > 0]
    green_candidates = [x for x in all_items if x["effect"] < 0]

    red_candidates.sort(
        key=lambda x: (
            x["split_half_same_sign"],
            x["abs_effect"],
            x["abs_t_score"],
            x["abs_raw_diff"],
        ),
        reverse=True,
    )
    green_candidates.sort(
        key=lambda x: (
            x["split_half_same_sign"],
            x["abs_effect"],
            x["abs_t_score"],
            x["abs_raw_diff"],
        ),
        reverse=True,
    )

    top_red = red_candidates[:top_n]
    top_green = green_candidates[:top_n]

    for i, x in enumerate(top_red, 1):
        x["rank"] = i
        x["group"] = "red_positive"

    for i, x in enumerate(top_green, 1):
        x["rank"] = i
        x["group"] = "green_positive"

    return top_red, top_green, all_items


# ----------------------------
# Output
# ----------------------------

def fmt_tokens(tokens: Iterable[str], max_items: int = 10) -> str:
    shown = list(tokens)[:max_items]
    return "[" + ", ".join(repr(t) for t in shown) + "]"


def group_neurons_by_layer(neurons: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    grouped = defaultdict(list)
    for n in neurons:
        grouped[int(n["layer"])].append(int(n["neuron"]))
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def print_table(title: str, rows: List[Dict[str, Any]], color: str):
    print("\n" + "=" * 150)
    print(title)
    print("=" * 150)
    print(
        f"{'Rank':<5} | {'Layer':<5} | {'Neuron':<6} | {'Effect':<10} | {'RawDiff':<10} | "
        f"{'T':<10} | {'Split':<5} | {'RedMean':<10} | {'GreenMean':<10} | {'Top1':<16} | Top tokens"
    )
    print("-" * 150)

    for n in rows:
        print(
            f"{n['rank']:<5} | "
            f"L{n['layer']:<4} | "
            f"{n['neuron']:<6} | "
            f"{n['effect']:<10.4f} | "
            f"{n['raw_diff']:<10.6f} | "
            f"{n['t_score']:<10.3f} | "
            f"{int(n['split_half_same_sign']):<5} | "
            f"{n['red_mean']:<10.6f} | "
            f"{n['green_mean']:<10.6f} | "
            f"{repr(n['top1_token']):<16} | "
            f"{fmt_tokens(n['top_tokens'])}"
        )


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fieldnames = [
        "group",
        "rank",
        "layer",
        "neuron",
        "effect",
        "abs_effect",
        "raw_diff",
        "abs_raw_diff",
        "t_score",
        "abs_t_score",
        "red_mean",
        "green_mean",
        "red_std",
        "green_std",
        "split_half_a",
        "split_half_b",
        "split_half_same_sign",
        "top1_token",
        "top_tokens",
        "top_logits",
        "red_token_score",
        "green_token_score",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["top_tokens"] = " | ".join(r.get("top_tokens", []))
            out["top_logits"] = " | ".join(f"{x:.6f}" for x in r.get("top_logits", []))
            w.writerow({k: out.get(k, "") for k in fieldnames})


def save_json(path: str, data: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ----------------------------
# Main
# ----------------------------

def select_episode_indices(num_episodes: int, start: int, end: int, max_episodes: Optional[int], seed: int) -> List[int]:
    end = min(end, num_episodes - 1)
    indices = list(range(start, end + 1))
    if max_episodes is not None and max_episodes > 0 and len(indices) > max_episodes:
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, max_episodes))
    return indices


@torch.no_grad()
def run(args):
    if args.device is None:
        device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[*] Using device: {device}")

    dataset, policy, preprocessor, text_model = build_policy_and_preprocessor(args.repo_id, device)

    _, metadata_by_key = extract_top_tokens_for_all_neurons(
        policy=policy,
        text_model=text_model,
        top_k_tokens=args.top_k_tokens,
        device=device,
    )

    ep_index = get_episode_indices(dataset)
    if ep_index is None:
        raise AttributeError("Could not find episode indexing information")

    if hasattr(ep_index, "column_names"):
        print("[*] Episode index columns:", list(ep_index.column_names))

    red_episodes = select_episode_indices(dataset.num_episodes, args.red_start, args.red_end, args.max_episodes_per_group, args.seed)
    green_episodes = select_episode_indices(dataset.num_episodes, args.green_start, args.green_end, args.max_episodes_per_group, args.seed + 1)

    print(f"[*] Red episodes   : {red_episodes[0]}..{red_episodes[-1]} ({len(red_episodes)} selected)")
    print(f"[*] Green episodes : {green_episodes[0]}..{green_episodes[-1]} ({len(green_episodes)} selected)")
    print(f"[*] Frame stride   : {args.frame_stride}")
    print(f"[*] Force fresh forward: {args.force_fresh_forward}")
    print(f"[*] Activation reduction: {args.activation_reduction}")

    red_by_layer, red_counts = collect_episode_mean_activations(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        text_model=text_model,
        ep_index=ep_index,
        episode_indices=red_episodes,
        frame_stride=args.frame_stride,
        max_frames_per_episode=args.max_frames_per_episode,
        force_fresh_forward=args.force_fresh_forward,
        reduction=args.activation_reduction,
        group_name="red",
    )

    green_by_layer, green_counts = collect_episode_mean_activations(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        text_model=text_model,
        ep_index=ep_index,
        episode_indices=green_episodes,
        frame_stride=args.frame_stride,
        max_frames_per_episode=args.max_frames_per_episode,
        force_fresh_forward=args.force_fresh_forward,
        reduction=args.activation_reduction,
        group_name="green",
    )

    print(f"[*] Red episodes with captured activations  : {len(red_counts)} / {len(red_episodes)}")
    print(f"[*] Green episodes with captured activations: {len(green_counts)} / {len(green_episodes)}")
    print(f"[*] Red captured frames total  : {sum(red_counts)}")
    print(f"[*] Green captured frames total: {sum(green_counts)}")

    if len(red_counts) < 5 or len(green_counts) < 5:
        print("[WARN] Very few episodes produced activation samples. The ranking may be unstable.")

    top_red, top_green, all_items = rank_neurons(
        red_by_layer=red_by_layer,
        green_by_layer=green_by_layer,
        metadata_by_key=metadata_by_key,
        top_n=args.top_n,
        exclude_height_overlap=args.exclude_height_overlap,
        min_abs_effect=args.min_abs_effect,
        seed=args.seed,
    )

    print_table("TOP PHYSICAL RED-POSITIVE NEURONS, ranked by standardized effect", top_red, "red")
    print_table("TOP PHYSICAL GREEN-POSITIVE NEURONS, ranked by standardized effect", top_green, "green")

    red_dict = group_neurons_by_layer(top_red)
    green_dict = group_neurons_by_layer(top_green)

    print("\n" + "=" * 150)
    print("STEERING CANDIDATE DICTS")
    print("=" * 150)
    print("# Add positive alpha to these neurons to test red steering.")
    print("RED_NEURONS_BY_LAYER = " + repr(red_dict))
    print()
    print("# Add positive alpha to these neurons to test green steering.")
    print("GREEN_NEURONS_BY_LAYER = " + repr(green_dict))
    print()
    print("# For steering test, do NOT use a red-specific prompt if testing green steering.")
    print('# Use a neutral prompt like: "Put the cube in the box."')
    print("# Then compare: baseline, green +alpha, green -alpha, random control.")

    rows = top_red + top_green
    save_csv(args.output_csv, rows)

    save_json(args.output_json, {
        "repo_id": args.repo_id,
        "red_episode_range": [args.red_start, args.red_end],
        "green_episode_range": [args.green_start, args.green_end],
        "red_episodes_used": red_episodes,
        "green_episodes_used": green_episodes,
        "red_captured_frames_total": int(sum(red_counts)),
        "green_captured_frames_total": int(sum(green_counts)),
        "red_episode_frame_counts": red_counts,
        "green_episode_frame_counts": green_counts,
        "frame_stride": args.frame_stride,
        "force_fresh_forward": bool(args.force_fresh_forward),
        "activation_reduction": args.activation_reduction,
        "exclude_height_overlap": bool(args.exclude_height_overlap),
        "top_red": top_red,
        "top_green": top_green,
    })

    print(f"[*] Saved CSV : {args.output_csv}")
    print(f"[*] Saved JSON: {args.output_json}")


def parse_args():
    p = argparse.ArgumentParser(description="Improved physical red/green neuron finder.")

    p.add_argument("--repo-id", type=str, default="ethanCSL/svla_koch_pick_n_place_vla_steering_color")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--red-start", type=int, default=0)
    p.add_argument("--red-end", type=int, default=29)
    p.add_argument("--green-start", type=int, default=30)
    p.add_argument("--green-end", type=int, default=59)

    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--max-frames-per-episode", type=int, default=None)
    p.add_argument("--max-episodes-per-group", type=int, default=None)

    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--top-k-tokens", type=int, default=20)
    p.add_argument("--min-abs-effect", type=float, default=0.0)

    p.add_argument(
        "--activation-reduction",
        type=str,
        default="mean",
        choices=["mean", "last_token", "max", "mean_abs"],
        help="How to reduce [batch, seq, d_ff] activation to [d_ff].",
    )

    p.add_argument(
        "--force-fresh-forward",
        action="store_true",
        help="Reset policy/preprocessor before each sampled frame so select_action cannot reuse cached action chunks.",
    )

    p.add_argument(
        "--exclude-height-overlap",
        action="store_true",
        help="Exclude neurons already found in the previous high/low transport run.",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", type=str, default="physical_color_neurons_v2.csv")
    p.add_argument("--output-json", type=str, default="physical_color_neurons_v2.json")

    args = p.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0")

    return args


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()