#!/usr/bin/env python3
"""
Compare two neuron-selection methods side by side and attach top tokens:

1) Keyword baseline from lerobot_reord_top_token.py
   - For every FFN value vector / neuron, compute top tokens from lm_head @ down_proj column.
   - Rank neurons whose top tokens match concept keywords such as high / low.

2) Physical trajectory-delta method from physical_neuron_picking.py
   - Hook each MLP down_proj input activation.
   - Average activations over high episodes and low episodes.
   - Rank by avg_high - avg_low.

Example:
    python compare_neuron_top_tokens.py \
        --repo-id ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
        --top-n 10 \
        --top-k-tokens 10 \
        --frame-stride 15 \
        --query-neuron 14:805 \
        --query-neuron 3:962
"""

import argparse
import copy
import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device


NeuronKey = Tuple[int, int]


@dataclass
class NeuronTokenInfo:
    layer: int
    neuron: int
    top_token_ids: List[int]
    top_tokens: List[str]
    top_logits: List[float]

    @property
    def top1_token(self) -> str:
        return self.top_tokens[0] if self.top_tokens else ""

    @property
    def max_logit(self) -> float:
        return float(self.top_logits[0]) if self.top_logits else float("nan")


# -----------------------------
# Episode indexing helpers
# -----------------------------

def get_episode_indices(dataset):
    """Return frame boundary information if available."""
    for obj in [dataset, dataset.meta]:
        if hasattr(obj, "episode_data_index"):
            return getattr(obj, "episode_data_index")

    if hasattr(dataset.meta, "episodes"):
        return dataset.meta.episodes

    return None


def _to_int(x) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def get_episode_frame_range(ep_index, ep_idx: int) -> Tuple[int, int]:
    """
    Robustly parse LeRobot episode frame ranges.

    Supports:
      - HuggingFace datasets.Dataset with dataset_from_index / dataset_to_index
      - dict/list variants with from/to, start/end, length
      - cumulative stop indices
    """
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
        if "from" in ep and "to" in ep:
            return _to_int(ep["from"]), _to_int(ep["to"])
        if "start" in ep and "end" in ep:
            return _to_int(ep["start"]), _to_int(ep["end"])
        if "dataset_from_index" in ep and "dataset_to_index" in ep:
            return _to_int(ep["dataset_from_index"]), _to_int(ep["dataset_to_index"])
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
            raise KeyError(f"Cannot find episode {ep_idx}; keys sample: {list(ep_index.keys())[:10]}")

        if isinstance(ep, dict):
            if "from" in ep and "to" in ep:
                return _to_int(ep["from"]), _to_int(ep["to"])
            if "start" in ep and "end" in ep:
                return _to_int(ep["start"]), _to_int(ep["end"])
            if "dataset_from_index" in ep and "dataset_to_index" in ep:
                return _to_int(ep["dataset_from_index"]), _to_int(ep["dataset_to_index"])
            if "length" in ep:
                start_frame = 0
                for i in range(ep_idx):
                    prev_ep = ep_index[i] if i in ep_index else ep_index[str(i)]
                    start_frame += _to_int(prev_ep["length"])
                return start_frame, start_frame + _to_int(ep["length"])
            raise KeyError(f"Unsupported episode dict keys: {ep.keys()}")

        start_frame = 0 if ep_idx == 0 else _to_int(ep_index[ep_idx - 1])
        return start_frame, _to_int(ep)

    start_frame = 0 if ep_idx == 0 else _to_int(ep_index[ep_idx - 1])
    return start_frame, _to_int(ep_index[ep_idx])


# -----------------------------
# Model loading
# -----------------------------

def build_default_rename_map() -> Dict[str, str]:
    return {
        "observation.images.front": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
        "observation.images.wrist": "observation.images.camera3",
    }


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device(get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu"))


def load_dataset_policy_preprocessor(repo_id: str, device: torch.device):
    print(f"[*] Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, video_backend="pyav")

    print("[*] Loading policy config...")
    policy_cfg = PreTrainedConfig.from_pretrained(repo_id)

    print("[*] Adjusting metadata to match policy expected features...")
    policy_meta = copy.deepcopy(dataset.meta)
    rename_map = build_default_rename_map()
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

    return dataset, policy, preprocessor


def _object_has_lm_head(obj) -> bool:
    return obj is not None and hasattr(obj, "lm_head") and hasattr(obj.lm_head, "weight")


def _object_has_text_model(obj) -> bool:
    return obj is not None and hasattr(obj, "text_model") and hasattr(obj.text_model, "layers")


def get_vlm_text_model_tokenizer(policy):
    """
    Return the model object that owns lm_head, the text_model, and tokenizer.

    In SmolVLA / SmolVLM wrappers these are not always on the same object:
      - policy.model.vlm_with_expert.vlm usually owns lm_head
      - policy.model.vlm_with_expert.get_vlm_model() usually owns text_model

    The previous version assumed one object had both attributes, which can fail with:
      AttributeError: VLM model has no lm_head attribute
    """
    vlm_with_expert = policy.model.vlm_with_expert

    candidates = []

    if hasattr(vlm_with_expert, "vlm"):
        candidates.append(("vlm_with_expert.vlm", vlm_with_expert.vlm))

    if hasattr(vlm_with_expert, "get_vlm_model"):
        candidates.append(("vlm_with_expert.get_vlm_model()", vlm_with_expert.get_vlm_model()))

    # Also inspect one common nesting level used by HF wrappers.
    expanded_candidates = []
    for name, obj in candidates:
        expanded_candidates.append((name, obj))
        for attr in ("model", "language_model"):
            if obj is not None and hasattr(obj, attr):
                expanded_candidates.append((f"{name}.{attr}", getattr(obj, attr)))

    lm_head_owner = None
    lm_head_owner_name = None
    for name, obj in expanded_candidates:
        if _object_has_lm_head(obj):
            lm_head_owner = obj
            lm_head_owner_name = name
            break

    text_model = None
    text_model_owner_name = None
    for name, obj in expanded_candidates:
        if _object_has_text_model(obj):
            text_model = obj.text_model
            text_model_owner_name = name
            break

    if lm_head_owner is None:
        available = [name for name, _ in expanded_candidates]
        raise AttributeError(
            "Cannot find lm_head.weight. Checked: " + ", ".join(available)
        )

    if text_model is None:
        available = [name for name, _ in expanded_candidates]
        raise AttributeError(
            "Cannot find text_model.layers. Checked: " + ", ".join(available)
        )

    if hasattr(vlm_with_expert, "processor") and hasattr(vlm_with_expert.processor, "tokenizer"):
        tokenizer = vlm_with_expert.processor.tokenizer
    elif hasattr(policy, "processor") and hasattr(policy.processor, "tokenizer"):
        tokenizer = policy.processor.tokenizer
    else:
        raise AttributeError(
            "Cannot find tokenizer at policy.model.vlm_with_expert.processor.tokenizer"
        )

    print(f"[*] lm_head source : {lm_head_owner_name}")
    print(f"[*] text_model source: {text_model_owner_name}")

    return lm_head_owner, text_model, tokenizer


# -----------------------------
# Top-token extraction
# -----------------------------

@torch.no_grad()
def extract_top_tokens_for_all_neurons(
    policy,
    device: torch.device,
    top_k_tokens: int = 10,
    neuron_chunk_size: int = 512,
) -> Tuple[List[NeuronTokenInfo], Dict[NeuronKey, NeuronTokenInfo]]:
    """
    For each FFN neuron, compute tokens most promoted by that neuron's value vector.

    Formula:
        token_logits[:, neuron] = lm_head.weight @ down_proj.weight[:, neuron]

    down_proj.weight shape is [d_model, d_ff], so each column is one FFN value vector.
    """
    vlm_model, text_model, tokenizer = get_vlm_text_model_tokenizer(policy)

    W_out = vlm_model.lm_head.weight.detach().to(device=device, dtype=torch.float32)
    num_layers = len(text_model.layers)

    metadata: List[NeuronTokenInfo] = []
    metadata_by_key: Dict[NeuronKey, NeuronTokenInfo] = {}

    print(f"[*] Extracting top-{top_k_tokens} tokens from {num_layers} VLM text layers...")

    for layer_idx in tqdm(range(num_layers), desc="Top-token extraction"):
        down_proj = text_model.layers[layer_idx].mlp.down_proj
        W_value = down_proj.weight.detach().to(device=device, dtype=torch.float32)  # [d_model, d_ff]
        d_ff = W_value.shape[1]

        for start in range(0, d_ff, neuron_chunk_size):
            end = min(start + neuron_chunk_size, d_ff)
            W_chunk = W_value[:, start:end]  # [d_model, chunk]
            token_logits = W_out @ W_chunk  # [vocab_size, chunk]
            top_logits, top_token_ids = torch.topk(token_logits, k=top_k_tokens, dim=0)

            top_logits_t = top_logits.transpose(0, 1).contiguous().cpu().numpy()
            top_token_ids_t = top_token_ids.transpose(0, 1).contiguous().cpu().numpy()

            for local_idx, neuron_idx in enumerate(range(start, end)):
                token_ids = top_token_ids_t[local_idx].tolist()
                decoded_tokens = [
                    tokenizer.decode([tok_id]).replace("\n", " ").strip()
                    for tok_id in token_ids
                ]
                info = NeuronTokenInfo(
                    layer=layer_idx,
                    neuron=neuron_idx,
                    top_token_ids=[int(x) for x in token_ids],
                    top_tokens=decoded_tokens,
                    top_logits=[float(x) for x in top_logits_t[local_idx].tolist()],
                )
                metadata.append(info)
                metadata_by_key[(layer_idx, neuron_idx)] = info

            del token_logits, top_logits, top_token_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[*] Extracted token metadata for {len(metadata)} neurons.")
    return metadata, metadata_by_key


def token_info_for_key(metadata_by_key: Dict[NeuronKey, NeuronTokenInfo], layer: int, neuron: int) -> Optional[NeuronTokenInfo]:
    return metadata_by_key.get((int(layer), int(neuron)))


# -----------------------------
# Previous keyword baseline
# -----------------------------

def token_matches_keywords(token: str, pos_keywords: Sequence[str], neg_keywords: Sequence[str]) -> bool:
    token_lower = token.lower()
    if any(neg_kw in token_lower for neg_kw in neg_keywords):
        return False
    return any(pos_kw in token_lower for pos_kw in pos_keywords)


def rank_neurons_by_keyword_frequency(
    metadata: Sequence[NeuronTokenInfo],
    keywords_dict,
    top_n: int = 10,
) -> List[dict]:
    if isinstance(keywords_dict, list):
        pos_keywords = [kw.lower() for kw in keywords_dict]
        neg_keywords: List[str] = []
    else:
        pos_keywords = [kw.lower() for kw in keywords_dict.get("pos", [])]
        neg_keywords = [kw.lower() for kw in keywords_dict.get("neg", [])]

    scored = []
    for info in metadata:
        match_count = sum(
            1 for token in info.top_tokens
            if token_matches_keywords(token, pos_keywords, neg_keywords)
        )
        scored.append({
            "rank": None,
            "layer": info.layer,
            "neuron": info.neuron,
            "match_count": match_count,
            "max_logit": info.max_logit,
            "top_token": info.top1_token,
            "top_tokens": info.top_tokens,
            "top_logits": info.top_logits,
        })

    scored.sort(key=lambda x: (x["match_count"], x["max_logit"]), reverse=True)
    selected = scored[:top_n]
    for i, item in enumerate(selected, start=1):
        item["rank"] = i
    return selected


def run_keyword_baseline(metadata: Sequence[NeuronTokenInfo], top_n: int) -> Dict[str, List[dict]]:
    concept_keywords_map = {
        "High Transport": {
            "pos": ["high"],
            "neg": ["thigh", "low", "slow", "lower"],
        },
        "Low Transport": {
            "pos": ["low"],
            "neg": ["follow", "allow", "slow", "blow", "glow", "yellow", "hollow"],
        },
    }

    return {
        concept: rank_neurons_by_keyword_frequency(metadata, keywords, top_n=top_n)
        for concept, keywords in concept_keywords_map.items()
    }


# -----------------------------
# Physical trajectory-delta ranking
# -----------------------------

@torch.no_grad()
def compute_physical_trajectory_delta_neurons(
    dataset,
    policy,
    preprocessor,
    top_n: int = 10,
    frame_stride: int = 15,
    high_last_episode: int = 29,
) -> Dict[str, List[dict]]:
    _, text_model, _ = get_vlm_text_model_tokenizer(policy)
    num_layers = len(text_model.layers)

    high_sums: List[Optional[np.ndarray]] = [None] * num_layers
    low_sums: List[Optional[np.ndarray]] = [None] * num_layers
    high_count = 0
    low_count = 0

    captured_act: Dict[int, np.ndarray] = {}

    def get_hook(layer_idx: int):
        def hook(module, inputs):
            # inputs[0] is activation before down_proj: [batch, seq_len, intermediate_dim]
            captured_act[layer_idx] = inputs[0].detach().mean(dim=(0, 1)).float().cpu().numpy()
        return hook

    handles = []
    for layer_idx in range(num_layers):
        handle = text_model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(get_hook(layer_idx))
        handles.append(handle)

    try:
        ep_index = get_episode_indices(dataset)
        if ep_index is None:
            raise AttributeError("Could not find episode_data_index or episodes metadata.")

        print(f"[*] Physical analysis starting for {dataset.num_episodes} episodes...")
        print(f"[*] High episodes: 0..{high_last_episode}; Low episodes: {high_last_episode + 1}..{dataset.num_episodes - 1}")

        for ep_idx in tqdm(range(dataset.num_episodes), desc="Physical activation analysis"):
            is_high = ep_idx <= high_last_episode
            start_frame, end_frame = get_episode_frame_range(ep_index, ep_idx)

            for frame_idx in range(start_frame, end_frame, frame_stride):
                frame = dataset[frame_idx]
                captured_act.clear()

                obs_processed = preprocessor(frame)
                policy.select_action(obs_processed)

                for layer_idx, act in captured_act.items():
                    if is_high:
                        if high_sums[layer_idx] is None:
                            high_sums[layer_idx] = np.zeros_like(act)
                        high_sums[layer_idx] += act
                    else:
                        if low_sums[layer_idx] is None:
                            low_sums[layer_idx] = np.zeros_like(act)
                        low_sums[layer_idx] += act

                if is_high:
                    high_count += 1
                else:
                    low_count += 1

        if high_count == 0 or low_count == 0:
            raise RuntimeError(f"Invalid counts: high_count={high_count}, low_count={low_count}")

        all_neurons = []
        for layer_idx in range(num_layers):
            if high_sums[layer_idx] is None or low_sums[layer_idx] is None:
                continue
            avg_high = high_sums[layer_idx] / high_count
            avg_low = low_sums[layer_idx] / low_count
            diff = avg_high - avg_low
            for neuron_idx, signed_score in enumerate(diff):
                all_neurons.append({
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "signed_score": float(signed_score),
                    "contrast_score": float(abs(signed_score)),
                })

        top_high = sorted(all_neurons, key=lambda x: x["signed_score"], reverse=True)[:top_n]
        top_low = sorted(all_neurons, key=lambda x: x["signed_score"])[:top_n]

        for i, item in enumerate(top_high, start=1):
            item["rank"] = i
            item["direction"] = "high"
        for i, item in enumerate(top_low, start=1):
            item["rank"] = i
            item["direction"] = "low"

        return {
            "High Transport": top_high,
            "Low Transport": top_low,
        }

    finally:
        for handle in handles:
            handle.remove()


def attach_token_info_to_physical_results(
    physical_results: Dict[str, List[dict]],
    metadata_by_key: Dict[NeuronKey, NeuronTokenInfo],
) -> Dict[str, List[dict]]:
    enriched = {}
    for concept, rows in physical_results.items():
        enriched_rows = []
        for row in rows:
            row = dict(row)
            info = token_info_for_key(metadata_by_key, row["layer"], row["neuron"])
            if info is None:
                row.update({
                    "top_token": "<missing>",
                    "top_tokens": [],
                    "top_logits": [],
                    "max_logit": float("nan"),
                })
            else:
                row.update({
                    "top_token": info.top1_token,
                    "top_tokens": info.top_tokens,
                    "top_logits": info.top_logits,
                    "max_logit": info.max_logit,
                })
            enriched_rows.append(row)
        enriched[concept] = enriched_rows
    return enriched


# -----------------------------
# Printing / exporting
# -----------------------------

def compact_tokens(tokens: Sequence[str], max_tokens: int = 8) -> str:
    tokens = list(tokens)[:max_tokens]
    return "[" + ", ".join(tokens) + "]"


def fmt_neuron(layer: int, neuron: int) -> str:
    return f"L{int(layer)}:{int(neuron)}"


def trim(s: str, width: int) -> str:
    s = str(s)
    if len(s) <= width:
        return s
    return s[: max(0, width - 3)] + "..."


def print_single_neuron_queries(
    query_neurons: Sequence[str],
    metadata_by_key: Dict[NeuronKey, NeuronTokenInfo],
) -> None:
    if not query_neurons:
        return

    print("\n" + "=" * 110)
    print("Queried neuron top tokens")
    print("=" * 110)
    print(f"{'Neuron':<10} | {'Top1':<18} | {'MaxLogit':<9} | Top tokens")
    print("-" * 110)

    for spec in query_neurons:
        try:
            layer_s, neuron_s = spec.split(":", 1)
            layer = int(layer_s)
            neuron = int(neuron_s)
        except Exception:
            print(f"{spec:<10} | INVALID: expected format layer:neuron, e.g. 14:805")
            continue

        info = metadata_by_key.get((layer, neuron))
        if info is None:
            print(f"{spec:<10} | NOT FOUND")
            continue
        print(
            f"{fmt_neuron(layer, neuron):<10} | "
            f"{trim(info.top1_token, 18):<18} | "
            f"{info.max_logit:<9.4f} | "
            f"{compact_tokens(info.top_tokens, max_tokens=10)}"
        )


def print_side_by_side_tables(
    keyword_results: Dict[str, List[dict]],
    physical_results: Dict[str, List[dict]],
    top_n: int,
) -> None:
    for concept in ["High Transport", "Low Transport"]:
        print("\n" + "=" * 150)
        print(f"{concept}: keyword baseline vs physical trajectory-delta")
        print("=" * 150)
        print(
            f"{'Rank':<4} | "
            f"{'Keyword neuron':<15} | {'KW cnt':<6} | {'KW top1':<16} | {'KW top tokens':<40} || "
            f"{'Physical neuron':<15} | {'Phys score':<10} | {'Phys top1':<16} | {'Phys top tokens':<40}"
        )
        print("-" * 150)

        kw_rows = keyword_results.get(concept, [])
        ph_rows = physical_results.get(concept, [])
        for i in range(top_n):
            kw = kw_rows[i] if i < len(kw_rows) else None
            ph = ph_rows[i] if i < len(ph_rows) else None

            if kw is None:
                kw_neuron = kw_cnt = kw_top1 = kw_tokens = ""
            else:
                kw_neuron = fmt_neuron(kw["layer"], kw["neuron"])
                kw_cnt = str(kw["match_count"])
                kw_top1 = trim(kw["top_token"], 16)
                kw_tokens = trim(compact_tokens(kw["top_tokens"], max_tokens=6), 40)

            if ph is None:
                ph_neuron = ph_score = ph_top1 = ph_tokens = ""
            else:
                ph_neuron = fmt_neuron(ph["layer"], ph["neuron"])
                if concept == "Low Transport":
                    # Low neurons have negative signed_score; print abs for readability.
                    ph_score = f"{abs(ph['signed_score']):.6f}"
                else:
                    ph_score = f"{ph['signed_score']:.6f}"
                ph_top1 = trim(ph["top_token"], 16)
                ph_tokens = trim(compact_tokens(ph["top_tokens"], max_tokens=6), 40)

            print(
                f"{i + 1:<4} | "
                f"{kw_neuron:<15} | {kw_cnt:<6} | {kw_top1:<16} | {kw_tokens:<40} || "
                f"{ph_neuron:<15} | {ph_score:<10} | {ph_top1:<16} | {ph_tokens:<40}"
            )


def print_physical_only_tables(physical_results: Dict[str, List[dict]]) -> None:
    for concept in ["High Transport", "Low Transport"]:
        print("\n" + "=" * 110)
        print(f"Physical {concept} neurons with top tokens")
        print("=" * 110)
        print(f"{'Rank':<4} | {'Neuron':<10} | {'SignedScore':<12} | {'AbsScore':<10} | {'Top1':<18} | Top tokens")
        print("-" * 110)
        for row in physical_results.get(concept, []):
            print(
                f"{row['rank']:<4} | "
                f"{fmt_neuron(row['layer'], row['neuron']):<10} | "
                f"{row['signed_score']:<12.6f} | "
                f"{abs(row['signed_score']):<10.6f} | "
                f"{trim(row['top_token'], 18):<18} | "
                f"{compact_tokens(row['top_tokens'], max_tokens=10)}"
            )


def write_csv(
    csv_path: str,
    keyword_results: Dict[str, List[dict]],
    physical_results: Dict[str, List[dict]],
    top_n: int,
) -> None:
    fieldnames = [
        "concept",
        "rank",
        "keyword_layer",
        "keyword_neuron",
        "keyword_match_count",
        "keyword_max_logit",
        "keyword_top1",
        "keyword_top_tokens",
        "physical_layer",
        "physical_neuron",
        "physical_signed_score",
        "physical_abs_score",
        "physical_max_logit",
        "physical_top1",
        "physical_top_tokens",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for concept in ["High Transport", "Low Transport"]:
            kw_rows = keyword_results.get(concept, [])
            ph_rows = physical_results.get(concept, [])
            for i in range(top_n):
                kw = kw_rows[i] if i < len(kw_rows) else {}
                ph = ph_rows[i] if i < len(ph_rows) else {}
                writer.writerow({
                    "concept": concept,
                    "rank": i + 1,
                    "keyword_layer": kw.get("layer", ""),
                    "keyword_neuron": kw.get("neuron", ""),
                    "keyword_match_count": kw.get("match_count", ""),
                    "keyword_max_logit": kw.get("max_logit", ""),
                    "keyword_top1": kw.get("top_token", ""),
                    "keyword_top_tokens": " | ".join(kw.get("top_tokens", [])),
                    "physical_layer": ph.get("layer", ""),
                    "physical_neuron": ph.get("neuron", ""),
                    "physical_signed_score": ph.get("signed_score", ""),
                    "physical_abs_score": abs(ph["signed_score"]) if "signed_score" in ph else "",
                    "physical_max_logit": ph.get("max_logit", ""),
                    "physical_top1": ph.get("top_token", ""),
                    "physical_top_tokens": " | ".join(ph.get("top_tokens", [])),
                })

    print(f"\n[*] Wrote CSV: {csv_path}")


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2")
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu. Default uses LeRobot safe device.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-k-tokens", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--high-last-episode", type=int, default=29)
    parser.add_argument("--neuron-chunk-size", type=int, default=512)
    parser.add_argument("--csv-out", default="neuron_top_token_comparison.csv")
    parser.add_argument(
        "--query-neuron",
        action="append",
        default=[],
        help="Print top tokens for a specific neuron. Format: layer:neuron. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[*] Using device: {device}")

    dataset, policy, preprocessor = load_dataset_policy_preprocessor(args.repo_id, device)

    token_metadata, metadata_by_key = extract_top_tokens_for_all_neurons(
        policy=policy,
        device=device,
        top_k_tokens=args.top_k_tokens,
        neuron_chunk_size=args.neuron_chunk_size,
    )

    print_single_neuron_queries(args.query_neuron, metadata_by_key)

    keyword_results = run_keyword_baseline(token_metadata, top_n=args.top_n)

    physical_results_raw = compute_physical_trajectory_delta_neurons(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        top_n=args.top_n,
        frame_stride=args.frame_stride,
        high_last_episode=args.high_last_episode,
    )
    physical_results = attach_token_info_to_physical_results(physical_results_raw, metadata_by_key)

    print_physical_only_tables(physical_results)
    print_side_by_side_tables(keyword_results, physical_results, top_n=args.top_n)
    write_csv(args.csv_out, keyword_results, physical_results, top_n=args.top_n)


if __name__ == "__main__":
    main()