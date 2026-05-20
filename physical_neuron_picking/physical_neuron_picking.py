import torch
import numpy as np
import copy
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.utils import get_safe_torch_device


def get_episode_indices(dataset):
    """
    Return frame boundary information if available.

    Important:
    dataset.meta.episodes is often episode-level metadata, not frame boundaries.
    So we should prefer episode_data_index first.
    """

    # Best source: direct frame boundary index
    for obj in [dataset, dataset.meta]:
        if hasattr(obj, "episode_data_index"):
            return getattr(obj, "episode_data_index")

    # Fallback only: episode metadata
    if hasattr(dataset.meta, "episodes"):
        return dataset.meta.episodes

    return None


def _to_int(x):
    """Convert torch/numpy scalar or Python scalar to int."""
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def get_episode_frame_range(ep_index, ep_idx):
    """
    Robustly parse LeRobot episode frame ranges.

    Supports:
    1. HuggingFace datasets.Dataset with dataset_from_index / dataset_to_index
    2. {"from": tensor/list, "to": tensor/list}
    3. [{"from": ..., "to": ...}, ...]
    4. {0: {"from": ..., "to": ...}, ...}
    5. {0: {"length": ...}, ...}
    6. cumulative stop indices
    """

    # Case 0: HuggingFace datasets.Dataset
    # Your debug output shows this is the correct case:
    # features include dataset_from_index and dataset_to_index.
    if hasattr(ep_index, "column_names"):
        columns = set(ep_index.column_names)

        if "dataset_from_index" in columns and "dataset_to_index" in columns:
            row = ep_index[ep_idx]
            return (
                _to_int(row["dataset_from_index"]),
                _to_int(row["dataset_to_index"]),
            )

        if "from" in columns and "to" in columns:
            row = ep_index[ep_idx]
            return _to_int(row["from"]), _to_int(row["to"])

        if "length" in columns:
            start_frame = 0
            for i in range(ep_idx):
                start_frame += _to_int(ep_index[i]["length"])
            end_frame = start_frame + _to_int(ep_index[ep_idx]["length"])
            return start_frame, end_frame

        raise KeyError(
            f"Unsupported HuggingFace Dataset episode index columns: {ep_index.column_names}"
        )

    # Case 1: dict with from/to arrays
    if isinstance(ep_index, dict) and "from" in ep_index and "to" in ep_index:
        return _to_int(ep_index["from"][ep_idx]), _to_int(ep_index["to"][ep_idx])

    # Case 2: list of episode dicts
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
            end_frame = start_frame + _to_int(ep["length"])
            return start_frame, end_frame

        raise KeyError(f"Unsupported episode dict keys: {ep.keys()}")

    # Case 3/4: dict keyed by episode id
    if isinstance(ep_index, dict):
        if ep_idx in ep_index:
            ep = ep_index[ep_idx]
        elif str(ep_idx) in ep_index:
            ep = ep_index[str(ep_idx)]
        else:
            raise KeyError(
                f"Cannot find episode {ep_idx}. "
                f"Available keys sample: {list(ep_index.keys())[:10]}"
            )

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

                end_frame = start_frame + _to_int(ep["length"])
                return start_frame, end_frame

            raise KeyError(f"Unsupported episode dict keys: {ep.keys()}")

        start_frame = 0 if ep_idx == 0 else _to_int(ep_index[ep_idx - 1])
        end_frame = _to_int(ep)
        return start_frame, end_frame

    # Case 5: cumulative stop indices
    start_frame = 0 if ep_idx == 0 else _to_int(ep_index[ep_idx - 1])
    end_frame = _to_int(ep_index[ep_idx])
    return start_frame, end_frame


def find_physical_height_neurons(repo_id, device=None):
    if device is None:
        device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"[*] Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, video_backend="pyav")
    
    print(f"[*] Loading policy config...")
    policy_cfg = PreTrainedConfig.from_pretrained(repo_id)

    # --- ⚡ METADATA SHIM ⚡ ---
    # We modify a copy of the metadata to pass the policy's validation check.
    print(f"[*] Adjusting metadata to match policy expected features...")
    policy_meta = copy.deepcopy(dataset.meta)
    
    # Mapping dataset's descriptive names to policy's expected generic names
    rename_map = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
        "observation.images.wrist": "observation.images.camera3",
    }
    
    for actual, expected in rename_map.items():
        if actual in policy_meta.features:
            policy_meta.features[expected] = policy_meta.features.pop(actual)

    print(f"[*] Initializing policy...")
    policy = make_policy(policy_cfg, ds_meta=policy_meta).to(device)
    policy.eval()

    # Preprocessor handles renaming of image keys during the forward pass
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=repo_id,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": rename_map},
        }
    )

    num_layers = len(policy.model.vlm_with_expert.get_vlm_model().text_model.layers)
    high_sums = [None] * num_layers
    low_sums = [None] * num_layers
    high_count = 0
    low_count = 0

    captured_act = {}
    def get_hook(layer_idx):
        def hook(module, inputs):
            # inputs[0] is the activation before down_proj [batch, seq_len, intermediate_dim]
            captured_act[layer_idx] = inputs[0].detach().mean(dim=(0, 1)).float().cpu().numpy()
        return hook

    handles = []
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    for i in range(num_layers):
        h = text_model.layers[i].mlp.down_proj.register_forward_pre_hook(get_hook(i))
        handles.append(h)

    # --- ⚡ ROBUST INDEX RETRIEVAL ⚡ ---
    ep_index = get_episode_indices(dataset)

    print("[DEBUG] ep_index type:", type(ep_index))

    if isinstance(ep_index, dict):
        print("[DEBUG] ep_index keys sample:", list(ep_index.keys())[:5])
        first_key = list(ep_index.keys())[0]
        print("[DEBUG] first episode value:", ep_index[first_key])
    elif isinstance(ep_index, list):
        print("[DEBUG] first episode value:", ep_index[0])
    else:
        print("[DEBUG] ep_index:", ep_index)
    
    if ep_index is None:
        raise AttributeError(f"Could not find indexing information. Available: {dir(dataset.meta)}")

    print(f"[*] Analysis starting for {dataset.num_episodes} episodes...")

    for ep_idx in tqdm(range(dataset.num_episodes), desc="Analyzing Episodes"):
        # Episodes 0-29: High trajectory | 30-61: Low trajectory
        is_high = (ep_idx <= 29)
        
        # --- ⚡ UPDATED INDEXING LOGIC ⚡ ---
        # If ep_index is a list (common for the 'episodes' attribute), 
        # it usually contains dictionaries with 'from' and 'to' keys.
        start_frame, end_frame = get_episode_frame_range(ep_index, ep_idx)
        # ------------------------------------

        # Sample every 15th frame for performance
        for frame_idx in range(start_frame, end_frame, 15):
            frame = dataset[frame_idx]
            with torch.no_grad():
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
            
            if is_high: high_count += 1
            else: low_count += 1

    # --- SCORING & RESULTS ---
    all_neurons = []
    for i in range(num_layers):
        if high_sums[i] is None or low_sums[i] is None: continue
        avg_high = high_sums[i] / high_count
        avg_low = low_sums[i] / low_count
        diff = avg_high - avg_low 
        for neuron_idx, score in enumerate(diff):
            all_neurons.append({"layer": i, "neuron": neuron_idx, "score": score})

    top_high = sorted(all_neurons, key=lambda x: x['score'], reverse=True)[:10]
    top_low = sorted(all_neurons, key=lambda x: x['score'])[:10]

    print("\n" + "="*70)
    print("TOP PHYSICAL 'HIGH' NEURONS (Based on Trajectory Delta)")
    print("="*70)
    for n in top_high:
        print(f"Layer {n['layer']:>2} | Neuron {n['neuron']:>5} | Contrast Score: {n['score']:.6f}")

    print("\n" + "="*70)
    print("TOP PHYSICAL 'LOW' NEURONS (Based on Trajectory Delta)")
    print("="*70)
    for n in top_low:
        print(f"Layer {n['layer']:>2} | Neuron {n['neuron']:>5} | Contrast Score: {abs(n['score']):.6f}")

    # Remove hooks
    for h in handles: h.remove()

if __name__ == "__main__":
    find_physical_height_neurons("ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2")
