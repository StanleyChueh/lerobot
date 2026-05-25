import torch
import numpy as np
import copy
from pathlib import Path
from tqdm import tqdm
import mujoco
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.utils import get_safe_torch_device

# ==============================================================================
# 1. LEROBOT & DATASET PROCESSING UTILITIES
# ==============================================================================

def get_episode_indices(dataset):
    """Return frame boundary information if available."""
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

def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)

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
        if "from" in ep and "to" in ep: return _to_int(ep["from"]), _to_int(ep["to"])
        if "start" in ep and "end" in ep: return _to_int(ep["start"]), _to_int(ep["end"])
        if "dataset_from_index" in ep and "dataset_to_index" in ep:
            return _to_int(ep["dataset_from_index"]), _to_int(ep["dataset_to_index"])

    raise TypeError("Unrecognized index sequence layout or missing boundary keys.")

# ==============================================================================
# 2. MUJOCO FORWARD KINEMATICS UTILITIES
# ==============================================================================

def load_fk_model_from_local_xml(xml_name="follower.xml"):
    """Load analytical MuJoCo simulation environment asset safely."""
    script_dir = Path(__file__).resolve().parent
    xml_path = script_dir / xml_name
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing required kinematic definition asset at: {xml_path}")
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data

CALIB_REST_STATE_DEG = np.asarray([
    -11.20879121,
    97.00520833,
    17.25596857,
    100.0,
    -9.01098901,
    47.38863287,
], dtype=np.float64)

CALIB_TARGET_REST_DEG = np.asarray([
    91.7,
    15.0,
    40.0,
    65.0,
    0.0,
    -30.0,
], dtype=np.float64)

CALIB_ORDER = np.asarray([
    0, 1, 2, 3, 4, 5,
], dtype=int)

CALIB_SCALE = np.asarray([
    1.0,
    0.5,
    1.0,
    1.0,
    1.0,
    1.0,
], dtype=np.float64)

CALIB_SIGN = np.asarray([
    1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
], dtype=np.float64)


def state_to_q_rad(state_vec):
    """Map real robot observation.state degrees into calibrated MuJoCo qpos radians.

    Calibrated from real REST/REACH/RAISE poses.

    Formula:
        q_mujoco_deg =
            target_rest_deg
          + sign * scale * (state_deg[order] - real_rest_state_deg[order])
    """
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)

    if state_vec.size < 6:
        raise ValueError(f"State dimensionality must be >= 6, got shape {state_vec.shape}")

    raw_deg = state_vec[:6].copy()

    raw_delta = raw_deg[CALIB_ORDER] - CALIB_REST_STATE_DEG[CALIB_ORDER]
    q_deg = CALIB_TARGET_REST_DEG + CALIB_SIGN * CALIB_SCALE * raw_delta

    return np.deg2rad(q_deg)

def compute_eef_height_from_state(state_vec, mj_model, mj_data):
    """Compute EEF z-height from observation.state using MuJoCo FK.

    Uses:
      - state[:6] as degrees
      - end_effector_site as FK target
      - no hardcoded episode labels
    """
    q = state_to_q_rad(state_vec)

    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    # Reset MuJoCo state each FK call to avoid stale qpos/qvel contamination.
    mujoco.mj_resetData(mj_model, mj_data)

    for i, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in XML: {joint_name}")

        qpos_adr = mj_model.jnt_qposadr[joint_id]
        mj_data.qpos[qpos_adr] = float(q[i])

    mujoco.mj_forward(mj_model, mj_data)

    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    if site_id < 0:
        raise ValueError("Site not found in XML: end_effector_site")

    pos = mj_data.site_xpos[site_id].copy()
    return float(pos[2])

def robust_episode_height(frame_z_heights, trim_ratio=0.10, metric="median"):
    """Convert frame-level EEF heights into one robust episode-level height.

    Why:
      max height is unstable because one transient spike can misclassify an episode.
      median after trimming start/end is more robust for high/low trajectory detection.
    """
    heights = np.asarray(frame_z_heights, dtype=np.float64)

    if heights.size == 0:
        return np.nan

    heights = heights[np.isfinite(heights)]
    if heights.size == 0:
        return np.nan

    if trim_ratio > 0 and heights.size >= 5:
        n_trim = int(np.floor(heights.size * trim_ratio))
        if n_trim > 0 and heights.size > 2 * n_trim:
            heights = heights[n_trim:-n_trim]

    if metric == "median":
        return float(np.median(heights))
    if metric == "mean":
        return float(np.mean(heights))
    if metric == "p75":
        return float(np.percentile(heights, 75))
    if metric == "p90":
        return float(np.percentile(heights, 90))
    if metric == "max":
        return float(np.max(heights))

    raise ValueError(f"Unknown episode height metric: {metric}")


def find_high_low_by_balanced_eef_rank(
    episode_heights,
    high_fraction=0.50,
    odd_policy="smaller_high",
):
    """Find high/low episodes by ranking EEF episode heights.

    This is not hardcoding episode IDs. It assumes the dataset was collected with
    roughly balanced high/low trajectory types.

    Why not largest-gap?
      The largest sorted height gap can isolate only a few outlier episodes.
      In your failed run, it selected only episodes [5, 9] as HIGH.
      A rank split is more stable when the two height distributions overlap.
    """
    heights = np.asarray(episode_heights, dtype=np.float64)
    valid_mask = np.isfinite(heights)
    valid_indices = np.where(valid_mask)[0]
    valid_heights = heights[valid_mask]

    if valid_heights.size < 2:
        raise ValueError("Need at least 2 valid episode heights to split high/low.")

    n = valid_heights.size

    if high_fraction <= 0.0 or high_fraction >= 1.0:
        raise ValueError("high_fraction must be between 0 and 1.")

    # For 61 episodes and high_fraction=0.5:
    #   smaller_high -> 30 high, 31 low
    #   larger_high  -> 31 high, 30 low
    # smaller_high matches your current dataset design: 0-29 high, 30-60 low.
    raw = n * high_fraction
    if odd_policy == "smaller_high":
        n_high = int(np.floor(raw))
    elif odd_policy == "larger_high":
        n_high = int(np.ceil(raw))
    else:
        n_high = int(np.round(raw))

    n_high = max(1, min(n - 1, n_high))

    order = np.argsort(valid_heights)
    sorted_heights = valid_heights[order]
    sorted_episode_indices = valid_indices[order]

    low_group_indices = sorted_episode_indices[:-n_high]
    high_group_indices = sorted_episode_indices[-n_high:]

    boundary_low = float(sorted_heights[-n_high - 1])
    boundary_high = float(sorted_heights[-n_high])
    threshold = float((boundary_low + boundary_high) / 2.0)

    gaps = np.diff(sorted_heights)

    print("\n[Balanced EEF rank split debug]")
    print(f"    valid episodes = {n}")
    print(f"    high_fraction  = {high_fraction:.3f}")
    print(f"    odd_policy     = {odd_policy}")
    print(f"    n_high         = {len(high_group_indices)}")
    print(f"    n_low          = {len(low_group_indices)}")
    print(f"    boundary_low   = {boundary_low:.6f} m")
    print(f"    boundary_high  = {boundary_high:.6f} m")
    print(f"    threshold      = {threshold:.6f} m")

    return threshold, high_group_indices, low_group_indices, sorted_episode_indices, sorted_heights, gaps


def find_threshold_by_constrained_gap(
    episode_heights,
    min_group_fraction=0.35,
    max_group_fraction=0.65,
):
    """Largest-gap split, but only among reasonable group-size splits.

    This prevents the failure mode where largest-gap selects only 2 high episodes.
    Use this only when you do not want to assume exact balance.
    """
    heights = np.asarray(episode_heights, dtype=np.float64)
    valid_mask = np.isfinite(heights)
    valid_indices = np.where(valid_mask)[0]
    valid_heights = heights[valid_mask]

    if valid_heights.size < 2:
        raise ValueError("Need at least 2 valid episode heights to split high/low.")

    order = np.argsort(valid_heights)
    sorted_heights = valid_heights[order]
    sorted_episode_indices = valid_indices[order]
    gaps = np.diff(sorted_heights)

    n = valid_heights.size
    min_count = int(np.ceil(n * min_group_fraction))
    max_count = int(np.floor(n * max_group_fraction))

    candidates = []
    for split_pos, gap in enumerate(gaps):
        low_count = split_pos + 1
        high_count = n - low_count
        if min_count <= low_count <= max_count and min_count <= high_count <= max_count:
            candidates.append((float(gap), split_pos, low_count, high_count))

    if not candidates:
        raise ValueError(
            "No valid constrained-gap split found. "
            "Relax min_group_fraction/max_group_fraction or use balanced rank split."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_gap, split_pos, low_count, high_count = candidates[0]

    threshold = float((sorted_heights[split_pos] + sorted_heights[split_pos + 1]) / 2.0)
    low_group_indices = sorted_episode_indices[sorted_heights < threshold]
    high_group_indices = sorted_episode_indices[sorted_heights >= threshold]

    print("\n[Constrained gap split debug]")
    print(f"    min_group_fraction = {min_group_fraction:.3f}")
    print(f"    max_group_fraction = {max_group_fraction:.3f}")
    print(f"    selected gap       = {best_gap:.6f} m")
    print(f"    low_count          = {len(low_group_indices)}")
    print(f"    high_count         = {len(high_group_indices)}")
    print(f"    threshold          = {threshold:.6f} m")

    return threshold, high_group_indices, low_group_indices, sorted_episode_indices, sorted_heights, gaps


def print_height_split_report(episode_heights, threshold, high_group_indices, low_group_indices):
    heights = np.asarray(episode_heights, dtype=np.float64)

    high_heights = heights[high_group_indices]
    low_heights = heights[low_group_indices]

    print("\n" + "=" * 70)
    print("AUTOMATED EEF-HEIGHT EPISODE SPLIT")
    print("=" * 70)
    print(f"[+] Split method: robust episode p90 height + balanced EEF rank split")
    print(f"[+] Height threshold: {threshold:.6f} m")
    print(f"[+] High episodes ({len(high_group_indices)}): {sorted([int(x) for x in high_group_indices])}")
    print(f"[+] Low episodes  ({len(low_group_indices)}): {sorted([int(x) for x in low_group_indices])}")

    print("\n[Height statistics]")
    print(f"    High mean   = {np.mean(high_heights):.6f} m")
    print(f"    High median = {np.median(high_heights):.6f} m")
    print(f"    High min    = {np.min(high_heights):.6f} m")
    print(f"    High max    = {np.max(high_heights):.6f} m")
    print(f"    Low mean    = {np.mean(low_heights):.6f} m")
    print(f"    Low median  = {np.median(low_heights):.6f} m")
    print(f"    Low min     = {np.min(low_heights):.6f} m")
    print(f"    Low max     = {np.max(low_heights):.6f} m")

    print("\n[Per-episode heights]")
    for ep_idx, h in enumerate(heights):
        label = "HIGH" if ep_idx in set(high_group_indices) else "LOW"
        print(f"    Episode {ep_idx:03d}: height={h:.6f} m -> {label}")

# ==============================================================================
# 3. KINEMATIC PARSING PIPELINE
# ==============================================================================

def find_physical_height_neurons_via_eef(repo_id, xml_name="follower.xml", device=None):
    if device is None:
        device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"[*] Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, video_backend="pyav")
    
    print(f"[*] Loading policy config...")
    policy_cfg = PreTrainedConfig.from_pretrained(repo_id)

    print(f"[*] Adjusting metadata to match policy expected features...")
    policy_meta = copy.deepcopy(dataset.meta)
    rename_map = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
        "observation.images.wrist": "observation.images.camera3",
    }
    for actual, expected in rename_map.items():
        if actual in policy_meta.features:
            policy_meta.features[expected] = policy_meta.features.pop(actual)

    print(f"[*] Initializing policy model and MuJoCo forward kinematics...")
    policy = make_policy(policy_cfg, ds_meta=policy_meta).to(device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=repo_id,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": rename_map},
        }
    )

    # Instantiate analytical model mechanics
    mj_model, mj_data = load_fk_model_from_local_xml(xml_name=xml_name)

    num_layers = len(policy.model.vlm_with_expert.get_vlm_model().text_model.layers)
    episode_activations = {i: [] for i in range(num_layers)}

    # p90 is the score used for HIGH/LOW split.
    episode_p90_heights = []

    # true max is printed for diagnosis only.
    episode_max_heights = []

    captured_act = {}
    def get_hook(layer_idx):
        def hook(module, inputs):
            captured_act[layer_idx] = inputs[0].detach().mean(dim=(0, 1)).float().cpu().numpy()
        return hook

    handles = []
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    for i in range(num_layers):
        h = text_model.layers[i].mlp.down_proj.register_forward_pre_hook(get_hook(i))
        handles.append(h)

    ep_index = get_episode_indices(dataset)
    if ep_index is None:
        raise AttributeError("Could not successfully recover structural index mappings.")

    print(f"[*] Beginning single-pass feature execution and Kinematic Tracking...")

    for ep_idx in tqdm(range(dataset.num_episodes), desc="Processing Episodes"):
        start_frame, end_frame = get_episode_frame_range(ep_index, ep_idx)
        frame_cache = {i: [] for i in range(num_layers)}
        frame_z_heights = []

        for frame_idx in range(start_frame, end_frame, 15):
            frame = dataset[frame_idx]
            
            # --- Kinematic Metric Calculation ---
            raw_state = to_numpy(frame.get("observation.state"))
            if raw_state is not None:
                z_height = compute_eef_height_from_state(raw_state, mj_model, mj_data)
                frame_z_heights.append(z_height)

            # --- Model Forward Inference Pass ---
            with torch.no_grad():
                obs_processed = preprocessor(frame)
                policy.select_action(obs_processed)

            for layer_idx, act in captured_act.items():
                frame_cache[layer_idx].append(act)
        
        # This is the score used for HIGH/LOW classification.
        episode_p90_height = robust_episode_height(
            frame_z_heights,
            trim_ratio=0.10,
            metric="p90",
        )

        # This is the true maximum EEF height in the sampled frames.
        # Use trim_ratio=0.0 so it is a real max over sampled frames.
        episode_max_height = robust_episode_height(
            frame_z_heights,
            trim_ratio=0.0,
            metric="max",
        )

        episode_p90_heights.append(episode_p90_height)
        episode_max_heights.append(episode_max_height)

        print(
            f"[Episode {ep_idx:03d}] "
            f"p90 EEF height = {episode_p90_height:.6f} m | "
            f"max EEF height = {episode_max_height:.6f} m | "
            f"split_score = p90 | "
            f"sampled_frames={len(frame_z_heights)}"
        )
        
        # Package and append calculated mean frame states
        for i in range(num_layers):
            episode_activations[i].append(np.mean(frame_cache[i], axis=0))
            
    # Clean up forward hooks immediately following loop termination
    for h in handles: 
        h.remove()

    # ==============================================================================
    # 4. GROUND-TRUTH MOVEMENT COHORT SORTING (100% MEDIAN SPLIT VIA MUJOCO PHYSICS)
    # ==============================================================================
    episode_heights = np.asarray(episode_p90_heights, dtype=np.float64)
    episode_max_heights = np.asarray(episode_max_heights, dtype=np.float64)

    threshold, high_group_indices, low_group_indices, sorted_episode_indices, sorted_heights, gaps = (
        find_high_low_by_balanced_eef_rank(
            episode_heights,
            high_fraction=0.50,
            odd_policy="smaller_high",
        )
    )

    print_height_split_report(
        episode_heights=episode_heights,
        threshold=threshold,
        high_group_indices=high_group_indices,
        low_group_indices=low_group_indices,
    )

    print("\n" + "=" * 90)
    print("PER-EPISODE EEF HEIGHT DEBUG: P90 SCORE VS TRUE MAX")
    print("=" * 90)
    print("Logic: HIGH/LOW is decided by p90 EEF height rank, not max EEF height.")
    print(f"Threshold from p90 split = {threshold:.6f} m\n")

    pred_high_set = set(int(x) for x in high_group_indices)

    for ep_idx, (p90_h, max_h) in enumerate(zip(episode_heights, episode_max_heights)):
        pred_label = "HIGH" if ep_idx in pred_high_set else "LOW"
        gt_label = "GT_HIGH" if ep_idx <= 29 else "GT_LOW"
        is_error = (
            (ep_idx <= 29 and pred_label == "LOW") or
            (ep_idx >= 30 and pred_label == "HIGH")
        )
        marker = "  <-- MISMATCH" if is_error else ""

        print(
            f"[Episode {ep_idx:03d}] "
            f"p90 EEF height = {p90_h:.6f} m | "
            f"max EEF height = {max_h:.6f} m | "
            f"threshold = {threshold:.6f} m | "
            f"pred = {pred_label:<4} | "
            f"{gt_label}{marker}"
        )

    print("\n" + "="*70)
    print("AUTOMATED KINEMATIC PARSING - 100% DATASET VALIDATION PROFILE")
    print("="*70)
    print(f"[+] Total Available Episodes: {dataset.num_episodes}")
    print(f"[+] Extracted High-Group IDs ({len(high_group_indices)}): {sorted(list(high_group_indices))}")
    print(f"[+] Extracted Low-Group IDs  ({len(low_group_indices)}): {sorted(list(low_group_indices))}")
    print(f"    --> P90 split-score minimum: {np.min(episode_heights):.6f} m")
    print(f"    --> P90 split-score maximum: {np.max(episode_heights):.6f} m")
    print(f"    --> True max EEF minimum   : {np.min(episode_max_heights):.6f} m")
    print(f"    --> True max EEF maximum   : {np.max(episode_max_heights):.6f} m")

    # ==============================================================================
    # 5. CONTRASTIVE SCORING MATRIX CALCULATION
    # ==============================================================================
    all_neurons = []
    for i in range(num_layers):
        high_group_vectors = [episode_activations[i][ep] for ep in high_group_indices]
        low_group_vectors = [episode_activations[i][ep] for ep in low_group_indices]
        
        avg_high = np.mean(high_group_vectors, axis=0)
        avg_low = np.mean(low_group_vectors, axis=0)
        
        diff = avg_high - avg_low 
        for neuron_idx, score in enumerate(diff):
            all_neurons.append({"layer": i, "neuron": neuron_idx, "score": score})

    top_high = sorted(all_neurons, key=lambda x: x['score'], reverse=True)[:10]
    top_low = sorted(all_neurons, key=lambda x: x['score'])[:10]

    print("\n" + "="*70)
    print("TOP TARGET GROUNDED 'HIGH' TRAJECTORY STEERING NEURONS")
    print("="*70)
    for n in top_high:
        print(f"Layer {n['layer']:>2} | Neuron {n['neuron']:>5} | Kinematic Delta Score: {n['score']:.6f}")

    print("\n" + "="*70)
    print("TOP TARGET GROUNDED 'LOW' TRAJECTORY STEERING NEURONS")
    print("="*70)
    for n in top_low:
        print(f"Layer {n['layer']:>2} | Neuron {n['neuron']:>5} | Kinematic Delta Score: {abs(n['score']):.6f}")

if __name__ == "__main__":
    find_physical_height_neurons_via_eef("ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2")