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

def state_to_q_rad(state_vec, use_site_pose=True):
    """Map raw robot degree state arrays into radians configuration space."""
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)
    if state_vec.size < 5:
        raise ValueError(f"State dimensionality must be >= 5, got shape {state_vec.shape}")
    
    if state_vec.size >= 6:
        q_deg = state_vec[:6].copy()
    else:
        q_deg = np.concatenate([state_vec[:5], np.array([0.0], dtype=np.float64)], axis=0)

    q_rad = np.deg2rad(q_deg)
    if use_site_pose:
        q_rad[5] = 0.0  # End joint angle does not shift base platform sensor positioning
    return q_rad

def compute_eef_height_from_state(state_vec, mj_model, mj_data, use_site_pose=True):
    """Calculate absolute Z-axis physical height of tool flange using kinematic equations."""
    q = state_to_q_rad(state_vec, use_site_pose=use_site_pose)
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    
    for i, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_adr = mj_model.jnt_qposadr[joint_id]
        mj_data.qpos[qpos_adr] = float(q[i])

    mujoco.mj_fwdPosition(mj_model, mj_data)

    if use_site_pose:
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
        pos = mj_data.site_xpos[site_id].copy()
    else:
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        pos = mj_data.xpos[body_id].copy()

    return float(pos[2])  # Return isolated Cartesian coordinate (Z-height index)

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
        
        # Log peak Cartesian z height achieved during the specific episode trajectory
        peak_z = np.max(frame_z_heights) if len(frame_z_heights) > 0 else 0.0
        episode_max_heights.append(peak_z)
        print(f"[Episode {ep_idx:03d}] max EEF height = {peak_z:.4f} m")
        
        # Package and append calculated mean frame states
        for i in range(num_layers):
            episode_activations[i].append(np.mean(frame_cache[i], axis=0))
            
    # Clean up forward hooks immediately following loop termination
    for h in handles: 
        h.remove()

    # ==============================================================================
    # 4. GROUND-TRUTH MOVEMENT COHORT SORTING (100% MEDIAN SPLIT VIA MUJOCO PHYSICS)
    # ==============================================================================
    episode_max_heights = np.array(episode_max_heights)
    sorted_indices = np.argsort(episode_max_heights)
    
    # ⚡ SHIFT TO 100% UTILIZATION: Split the physically sorted array exactly in half
    # This automatically assigns the lower half of heights to Low, and the upper half to High,
    # cleanly absorbing human operational errors (like Episode 21) into the correct bucket.
    mid_point = len(sorted_indices) // 2
    low_group_indices = sorted_indices[:mid_point]
    high_group_indices = sorted_indices[mid_point:]

    print("\n" + "="*70)
    print("AUTOMATED KINEMATIC PARSING - 100% DATASET VALIDATION PROFILE")
    print("="*70)
    print(f"[+] Total Available Episodes: {dataset.num_episodes}")
    print(f"[+] Extracted High-Group IDs ({len(high_group_indices)}): {sorted(list(high_group_indices))}")
    print(f"[+] Extracted Low-Group IDs  ({len(low_group_indices)}): {sorted(list(low_group_indices))}")
    print(f"    --> Measured Height Boundary Minimum: {np.min(episode_max_heights):.4f} m")
    print(f"    --> Measured Height Boundary Maximum: {np.max(episode_max_heights):.4f} m")

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
