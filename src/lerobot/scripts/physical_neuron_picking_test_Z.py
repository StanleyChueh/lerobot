import torch
import numpy as np
import copy
import mujoco
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.utils import get_safe_torch_device

# ============================================================================
# 1. MuJoCo Z-Axis Kinematics Functions (整合自 compare_steering_motion_eef.py)
# ============================================================================
def load_fk_model_from_local_xml(xml_name="follower.xml"):
    script_dir = Path(__file__).resolve().parent
    xml_path = script_dir / xml_name
    if not xml_path.exists():
        raise FileNotFoundError(f"Cannot find XML at: {xml_path}")

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data

def state_to_q_rad(state_vec, use_site_pose=True):
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)
    if state_vec.size < 5:
        raise ValueError(f"state size must be >= 5, got shape {state_vec.shape}")

    if state_vec.size >= 6:
        q_deg = state_vec[:6].copy()
    else:
        q_deg = np.concatenate([state_vec[:5], np.array([0.0], dtype=np.float64)], axis=0)

    q_rad = np.deg2rad(q_deg)
    if use_site_pose:
        q_rad[5] = 0.0
    return q_rad

def compute_eef_z_height(state_vec, mj_model, mj_data, use_site_pose=True):
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

    return pos[2] # 唯獨返回 Z 軸高度

# ============================================================================
# 2. Episode Parsing Functions (來自原本的腳本)
# ============================================================================
def get_episode_indices(dataset):
    for obj in [dataset, dataset.meta]:
        if hasattr(obj, "episode_data_index"):
            return getattr(obj, "episode_data_index")
    if hasattr(dataset.meta, "episodes"):
        return dataset.meta.episodes
    return None

def _to_int(x):
    if hasattr(x, "item"): return int(x.item())
    return int(x)

def get_episode_frame_range(ep_index, ep_idx):
    if hasattr(ep_index, "column_names"):
        columns = set(ep_index.column_names)
        if "dataset_from_index" in columns and "dataset_to_index" in columns:
            row = ep_index[ep_idx]
            return _to_int(row["dataset_from_index"]), _to_int(row["dataset_to_index"])
    # 簡化版 fallback，適用於大多數 HuggingFace/LeRobot 格式
    if isinstance(ep_index, dict) and "from" in ep_index:
        return _to_int(ep_index["from"][ep_idx]), _to_int(ep_index["to"][ep_idx])
    
    raise ValueError("無法解析 Episode Frame Range，請確認 Dataset 結構。")

# ============================================================================
# 3. Main Steering Extraction Logic
# ============================================================================
def find_physical_height_neurons(repo_id, device=None, xml_name="follower.xml"):
    if device is None:
        device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"[*] Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, video_backend="pyav")
    
    # 載入 MuJoCo 模型
    print(f"[*] Loading MuJoCo FK Model from {xml_name}...")
    mj_model, mj_data = load_fk_model_from_local_xml(xml_name=xml_name)
    ep_index = get_episode_indices(dataset)

    # --- 步驟 A：資料驅動分析 - 找出 Z 軸最高點 ---
    print(f"[*] Pre-computing Z-height trajectories for {dataset.num_episodes} episodes...")
    ep_z_stats = []
    
    for ep_idx in tqdm(range(dataset.num_episodes), desc="Scanning Z-Axis"):
        start_frame, end_frame = get_episode_frame_range(ep_index, ep_idx)
        
        max_z = -float('inf')
        peak_frame_idx = start_frame
        
        # 尋找該 Episode 內 Z 軸最高的那一個 Frame
        for frame_idx in range(start_frame, end_frame):
            frame = dataset[frame_idx]
            state_vec = frame["observation.state"]
            z_height = compute_eef_z_height(state_vec, mj_model, mj_data)
            
            if z_height > max_z:
                max_z = z_height
                peak_frame_idx = frame_idx
                
        ep_z_stats.append({
            "ep_idx": ep_idx,
            "peak_frame_idx": peak_frame_idx,
            "max_z": max_z,
            "start_frame": start_frame,
            "end_frame": end_frame
        })

    # 根據 Z 軸高度排序，找出 Top 30 (High) 與 Bottom 30 (Low)
    ep_z_stats.sort(key=lambda x: x["max_z"], reverse=True)
    high_episodes = ep_z_stats[:30]
    low_episodes = ep_z_stats[-30:]
    
    print("\n[*] Data Splitting Complete:")
    print(f"  - Top 30 High Episodes: Avg Max Z = {np.mean([x['max_z'] for x in high_episodes]):.4f}")
    print(f"  - Bottom 30 Low Episodes: Avg Max Z = {np.mean([x['max_z'] for x in low_episodes]):.4f}")

    # --- 步驟 B：載入模型與前處理 ---
    print(f"\n[*] Loading policy config & model...")
    policy_cfg = PreTrainedConfig.from_pretrained(repo_id)
    policy_meta = copy.deepcopy(dataset.meta)
    
    rename_map = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
        "observation.images.wrist": "observation.images.camera3",
    }
    for actual, expected in rename_map.items():
        if actual in policy_meta.features:
            policy_meta.features[expected] = policy_meta.features.pop(actual)

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

    # --- 步驟 C：設定精準 Hook (限制在中後段層) ---
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    num_layers = len(text_model.layers)
    
    # 動態計算中後段層的範圍 (例如 32 層模型 -> 取 Layer 16 到 30)
    # 避開最後一層 (通常準備轉回 logits，特徵已被大幅壓縮)
    start_layer = num_layers // 2
    end_layer = num_layers - 1 
    
    print(f"\n[*] Targeting middle-to-late layers: Layer {start_layer} to {end_layer-1}")

    # 改用 dict 來儲存，避免 index 錯位
    high_sums = {i: None for i in range(start_layer, end_layer)}
    low_sums = {i: None for i in range(start_layer, end_layer)}
    high_count = 0
    low_count = 0
    captured_act = {}

    def get_forward_hook(layer_idx):
        def hook(module, inputs, output):
            # output shape: [batch, seq_len, hidden_dim]
            last_token_act = output[0, -1, :].detach().float().cpu().numpy()
            captured_act[layer_idx] = last_token_act
        return hook

    handles = []
    # 關鍵修正：只在中後段層註冊 Hook
    for i in range(start_layer, end_layer):
        h = text_model.layers[i].mlp.register_forward_hook(get_forward_hook(i))
        handles.append(h)

    # --- 步驟 D：只在關鍵幀 (Peak Frames) 執行推論 ---
    def process_frames(episode_list, is_high):
        nonlocal high_count, low_count
        
        for ep_info in tqdm(episode_list, desc="Extracting Activations"):
            peak_idx = ep_info["peak_frame_idx"]
            
            # 我們取 Peak Frame 以及其前後各 1 幀（若無越界）以增加穩定度
            target_frames = [
                idx for idx in [peak_idx - 1, peak_idx, peak_idx + 1]
                if ep_info["start_frame"] <= idx < ep_info["end_frame"]
            ]
            
            for frame_idx in target_frames:
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

    print("\n[*] Processing High Target Frames...")
    process_frames(high_episodes, is_high=True)
    
    print("\n[*] Processing Low Target Frames...")
    process_frames(low_episodes, is_high=False)

    # 移除 Hooks
    for h in handles: h.remove()

    # --- 步驟 E：計算對比分數並輸出結果 ---
    all_neurons = []
    for i in range(start_layer, end_layer):
        if high_sums[i] is None or low_sums[i] is None: continue
        avg_high = high_sums[i] / high_count
        avg_low = low_sums[i] / low_count
        
        diff = avg_high - avg_low 
        for neuron_idx, score in enumerate(diff):
            all_neurons.append({"layer": i, "neuron": neuron_idx, "score": float(score)})

    top_high = sorted(all_neurons, key=lambda x: x['score'], reverse=True)[:10]
    top_low = sorted(all_neurons, key=lambda x: x['score'])[:10]

    print("\n" + "="*70)
    print("TOP STEERING 'HIGH' NEURONS (Middle-to-Late Layers)")
    print("="*70)
    for n in top_high:
        print(f"Layer {n['layer']:>2} | Neuron/Dim {n['neuron']:>5} | Contrast Score: {n['score']:+.6f}")

    print("\n" + "="*70)
    print("TOP STEERING 'LOW' NEURONS (Middle-to-Late Layers)")
    print("="*70)
    for n in top_low:
        print(f"Layer {n['layer']:>2} | Neuron/Dim {n['neuron']:>5} | Contrast Score: {n['score']:+.6f}")

if __name__ == "__main__":
    # 確保 follower.xml 在執行路徑或同目錄下
    find_physical_height_neurons("ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2")