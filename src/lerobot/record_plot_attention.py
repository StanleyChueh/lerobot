import torch
import math
import matplotlib.pyplot as plt
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
import numpy as np
import cv2
import os
import copy  # 用於 deepcopy

# =========================
# 1. 初始化設定
# =========================
DATASET_REPO_ID = "ethanCSL/Ting_grip_box_svla" 
CKPT_PATH = "/home/bruce/CSL/lerobot_nn/outputs/train/Ting_grip_box_svla/checkpoints/020000/pretrained_model"

policy_cfg = PreTrainedConfig.from_pretrained(CKPT_PATH)

print("[INFO] Loading Dataset...")
dataset = LeRobotDataset(
    DATASET_REPO_ID,
    root=None,
    batch_encoding_size=1,
)

policy = make_policy(policy_cfg, ds_meta=dataset.meta)
device = get_safe_torch_device(policy.config.device)
policy.reset()
policy.model.vlm_with_expert.debug_attn = True

# =========================
# 2. 自動尋找有效的集數 (Length > 50)
# =========================
target_episode_idx = -1
min_length = 50

print(f"\n[INFO] Searching for an episode with length > {min_length}...")

# 遍歷前 20 集尋找
for i in range(len(dataset.meta.episodes)):
    ep_meta = dataset.meta.episodes[i]
    # 安全取得長度，避免 KeyError
    length = ep_meta.get('length', 0)
    
    print(f" - Episode {i}: length={length}")
    
    if length > min_length:
        target_episode_idx = i
        print(f" -> FOUND! Using Episode {i}")
        break

if target_episode_idx == -1:
    raise ValueError("找不到任何長度足夠的集數，請檢查 Dataset 是否損壞。")

# 載入目標集數
ep_meta = dataset.meta.episodes[target_episode_idx]
ep_length = ep_meta['length']

# 手動累加計算 start_idx (因為舊版 meta 沒有 episode_data_index)
start_idx = 0
for i in range(target_episode_idx):
    start_idx += dataset.meta.episodes[i]['length']

end_idx = start_idx + ep_length

print(f"[INFO] Processing Episode {target_episode_idx}")
print(f"       Length: {ep_length} frames")
print(f"       Global Range: {start_idx} to {end_idx}")

# =========================
# 3. 輔助函數
# =========================
PROMPTS = {
    "red":   "grip the red block and put it into box",
    "green": "grip the green block and put it into box",
    "white": "grip the white block and put it into box",
}

def find_token_index(processor, prompt, target_word):
    tokens = processor.tokenizer(prompt, return_tensors="pt")
    ids = tokens["input_ids"][0]
    words = processor.tokenizer.convert_ids_to_tokens(ids)
    for i, w in enumerate(words):
        if target_word in w:
            return i
    raise ValueError(f"Token '{target_word}' not found in {words}")

def extract_attention(policy, txt_token_idx):
    attn = policy.model.vlm_with_expert.last_attn["attn"][0]  # [H, Q, K]
    attn = attn.mean(0)  # [Q, K]
    heat = attn[txt_token_idx]  # text → all tokens
    return heat

def reshape_heat(heat):
    n = heat.numel()
    h = int(math.sqrt(n))
    while n % h != 0:
        h -= 1
    return heat.reshape(h, n // h)

def extract_raw_observation_from_step(step: dict):
    if "observation" in step and isinstance(step["observation"], dict):
        return step["observation"]
    raw = {}
    for k, v in step.items():
        if isinstance(k, str) and k.startswith("observation."):
            raw[k[len("observation."):]] = v
    if raw:
        return raw
    raise RuntimeError(f"Cannot find observation in step. keys: {list(step.keys())[:5]}")

def vector_state_to_named_dict(state_vec, dataset):
    names = dataset.features["observation.state"]["names"]
    if isinstance(state_vec, torch.Tensor):
        state_vec = state_vec.detach().cpu().numpy()
    state_vec = np.asarray(state_vec).reshape(-1)
    return {name.split("state.", 1)[-1] if False else name: float(val)
            for name, val in zip(names, state_vec)}

def get_timestep_step(episode: dict, t: int):
    # 使用 deepcopy 確保不影響原始數據
    step = {}
    for k, v in episode.items():
        if isinstance(v, dict):
            sub = {}
            for kk, vv in v.items():
                if isinstance(vv, (list, tuple, np.ndarray)) and len(vv) > t:
                    sub[kk] = vv[t]
                else:
                    sub[kk] = vv
            step[k] = sub
        elif isinstance(v, (list, tuple)) and len(v) > t:
            step[k] = v[t]
        elif isinstance(v, np.ndarray) and len(v) > t:
            step[k] = v[t]
        else:
            step[k] = v
    return copy.deepcopy(step)

# =========================
# 4. 主執行迴圈
# =========================

prompt = "grip the red block and put it into box"
processor = policy.model.vlm_with_expert.processor
txt_idx = find_token_index(processor, prompt, "red")

# 初始化 Video Writer
output_video_path = "attention_vis.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30  # 放慢速度以便觀察
video_writer = None

print(f"[INFO] Start processing Frames...")
current_frame = 0

# [修改] 改為遍歷全域索引
for global_idx in range(start_idx, end_idx):
    # 1. 取得單一幀數據 (已經是 Tensor 格式)
    item = dataset[global_idx]

    # 2. 構建模型輸入 (手動增加 Batch Dimension: C,H,W -> 1,C,H,W)
    observation_batch = {}
    for k, v in item.items():
        if k.startswith("observation."):
            # 這裡不要 replace，直接用原始的 k (例如 "observation.images.front")
            key_name = k 
            val = v
            
            # 如果是 Tensor，轉回 Numpy (解決 TypeError)
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            
            # 如果是影像 (判斷 key 含有 image 且維度是 3)
            # Dataset 讀出來是 (C, H, W)，但 predict_action 模擬真實環境需要 (H, W, C)
            if "image" in key_name and val.ndim == 3:
                val = np.transpose(val, (1, 2, 0)) # CHW -> HWC
                
                # [重要] 還原數值範圍 [0, 1] -> [0, 255]
                if val.max() <= 1.5: 
                    val = (val * 255).astype(np.uint8)
            
            observation_batch[key_name] = val

    # 3. 執行預測 (加上 no_grad 節省記憶體)
    with torch.no_grad():
        _ = predict_action(
            observation_batch,
            policy,
            device,
            policy.config.use_amp,
            task=prompt,
            robot_type=None,
        )

    # 4. 提取 Attention (跟原本一樣)
    heat = extract_attention(policy, txt_idx)
    heat_2d = reshape_heat(heat).cpu().numpy()

    # 5. [修改] 取得雙鏡頭 RGB 原圖 (Front & Top)
    # 取得 Front Image
    rgb_front = item["observation.images.front"].permute(1, 2, 0).cpu().numpy()
    
    # 取得 Top Image (假設 Dataset 有這個 key，根據 Log 應該有)
    if "observation.images.top" in item:
        rgb_top = item["observation.images.top"].permute(1, 2, 0).cpu().numpy()
    else:
        # 如果沒有 Top，就用全黑代替，避免報錯
        rgb_top = np.zeros_like(rgb_front)

    # 數值還原 [0, 1] -> [0, 255]
    if rgb_front.max() <= 1.5:
        rgb_front = (rgb_front * 255).astype(np.uint8)
        rgb_top = (rgb_top * 255).astype(np.uint8)
    else:
        rgb_front = rgb_front.astype(np.uint8)
        rgb_top = rgb_top.astype(np.uint8)

    # [關鍵步驟] 將兩張圖合併 (Side-by-Side)
    # 形狀變為 (480, 1280, 3)
    combined_rgb = np.hstack((rgb_front, rgb_top))

    # ---- 視覺化 Attention on Combined Image ----
    
    # 將 Attention Map Resize 到「合併後」的大圖尺寸
    # 這樣如果 Attention 分佈在兩張圖上，就會正確顯示在各自的位置
    heat_img = cv2.resize(heat_2d, (combined_rgb.shape[1], combined_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Normalize Heatmap
    heat_min, heat_max = heat_img.min(), heat_img.max()
    if heat_max - heat_min > 1e-6:
        heat_norm = (heat_img - heat_min) / (heat_max - heat_min)
    else:
        heat_norm = heat_img

    # Overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_JET)
    rgb_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
    
    # 疊合 (0.6 原圖 + 0.4 Heatmap)
    overlay = cv2.addWeighted(rgb_bgr, 0.6, heatmap_color, 0.4, 0)
    
    # 畫一條分隔線 (Optional)
    h, w, _ = rgb_front.shape
    cv2.line(overlay, (w, 0), (w, h), (255, 255, 255), 2)

    # 寫入影片
    if video_writer is None:
        H, W = overlay.shape[:2]
        print(f"[INFO] Initializing Side-by-Side Video: {W}x{H} @ {fps}fps")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    
    if current_frame % 30 == 0:
        print(f"Processing Frame {current_frame}/{ep_length}")

    video_writer.write(overlay)
    current_frame += 1

if video_writer is not None:
    video_writer.release()
    print(f"\n[DONE] Saved to {output_video_path}")
else:
    print("\n[WARN] Failed to create video.")