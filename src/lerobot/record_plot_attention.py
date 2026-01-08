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
#DATASET_REPO_ID = "ethanCSL/color_test_green"
CKPT_PATH = "/home/bruce/CSL/lerobot_nn/outputs/train/Ting_grip_box_svla/checkpoints/020000/pretrained_model"
#CKPT_PATH = "/home/bruce/CSL/lerobot_nn/model_test/koch/svla_color_complex/checkpoints/020000/pretrained_model"

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
# 2. 指定特定的集數 (Specify Episode Index)
# =========================

# [MODIFICATION] Change this number to the episode you want to watch
target_episode_idx = 40  

# Check if the index is valid
total_episodes = len(dataset.meta.episodes)
if target_episode_idx < 0 or target_episode_idx >= total_episodes:
    raise ValueError(f"Episode index {target_episode_idx} is out of range (0 to {total_episodes-1})")

# Load metadata for this specific episode
ep_meta = dataset.meta.episodes[target_episode_idx]
ep_length = ep_meta['length']

# Calculate global start index by summing lengths of all previous episodes
# (This is required because LeRobotDataset flattens all frames)
start_idx = 0
for i in range(target_episode_idx):
    start_idx += dataset.meta.episodes[i]['length']

end_idx = start_idx + ep_length

print(f"\n[INFO] Processing Specific Episode: {target_episode_idx}")
print(f"       Total Episodes available: {total_episodes}")
print(f"       Length: {ep_length} frames")
print(f"       Global Range: {start_idx} to {end_idx}")


# =========================
# 3. 輔助函數
# =========================

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

    # 4. 提取 Attention
    heat = extract_attention(policy, txt_idx)
    
    # ==========================================
    # [修正核心] 解決垂直條紋與分割問題
    # ==========================================
    
    # 1. 計算視覺 Tokens 總數
    # SmolVLM 的 Attention 通常包含 Text + Image。
    # 視覺 Tokens 數量通常非常大 (例如 2400)，而 Text 很少。
    # 我們假設 Attention 向量中，主要的長度都是視覺 Tokens。
    num_total_tokens = heat.numel()
    
    # 2. 強制對半切分 (Front / Top)
    # 因為兩張圖輸入大小一樣 (480x640)，模型產生的 Token 數一定一樣。
    tokens_per_image = num_total_tokens // 2
    
    heat_front_1d = heat[:tokens_per_image]
    heat_top_1d = heat[tokens_per_image:]

    # 3. [關鍵演算法] 強制依據 4:3 比例重塑 (修復垂直條紋)
    def reshape_to_4_3_ratio(heat_tensor):
        n = heat_tensor.numel()
        if n == 0: return np.zeros((1, 1))
        
        # 數學推導：h * w ≈ n 且 w / h ≈ 1.33 (4:3)
        # h * 1.33h ≈ n  =>  h ≈ sqrt(n / 1.33)
        h_est = int(math.sqrt(n * 0.75))
        
        # 為了保險，我們從估計值開始往下找，直到找到一個合理的矩形
        # 這裡我們放寬標準：只要丟棄的 tokens 不超過 20% 就可以接受
        best_h = 1
        best_w = n
        min_waste = n
        
        # 搜尋範圍：從 h_est 到 1
        for h in range(h_est, 0, -1):
            w = int(h * 4 / 3) # 強制 4:3 比例
            if w == 0: w = 1
            
            used = h * w
            if used <= n:
                waste = n - used
                if waste < min_waste:
                    min_waste = waste
                    best_h = h
                    best_w = w
                
                # 如果浪費很少 (例如少於 10 個 token)，就直接採用
                if waste < 15: 
                    break
        
        # [核心修正] 自動裁切多餘的 tokens
        # 例如：輸入 89，最佳形狀 9x9 (81)，我們只取 heat_tensor[:81]
        valid_len = best_h * best_w
        heat_trimmed = heat_tensor[:valid_len]
        
        return heat_trimmed.reshape(best_h, best_w).cpu().numpy()

    # 重塑為 2D (Height, Width)
    heat_2d_front = reshape_to_4_3_ratio(heat_front_1d)
    heat_2d_top = reshape_to_4_3_ratio(heat_top_1d)

    # 5. 取得雙鏡頭 RGB 原圖
    rgb_front = item["observation.images.front"].permute(1, 2, 0).cpu().numpy()
    
    if "observation.images.top" in item:
        rgb_top = item["observation.images.top"].permute(1, 2, 0).cpu().numpy()
    else:
        rgb_top = np.zeros_like(rgb_front)

    # 數值還原 [0, 1] -> [0, 255]
    if rgb_front.max() <= 1.5:
        rgb_front = (rgb_front * 255).astype(np.uint8)
        rgb_top = (rgb_top * 255).astype(np.uint8)
    else:
        rgb_front = rgb_front.astype(np.uint8)
        rgb_top = rgb_top.astype(np.uint8)

    # ---- 視覺化疊合 ----
    
    def apply_heatmap_overlay(rgb_img, heat_map_2d):
        # Resize 到跟原圖一樣大 (使用 Cubic 插值會比較平滑)
        heat_resized = cv2.resize(heat_map_2d, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Normalize 到 0-255
        h_min, h_max = heat_resized.min(), heat_resized.max()
        if h_max - h_min > 1e-6:
            heat_norm = (heat_resized - h_min) / (h_max - h_min)
        else:
            heat_norm = heat_resized
            
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_JET)
        
        # 轉成 BGR 給 OpenCV 寫入影片
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        
        # 疊合
        return cv2.addWeighted(rgb_bgr, 0.6, heatmap_color, 0.4, 0)

    # 分別疊加
    vis_front = apply_heatmap_overlay(rgb_front, heat_2d_front)
    vis_top = apply_heatmap_overlay(rgb_top, heat_2d_top)

    # 左右合併 (Side-by-Side)
    overlay = np.hstack((vis_front, vis_top))
    
    # 畫中間分隔線 (白色)
    h, w, _ = vis_front.shape
    cv2.line(overlay, (w, 0), (w, h), (255, 255, 255), 2)
    
    # 標示文字 (Front / Top)
    cv2.putText(overlay, "Front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, "Top", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 寫入影片
    if video_writer is None:
        H_out, W_out = overlay.shape[:2]
        print(f"[INFO] Initializing Side-by-Side Video: {W_out}x{H_out} @ {fps}fps")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W_out, H_out))
    
    if current_frame % 30 == 0:
        print(f"Processing Frame {current_frame}/{ep_length}")

    video_writer.write(overlay)
    current_frame += 1

# 迴圈結束
if video_writer is not None:
    video_writer.release()
    print(f"\n[DONE] Saved to {os.path.abspath(output_video_path)}")
else:
    print("\n[WARN] Failed to create video.")