'''
 python src/lerobot/scripts/record_attention_plot_cross.py     --repo_id "ethanCSL/svla_koch_sorting_n_stacking"     --ckpt "ethanCSL/svla_koch_sorting_n_stacking"     --episode 0     --prompt "Put the green cube in the box." --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.top":   "observation.images.camera2"
  }' 

'''
import os
import math
import json
import torch
import cv2
import numpy as np
import argparse
from torch import nn

# LeRobot 相關工具
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

# ==========================================
# 1. 核心提取與處理函式
# ==========================================

def extract_cross_attention_maps(model):
    """
    從 SmolVLMWithExpertModel 提取最後一層 Cross Attention 權重
    並針對 Action Tokens 取平均以獲得更穩定的注意力分佈
    """
    if not hasattr(model, "last_attn_weights") or model.last_attn_weights is None:
        print("[錯誤] 模型中找不到 last_attn_weights。")
        return None, None

    # attn_weights 維度: [Batch, Heads, Query_Len, Key_Len]
    # SmolVLA 在推理時，Query_Len 通常等於預測的 Action Chunk Size
    # attn_weights = model.last_attn_weights
    
    # avg_attn = attn_weights.mean(dim=1)[0]  

    # mean_action_attn = avg_attn.mean(dim=0) 
   # 修改 extract_cross_attention_maps
    # 指定看第 0 個 Head，通常某些 Head 會專門負責空間座標
    head_idx = 0 
    attn_weights = model.last_attn_weights[0, head_idx, :, :]
    current_action_attn = attn_weights[0, :]

    num_img_tokens = 152

    # 3. 動態分離影像 Token
    # 注意：Key_Len 可能包含 Prompt 文字 token，這裡我們假設影像 token 排在前面
    # total_tokens = mean_action_attn.numel()
    
    # # 根據你的模型配置，通常會平分給兩個相機
    # num_cam = 2
    # tokens_per_cam = total_tokens // num_cam
    
    # heat_cam1_1d = mean_action_attn[:tokens_per_cam]
    # heat_cam2_1d = mean_action_attn[tokens_per_cam : 2*tokens_per_cam]
    heat_cam1_1d = current_action_attn[:num_img_tokens]
    heat_cam2_1d = current_action_attn[num_img_tokens : 2 * num_img_tokens]
    
    # 在 extract 函式中加入 print
    print(f"DEBUG: attn_weights max: {attn_weights.max()}, min: {attn_weights.min()}")
    # 如果不同模型的 max/min 數值極度接近（小數點後 5 位都一樣），
    # 很大機率是你在 forward 過程中拿到了同一份快取資料。
    return heat_cam1_1d, heat_cam2_1d

# def process_heatmap(heat_1d, original_size=(480, 640)):
#     heat_1d = heat_1d.float().detach().cpu()
#     num_tokens = heat_1d.numel()
    
#     # 動態尋找網格大小 (例如 152 -> 8x19)
#     h_grid = int(math.sqrt(num_tokens / 2)) 
#     if h_grid == 0: h_grid = 1
#     w_grid = num_tokens // h_grid
    
#     if h_grid * w_grid != num_tokens:
#         h_grid = int(math.sqrt(num_tokens))
#         w_grid = num_tokens // h_grid
#         heat_1d = heat_1d[:h_grid * w_grid]

#     heat_2d = heat_1d.reshape(h_grid, w_grid).numpy()
#     heat_resized = cv2.resize(heat_2d, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
#     v_min, v_max = np.percentile(heat_resized, [0, 98])
#     heat_norm = np.clip((heat_resized - v_min) / (v_max - v_min + 1e-6), 0, 1)
#     return heat_norm
def process_heatmap(heat_1d, original_size=(480, 640)):
    heat_1d = heat_1d.float().detach().cpu()
    
    # --- 修改點 3: 固定物理網格尺寸 ---
    # 移除動態 sqrt 計算，直接給予對應模型架構的佈局
    # 對於 SmolVLA/SmolVLM2 (384 tokens): 16x24
    h_grid, w_grid = 8, 19 
    
    try:
        # 這裡 reshape 要確保數量符合 16*24=384
        heat_2d = heat_1d.reshape(h_grid, w_grid).numpy()
    except Exception as e:
        print(f"[警告] Token 數量不匹配，退回自動計算。錯誤: {e}")
        side = int(math.sqrt(heat_1d.numel()))
        heat_2d = heat_1d[:side*side].reshape(side, side).numpy()

    # 插值放大回原始圖片大小
    heat_resized = cv2.resize(heat_2d, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # 保持百分比歸一化，這能濾除極端離群點，讓熱力圖顏色更鮮明
    v_min, v_max = np.percentile(heat_resized, [0, 98])
    heat_norm = np.clip((heat_resized - v_min) / (v_max - v_min + 1e-6), 0, 1)
    return heat_norm

# ==========================================
# 2. 主執行邏輯
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="Put the green cube in the box.")
    parser.add_argument("--output_path", type=str, default="attention_video.mp4")
    parser.add_argument("--rename_map", type=str, default='{"observation.images.front": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}')
    args = parser.parse_args()

    device = get_safe_torch_device("cuda")
    rename_map = json.loads(args.rename_map)

    dataset = LeRobotDataset(args.repo_id, batch_encoding_size=1)
    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=rename_map)
    policy.to(device)
    policy.eval()
    policy.model.vlm_with_expert.attention_mode = "cross_attn"
    # 強制模型只跑第一層
    policy.model.vlm_with_expert.num_vlm_layers = 1

    preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset.meta.stats)

    if hasattr(dataset.meta, 'episode_data_index'):
        start_idx = dataset.meta.episode_data_index['from'][args.episode]
        end_idx = dataset.meta.episode_data_index['to'][args.episode]
    else:
        start_idx = sum(dataset.meta.episodes[i]['length'] for i in range(args.episode))
        end_idx = start_idx + dataset.meta.episodes[args.episode]['length']

    print(f"[開始] 處理第 {args.episode} 集，影片將儲存至: {args.output_path}")

    video_writer = None

    for i in range(start_idx, end_idx):
        item = dataset[i]
        observation_batch = {}
        for k, v in item.items():
            if k in rename_map:
                observation_batch[rename_map[k]] = v.unsqueeze(0).to(device)
            elif k.startswith("observation."):
                observation_batch[k] = v.unsqueeze(0).to(device)
        
        ref_cam = observation_batch["observation.images.camera1"]
        for tk in ["observation.images.camera3", "observation.images.empty_camera_0"]:
            if tk not in observation_batch:
                observation_batch[tk] = torch.zeros_like(ref_cam)

        observation_batch["task"] = args.prompt
        batch_pp = preprocessor(observation_batch)
        with torch.no_grad():
            policy.predict_action_chunk(batch_pp)

        h1_1d, h2_1d = extract_cross_attention_maps(policy.model.vlm_with_expert)
        
        if h1_1d is not None:
            img_front = item["observation.images.front"].permute(1, 2, 0).numpy()
            img_top = item["observation.images.top"].permute(1, 2, 0).numpy()
            
            if img_front.max() <= 1.0: img_front = (img_front * 255).astype(np.uint8)
            if img_top.max() <= 1.0: img_top = (img_top * 255).astype(np.uint8)

            mask_f = process_heatmap(h1_1d)
            mask_t = process_heatmap(h2_1d)

            heatmap_f = cv2.applyColorMap(np.uint8(255 * mask_f), cv2.COLORMAP_JET)
            heatmap_t = cv2.applyColorMap(np.uint8(255 * mask_t), cv2.COLORMAP_JET)

            vis_f = cv2.addWeighted(cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR), 0.6, heatmap_f, 0.4, 0)
            vis_t = cv2.addWeighted(cv2.cvtColor(img_top, cv2.COLOR_RGB2BGR), 0.6, heatmap_t, 0.4, 0)

            combined = np.hstack((vis_f, vis_t))

            # 初始化 VideoWriter
            if video_writer is None:
                height, width, _ = combined.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(args.output_path, fourcc, 15.0, (width, height))

            video_writer.write(combined)
            
            if i % 50 == 0:
                print(f"已處理幀數: {i - start_idx}")

    if video_writer:
        video_writer.release()
    print("[結束] 影片製作完成。")

if __name__ == "__main__":
    main()