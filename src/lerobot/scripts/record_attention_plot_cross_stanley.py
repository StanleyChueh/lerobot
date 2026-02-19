'''
Usage:
python src/lerobot/scripts/record_attention_plot_cross_stanley.py     --repo_id "ethanCSL/svla_koch_sorting_n_stacking"     --ckpt "ethanCSL/svla_koch_sorting_n_stacking"     --episode 0     --prompt "Put the red cube in the right box,the green cube in the left box." --rename_map='{                                                      
    "observation.images.front": "observation.images.camera1",
    "observation.images.top":   "observation.images.camera2"
}' 
'''

'''
Cross attention block in action expert 

'''

import os
import math
import json
import torch
import cv2
import numpy as np
import argparse
from torch import nn

# LeRobot tools
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

# plot tool
import matplotlib.pyplot as plt

def extract_cross_attention_maps(attn_matrix, num_img_tokens, t=0):
    """
    attn_matrix: [Q, K] (已經 mean over heads & batch)
    t: 取哪一個 action query token 的 attention。要跟當下動作同步，通常用 t=0。
    """
    Q, K = attn_matrix.shape
    t = max(0, min(t, Q - 1))  # clamp

    attn_1d = attn_matrix[t]   # [K]

    if 2 * num_img_tokens > K:
        print(f"[錯誤] 2*num_img_tokens={2*num_img_tokens} > K={K}，切分一定錯。")
        return None, None

    heat_cam1_1d = attn_1d[:num_img_tokens]
    heat_cam2_1d = attn_1d[num_img_tokens:2 * num_img_tokens]
    return heat_cam1_1d, heat_cam2_1d

def process_heatmap(heat_1d, original_size=(480, 640)):

    heat_1d = heat_1d.float().detach().cpu()

    num_tokens = heat_1d.numel()

    # 自動推測 grid 為正方形（最安全）
    side = int(math.sqrt(num_tokens))

    if side * side != num_tokens:
        # 如果不是完美平方，找最近可整除排列
        for h in range(side, 0, -1):
            if num_tokens % h == 0:
                w = num_tokens // h
                heat_2d = heat_1d.reshape(h, w).numpy()
                break
    else:
        heat_2d = heat_1d.reshape(side, side).numpy()

    heat_resized = cv2.resize(
        heat_2d,
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )

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
    parser.add_argument("--output_path", type=str, default="attention_video_cross_stanley.mp4")
    parser.add_argument("--rename_map", type=str, default='{"observation.images.front": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}')
    parser.add_argument("--video_backend", type=str, default="pyav")
    args = parser.parse_args()

    device = get_safe_torch_device("cuda")
    rename_map = json.loads(args.rename_map)

    dataset = LeRobotDataset(args.repo_id, batch_encoding_size=1,video_backend=args.video_backend)
    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=rename_map)
    policy.to(device)
    policy.eval()

    # cross attention
    policy.model.vlm_with_expert.attention_mode = "cross_attn"

    preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset.meta.stats)

    if hasattr(dataset.meta, 'episode_data_index'):
        start_idx = dataset.meta.episode_data_index['from'][args.episode]
        end_idx = dataset.meta.episode_data_index['to'][args.episode]
    else:
        start_idx = sum(dataset.meta.episodes[i]['length'] for i in range(args.episode))
        end_idx = start_idx + dataset.meta.episodes[args.episode]['length']

    print(f"[開始] 處理第 {args.episode} 集，影片將儲存至: {args.output_path}")

    block_dir = "cross_attn_block_every30"
    os.makedirs(block_dir, exist_ok=True)

    video_writer = None

    for i in range(start_idx, end_idx):
        if i % 4 != 0:
            continue

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

        # ---- Enable cross-attention recording ----
        model = policy.model.vlm_with_expert
        model.record_attn = True
        model.attn_records = {}

        with torch.no_grad():
            policy.predict_action_chunk(batch_pp)

        if hasattr(policy.model.vlm_with_expert, "last_attn_weights"):
            attn = policy.model.vlm_with_expert.last_attn_weights
            # print("last_attn_weights shape:", attn.shape)

        model = policy.model.vlm_with_expert

        layer_ids = [k[0] for k in model.attn_records.keys() if k[1] == "expert_cross"]
        if len(layer_ids) == 0:
            continue

        final_layer = max(layer_ids)
        attn_list = model.attn_records.get((final_layer, "expert_cross"), [])
        if len(attn_list) == 0:
            continue

        attn = attn_list[-1]

        attn_matrix = attn.mean(dim=1)[0]   # [Q, K]

        t = 0

        num_img_tokens = policy.model.vlm_with_expert._debug_num_img_tokens

        h1_1d, h2_1d = extract_cross_attention_maps(
            attn_matrix,
            num_img_tokens
        )

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