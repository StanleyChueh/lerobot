'''
 python src/lerobot/scripts/record_attention_plot_cross_stanley.py     --repo_id "ethanCSL/svla_koch_sorting_n_stacking"     --ckpt "ethanCSL/svla_koch_sorting_n_stacking"     --episode 0     --prompt "Put the red cube in the right box,the green cube in the left box." --rename_map='{                                                      
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

# LeRobot tools
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

import matplotlib.pyplot as plt

def extract_cross_attention_maps(model):

    if not hasattr(model, "last_attn_weights") or model.last_attn_weights is None:
        print("[錯誤] 模型中找不到 last_attn_weights。")
        return None, None

    # [B, Heads, Query_Len, Key_Len]
    attn = model.last_attn_weights

    # 1️⃣ 平均所有 head
    attn_mean_heads = attn.mean(dim=1)   # [B, Q, K]

    # 2️⃣ 取 batch 0
    attn_mean_heads = attn_mean_heads[0] # [Q, K]

    # 3️⃣ 平均所有 action tokens (Query 維度)
    mean_action_attn = attn_mean_heads[0]
    print("Cross K length:", mean_action_attn.shape[0])

    # 4️⃣ 自動取得 image token 數量
    num_img_tokens = model._debug_num_img_tokens

    heat_cam1_1d = mean_action_attn[:num_img_tokens]
    heat_cam2_1d = mean_action_attn[num_img_tokens:2*num_img_tokens]

    print("sum cam1 =", float(mean_action_attn[:num_img_tokens].sum()))
    print("sum cam2 =", float(mean_action_attn[num_img_tokens:2*num_img_tokens].sum()))
    print("sum rest =", float(mean_action_attn[2*num_img_tokens:].sum()))

    print(f"DEBUG: max={mean_action_attn.max()}, min={mean_action_attn.min()}")

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
        
        if hasattr(policy.model.vlm_with_expert, "last_attn_weights"):
            attn = policy.model.vlm_with_expert.last_attn_weights
            print("last_attn_weights shape:", attn.shape)
        
        # ===== 正確 debug：用 policy.prepare_images 取得 VLM 真正吃的 pixel_values =====
        images_list, img_masks = policy.prepare_images(batch_pp)   # images_list: list[tensor]，通常每個 camera 一個
        img_emb0 = policy.model.vlm_with_expert.embed_image(images_list[0])
        print("DEBUG embed_image(images_list[0]) shape:", img_emb0.shape)  # [B, num_img_tokens, hidden]

        # 將 num_img_tokens 存起來給 extract_cross_attention_maps 用
        policy.model.vlm_with_expert._debug_num_img_tokens = int(img_emb0.shape[1])

        h1_1d, h2_1d = extract_cross_attention_maps(policy.model.vlm_with_expert)

        if hasattr(policy.model.vlm_with_expert, "last_attn_weights"):

            # ===== Block-level Cross Attention Statistics =====
            attn = policy.model.vlm_with_expert.last_attn_weights

            # mean over heads
            attn_m = attn.mean(dim=1)[0]   # [Q, K]

            Q, K = attn_m.shape

            # ---- SAFE prefix split (only Image vs Rest) ----
            num_img_tokens = policy.model.vlm_with_expert._debug_num_img_tokens

            # ---- SAFE prefix split (only Image vs Rest) ----
            num_img_tokens = policy.model.vlm_with_expert._debug_num_img_tokens

            img_start = 0
            img_end = 2 * num_img_tokens

            rest_start = img_end
            rest_end = K

            action_to_img = attn_m[:, img_start:img_end].mean().item()

            if rest_end > rest_start:
                action_to_rest = attn_m[:, rest_start:rest_end].mean().item()
            else:
                action_to_rest = 0.0

            print("\n===== Cross Attention (Safe Split) =====")
            print(f"Action → Image : {action_to_img:.6f}")
            print(f"Action → Rest  : {action_to_rest:.6f}")
            print("========================================\n")

            img_start = 0
            img_end = 2 * num_img_tokens

            lang_start = img_end
            lang_end = lang_start + num_lang_tokens

            state_start = lang_end
            state_end = K

            # ---- compute means ----
            action_to_img = attn_m[:, img_start:img_end].max().item()
            action_to_lang = attn_m[:, lang_start:lang_end].max().item() if lang_end > lang_start else 0.0
            action_to_state = 0.0

            print("\n===== Cross Attention Block Mean =====")
            print(f"Action → Image   : {action_to_img:.6f}")
            print(f"Action → Language: {action_to_lang:.6f}")
            print(f"Action → State   : {action_to_state:.6f}")
            print("======================================\n")

            # mean over heads
            attn_m = attn.mean(dim=1)[0]   # [Q, K]

            plt.figure(figsize=(6,6))
            plt.imshow(
                attn_m.detach().cpu().numpy(),
                cmap="viridis",
                vmin=0,
                vmax=attn_m.max().item()
            )

            plt.colorbar()
            plt.title("Cross Attention Matrix (Action → Prefix)")
            plt.xlabel("Key (Image | Language | State)")
            plt.ylabel("Query (Action tokens)")
            plt.tight_layout()

            if (i - start_idx) % 30 == 0:
                save_path = os.path.join(
                    block_dir,
                    f"cross_block_frame_{i - start_idx}.png"
                )
                plt.savefig(save_path)

            plt.close()

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