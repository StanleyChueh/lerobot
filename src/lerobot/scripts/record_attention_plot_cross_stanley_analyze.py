'''
Usage:
python src/lerobot/scripts/record_attention_plot_cross_stanley_analyze.py     --repo_id "ethanCSL/svla_koch_sorting_n_stacking"     --ckpt "ethanCSL/svla_koch_sorting_n_stacking"     --episode 0     --prompt "Put the red cube in the right box,the green cube in the left box." --rename_map='{                                                      
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

def extract_cross_attention_maps(attn_matrix, num_img_tokens, real_text_len):
    mean_action_attn = attn_matrix.mean(dim=0)
    
    # 1. 定義有效區間 (Vision + Real Text)
    vision_end = 2 * num_img_tokens
    text_end = vision_end + real_text_len
    
    # 2. 提取有效部分的權重
    # 我們只想要視覺部分，但我們要用「視覺+文字」的總合來重新歸一化
    # 這樣如果文字權重變大，影像 Heatmap 就會變淡；如果文字沒用，影像就會變亮
    valid_total_attn = mean_action_attn[:text_end].sum() + 1e-8
    
    # 3. 提取影像 1d 權重並重新縮放
    # 這裡的邏輯：排除 Padding 後，影像在「有效資訊」中所佔的真實強度
    heat_cam1_1d = mean_action_attn[:num_img_tokens] / valid_total_attn
    heat_cam2_1d = mean_action_attn[num_img_tokens:vision_end] / valid_total_attn

    text_weights = mean_action_attn[vision_end:text_end].cpu().tolist()
    print(f"文字 Token 權重分佈: {text_weights}")

    return heat_cam1_1d, heat_cam2_1d, text_weights

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

def draw_prompt_weights_aligned(img, prompt, weights):
    h, w, _ = img.shape
    text_bar_height = 100
    text_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
    
    # 1. 計算文字佔用的寬度
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(prompt, font, font_scale, thickness)
    
    start_x = 20
    start_y = 40
    
    # 2. 繪製文字
    cv2.putText(text_bar, f"Prompt: {prompt}", (start_x, start_y), 
                font, font_scale, (50, 50, 50), thickness)

    # 3. 繪製精確對齊的權重條
    if len(weights) > 0:
        w_array = np.array(weights)
        # 歸一化
        norm_w = (w_array - w_array.min()) / (w_array.max() - w_array.min() + 1e-8)
        
        # 我們將權重條的寬度設定為與文字長度一致
        bar_width = text_w + 100 # 預留一點緩衝
        bar_y = start_y + 15
        
        # 建立一個小型的權重條並放大
        small_bar = (norm_w.reshape(1, -1) * 255).astype(np.uint8)
        color_bar = cv2.applyColorMap(cv2.resize(small_bar, (bar_width, 15)), cv2.COLORMAP_JET)
        
        # 貼上權重條
        text_bar[bar_y:bar_y+15, start_x:start_x+bar_width] = color_bar
        
        # 標註說明
        cv2.putText(text_bar, "Low", (start_x, bar_y + 30), font, 0.4, (100, 0, 0), 1)
        cv2.putText(text_bar, "High", (start_x + bar_width - 30, bar_y + 30), font, 0.4, (0, 0, 100), 1)
    
    return np.vstack((img, text_bar))

def analyze_attention_refined(attn_matrix, num_img_tokens, real_text_len):
    mean_attn = attn_matrix.mean(dim=0).float().cpu()
    total_k = mean_attn.shape[0]
    
    # 1. 影像區 (前 128)
    vision_attn = mean_attn[:2*num_img_tokens].sum().item()
    
    # 2. 真實文字區 (128 開始，往後數實質長度)
    text_start = 2 * num_img_tokens
    text_end = text_start + real_text_len
    text_attn = mean_attn[text_start:text_end].sum().item()
    
    # 3. 機器人狀態區 (末端 30 token，根據你的圖表 K=305-275)
    state_attn = mean_attn[275:].sum().item()
    
    # 4. 其他 (Padding)
    others_attn = mean_attn[text_end:275].sum().item()
    
    total = vision_attn + text_attn + state_attn + others_attn + 1e-8
    
    return {
        "Vision": vision_attn / total,
        "Text": text_attn / total,
        "State": state_attn / total,
        "Padding": others_attn / total
    }

# ==========================================
# 2. 主執行邏輯
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="Put the green cube in the box.")
    parser.add_argument("--output_path", type=str, default="attention_video_cross_stanley_analyze.mp4")
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
    
    print("\n" + "="*30)
    print("Preprocessor 處理順序 (Pipeline Steps):")
    for i, step in enumerate(preprocessor.steps):
        print(f"步驟 {i+1}: {type(step).__name__}")
        # 如果是 Tokenizer 步驟，額外印出參數
        if "Tokenizer" in type(step).__name__:
            print(f"   -> Max Length: {step.max_length}")
            print(f"   -> Padding: {step.padding}")
    print("="*30 + "\n")

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
        
        # 提取文字 Token 的實際長度 (排除 Padding)
        # SmolVLA 的 input_ids 通常在 batch_pp 中
        text_token_len = 0
        mask_key = "observation.language.attention_mask"
        token_key = "observation.language.tokens"

        if mask_key in batch_pp:
            # 計算非 padding 的 token 數量
            text_token_len = batch_pp[mask_key].sum().item()
        elif token_key in batch_pp:
            text_token_len = batch_pp[token_key].shape[1]
        

        # ---- Enable cross-attention recording ----
        model = policy.model.vlm_with_expert
        model.record_attn = True
        model.attn_records = {}

        with torch.no_grad():
            policy.predict_action_chunk(batch_pp)

        if hasattr(policy.model.vlm_with_expert, "last_attn_weights"):
            attn = policy.model.vlm_with_expert.last_attn_weights
            # print("last_attn_weights shape:", attn.shape)
        
        images_list, img_masks = policy.prepare_images(batch_pp)   # images_list: list[tensor]，通常每個 camera 一個
        img_emb0 = policy.model.vlm_with_expert.embed_image(images_list[0])
        #print("DEBUG embed_image(images_list[0]) shape:", img_emb0.shape)  # [B, num_img_tokens, hidden]

        # 將 num_img_tokens 存起來給 extract_cross_attention_maps 用
        policy.model.vlm_with_expert._debug_num_img_tokens = int(img_emb0.shape[1])

        model = policy.model.vlm_with_expert

        layer_ids = [k[0] for k in model.attn_records.keys() if k[1] == "expert_cross"]
        print("ALL attn keys:", model.attn_records.keys())
        
        print("a")
        if len(layer_ids) == 0:
            print("b")
            continue
        print("c")
        final_layer = max(layer_ids)
        attn_list = model.attn_records.get((final_layer, "expert_cross"), [])
        print("attn_records:", model.attn_records.keys())
        print("layer_ids:", layer_ids)
        print("attn_list len:", len(attn_list))
        if len(attn_list) == 0:
            continue

        attn = attn_list[-1]

        attn_matrix = attn.mean(dim=1)[0]   # [Q, K]
        num_img_tokens = policy.model.vlm_with_expert._debug_num_img_tokens
        # 插入在 attn_matrix 定義之後
        if i == start_idx:
            print("\n" + "="*50)
            print("序列順序診斷 (Sequence Order Diagnosis)")
            
            # 1. 取得基本維度
            total_k_tokens = attn_matrix.shape[1]
            mean_attn_values = attn_matrix.mean(dim=0).cpu().numpy()
            
            # 2. 獲取文字遮罩的實際長度
            mask_key = "observation.language.attention_mask"
            real_text_len = 0
            if mask_key in batch_pp:
                real_text_len = int(batch_pp[mask_key].sum().item())
                total_text_config_len = batch_pp[mask_key].shape[1]
            
            print(f"總 Token 數 (K): {total_k_tokens}")
            print(f"預期影像 Token 數 (2機): {2 * num_img_tokens}")
            print(f"文字有效長度 (Mask Sum): {real_text_len}")
            
            # 3. 尋找能量峰值位置 (通常影像區域能量較集中)
            # 我們切分前、中、後段來觀察平均權重
            head_segment = mean_attn_values[:2*num_img_tokens].mean()
            tail_segment = mean_attn_values[2*num_img_tokens:].mean()
            
            print(f"前段 (影像區間) 平均權重: {head_segment:.6f}")
            print(f"後段 (文字/其他區間) 平均權重: {tail_segment:.6f}")
            
            # 4. 判斷邏輯
            if head_segment > tail_segment:
                print(">>> 診斷結果: 權重集中在前段，符合 [Vision, Text] 順序。")
            else:
                print(">>> 診斷結果: 權重集中在後段，可能順序為 [Text, Vision] 或文字佔主導。")
            
            # 5. 輸出視覺化分佈圖 (Debug 用)
            plt.figure(figsize=(10, 4))
            plt.plot(mean_attn_values)
            plt.axvline(x=2*num_img_tokens, color='r', linestyle='--', label='Image-Text Split')

            text_start = 2 * num_img_tokens
            text_end = text_start + real_text_len
            plt.axvline(x=text_end, color='g', linestyle='--', label='Text-State Split')
                
            plt.title("Attention Weight Distribution across Tokens")
            plt.xlabel("Token Index")
            plt.ylabel("Mean Attention Value")
            plt.legend()
            plt.savefig("token_distribution_debug.png")
            print("已儲存 Token 分佈圖至: token_distribution_debug.png")
            print("="*50 + "\n")

        num_img_tokens = policy.model.vlm_with_expert._debug_num_img_tokens

        # 1. 呼叫函式並接收字典結果
        attn_results = analyze_attention_refined(
            attn_matrix, 
            num_img_tokens, 
            real_text_len=int(text_token_len)
        )

        # 2. 修改 print 語法，從字典中提取數據
        print(
            f"Frame {i}: "
            f"Vision:{attn_results['Vision']:.1%} | "
            f"Text:{attn_results['Text']:.1%} | "
            f"State:{attn_results['State']:.1%} | "
            f"Padding:{attn_results['Padding']:.1%}"
        )

        h1_1d, h2_1d , text_weights_list = extract_cross_attention_maps(
            attn_matrix,
            num_img_tokens,
            real_text_len=int(text_token_len)
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

            combined_with_text = draw_prompt_weights_aligned(
                combined, 
                args.prompt, 
                text_weights_list
            )

            # 3. 寫入影片 (注意高度增加了 80)
            if video_writer is None:
                fh, fw, _ = combined_with_text.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(args.output_path, fourcc, 15.0, (fw, fh))

            video_writer.write(combined_with_text)
            
            if i % 50 == 0:
                print(f"已處理幀數: {i - start_idx}")

    if video_writer:
        video_writer.release()
    print("[結束] 影片製作完成。")

if __name__ == "__main__":
    main()