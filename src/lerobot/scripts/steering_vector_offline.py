# # #high and low

# import torch
# import numpy as np
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.policies.factory import make_policy, make_pre_post_processors
# from lerobot.configs.policies import PreTrainedConfig
# from lerobot.utils.utils import get_safe_torch_device
# from lerobot.utils.control_utils import predict_action
# import copy

# class ActivationCapturer:
#     def __init__(self):
#         self.captured = None

#     def hook_fn(module, inputs, outputs):
#         # 如果已經抓到了，就不覆蓋 (確保我們拿到的是包含整個 prompt 的那一次 forward)
#         if captured_activation[0] is not None:
#             return 
            
#         if isinstance(outputs, tuple):
#             hidden_states = outputs[0]
#         else:
#             hidden_states = outputs
            
#         captured_activation[0] = hidden_states.detach().cpu()

# def extract_steering_vector_caa(policy, preprocessor, postprocessor, dataset, prompt_pairs, rename_map, target_layer=10, num_samples=30, device="cuda"):
#     """
#     CAA 提取流程：使用 Monkey Patching 確保 100% 擷取到 Activation
#     """
#     # 1. 定位到真正的層
#     try:
#         target_layer_module = policy.model.vlm_with_expert.lm_expert.layers[target_layer]
#         print(f"[✓] 成功定位到目標層：policy.model.vlm_with_expert.lm_expert.layers[{target_layer}]")
#     except Exception as e:
#         raise ValueError(f"[X] 定位目標層失敗，請檢查模型結構。錯誤訊息: {e}")

#     # 2. 準備閉包變數來儲存擷取到的 Activation
#     captured_activation = [None]

#     # 3. 定義 Hook 函數 (你原本寫的 hook_fn 已經有 tuple 檢查，非常完美！)
#     def hook_fn(module, inputs, outputs):
#         if isinstance(outputs, tuple):
#             hidden_states = outputs[0]
#         else:
#             hidden_states = outputs
            
#         # 將資料複製並移至 CPU，避免顯存爆炸
#         captured_activation[0] = hidden_states.detach().cpu()

#     # 4 & 5. 註冊 Forward Hook 至該層的 mlp 子模組
#     if hasattr(target_layer_module, "mlp"):
#         hook_handle = target_layer_module.mlp.register_forward_hook(hook_fn)
#         print(f"[*] 成功將 Forward Hook 註冊至 Layer {target_layer} 的 mlp 子模組。")
#     else:
#         # 備用方案：萬一結構命名不同（SmolVLM/Llama 結構標準命名皆為 mlp）
#         raise AttributeError(f"[X] 在 Layer {target_layer} 中找不到 'mlp' 子模組，請檢查模型結構。")

#     diff_accumulated = []
#     num_samples = min(num_samples, len(dataset))
#     print(f"[*] 開始從資料集中提取 {num_samples} 個觀測幀進行 CAA 差值計算...")

#     try:
#         for frame_idx in range(num_samples):
#             sample_frame = dataset[frame_idx]
            
#             # 建立標準的觀測字典 
#             observation_frame = {}
#             for k, v in sample_frame.items():
#                 if isinstance(v, str):
#                     continue
                    
#                 target_key = rename_map.get(k, k)
                
#                 if isinstance(v, torch.Tensor):
#                     if "image" in k:
#                         img_np = v.detach().cpu().numpy().transpose(1, 2, 0)
#                         if img_np.dtype == np.float32 and img_np.max() <= 1.0:
#                             img_np = (img_np * 255).astype(np.uint8)
#                         observation_frame[target_key] = img_np
#                     else:
#                         observation_frame[target_key] = v.detach().cpu().numpy()
#                 else:
#                     observation_frame[target_key] = v

#             # 遍歷多組對比 Prompt
#             for pair_idx, (prompt_high, prompt_low) in enumerate(prompt_pairs):
                
#                 # --- High Prompt ---
#                 captured_activation[0] = None  # 重置
#                 policy.reset()

#                 _ = predict_action(
#                     observation=observation_frame,
#                     policy=policy,
#                     preprocessor=preprocessor,
#                     postprocessor=postprocessor,
#                     use_amp=policy.config.use_amp,
#                     device=device,
#                     task=prompt_high,
#                 )
                
#                 if captured_activation[0] is None:
#                     print(f"[!] 警告: Layer {target_layer} 在 High Prompt 推理時未能成功擷取到 Activation。")
#                     continue
                    
#                 # 💡 這裡加上 shape 診斷，防止後續 mean 報錯
#                 act_high = captured_activation[0]
#                 if frame_idx == 0 and pair_idx == 0:
#                     print(f"    --> [診斷] 成功擷取到 Activation! 原始 Shape: {act_high.shape}")
                
#                 # 預設對 sequence 長度維度 (通常是 dim=1) 做平均
#                 h_high = act_high.mean(dim=1) 

#                 # --- Low Prompt ---
#                 captured_activation[0] = None  # 重置
#                 policy.reset()

#                 _ = predict_action(
#                     observation=observation_frame,
#                     policy=policy,
#                     preprocessor=preprocessor,
#                     postprocessor=postprocessor,
#                     use_amp=policy.config.use_amp,
#                     device=device,
#                     task=prompt_low,
#                 )
                
#                 if captured_activation[0] is None:
#                     print(f"[!] 警告: Layer {target_layer} 在 Low Prompt 推理時未能成功擷取到 Activation。")
#                     continue
                    
#                 h_low = captured_activation[0].mean(dim=1)  

#                 diff = h_high - h_low
#                 diff_accumulated.append(diff)

#     finally:
#         # 6. 無論成功或失敗，最後一定要移除 Hook，避免影響其他地方或造成 Memory Leak
#         hook_handle.remove()
#         print("[*] 已安全移除 Forward Hook。")

#     if not diff_accumulated:
#         print("[X] 錯誤: 未能收集到任何激活值差值，提取失敗。")
#         return

#     mean_diff = torch.cat(diff_accumulated, dim=0).mean(dim=0, keepdim=True)  
#     v_steer = torch.nn.functional.normalize(mean_diff, p=2, dim=-1)

#     output_filename = f"steering_vector_L{target_layer}_caa.pt"
#     torch.save(v_steer, output_filename)
#     print(f"\n[✓] CAA Steering Vector 成功儲存至: {output_filename}")
#     print(f"    Vector Shape: {v_steer.shape}")

# def main():
#     # --- 1. 實驗參數與路徑更正 ---
#     policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2"
#     dataset_repo_id = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2"
    
#     target_layer = 14
#     num_samples = 60  

#     prompt_pairs = [
#         ("Put the red cube in the box moving high.", "Put the red cube in the box moving low."),
#         ("Lift the red cube higher.", "Keep the red cube lower."),
#         ("Move the object higher.", "Move the object lower."),
#         ("Use a higher trajectory.", "Use a lower trajectory.")
#     ]
    
#     device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[*] 當前執行環境設備: {device}")

#     # --- 2. 載入資料集 ---
#     print(f"[*] 正在載入資料集: {dataset_repo_id}")
#     dataset = LeRobotDataset(dataset_repo_id)
#     print(f"    --> [診斷] 資料集總幀數: {len(dataset)}")

#     if len(dataset) > 0:
#         print(f"    --> [診斷] 資料集包含的原始特徵 Keys:\n        {list(dataset[0].keys())}")

#     # 💡 修正 1：將 RENAME_MAP 移到 make_policy 之前
#     RENAME_MAP = {
#         "observation.images.front": "observation.images.camera1",
#         "observation.images.top": "observation.images.camera2",
#         "observation.images.wrist": "observation.images.camera3",
#     }

#     # 💡 修正 2：建立一個修補過的 meta 複本，替換掉不符合的 key，用來通過 make_policy 檢查
#     patched_meta = copy.deepcopy(dataset.meta)
#     for old_key, new_key in RENAME_MAP.items():
#         # 替換 features 字典中的 key
#         if old_key in patched_meta.features:
#             patched_meta.features[new_key] = patched_meta.features.pop(old_key)
#         # 替換 stats 字典中的 key (preprocessor 會用到)
#         if hasattr(patched_meta, "stats") and patched_meta.stats and old_key in patched_meta.stats:
#             patched_meta.stats[new_key] = patched_meta.stats.pop(old_key)

#     # --- 3. 載入 VLA Policy ---
#     print(f"[*] 正在載入 Policy 設定與模型權重: {policy_path}")
#     policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
#     policy_cfg.device = str(device)
    
#     # 💡 修正 3：傳入 patched_meta 而不是原始的 dataset.meta
#     policy = make_policy(policy_cfg, ds_meta=patched_meta)
#     policy.eval()

#     preprocessor, postprocessor = make_pre_post_processors(
#         policy_cfg=policy_cfg,
#         pretrained_path=policy_path,
#         dataset_stats=patched_meta.stats, # 這裡同樣使用修補過的 stats
#     )
    
#     if hasattr(preprocessor, "keys"):
#         print(f"    --> [診斷] Policy (Preprocessor) 預期接收的輸入 Keys:\n        {list(preprocessor.keys())}")

#     # --- 4. 執行提取 ---
#     extract_steering_vector_caa(
#         policy=policy,
#         preprocessor=preprocessor,
#         postprocessor=postprocessor,
#         dataset=dataset,
#         prompt_pairs=prompt_pairs,
#         rename_map=RENAME_MAP, # 將 RENAME_MAP 傳入迴圈繼續使用
#         target_layer=target_layer,
#         num_samples=num_samples,
#         device=device
#     )

# if __name__ == "__main__":
#     main()


# #left and right

# import torch
# import numpy as np
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.policies.factory import make_policy, make_pre_post_processors
# from lerobot.configs.policies import PreTrainedConfig
# from lerobot.utils.utils import get_safe_torch_device
# from lerobot.utils.control_utils import predict_action
# import copy

# class ActivationCapturer:
#     def __init__(self):
#         self.captured = None

#     def hook_fn(module, inputs, outputs):
#         # 如果已經抓到了，就不覆蓋 (確保我們拿到的是包含整個 prompt 的那一次 forward)
#         if captured_activation[0] is not None:
#             return 
            
#         if isinstance(outputs, tuple):
#             hidden_states = outputs[0]
#         else:
#             hidden_states = outputs
            
#         captured_activation[0] = hidden_states.detach().cpu()

# def extract_steering_vector_caa(policy, preprocessor, postprocessor, dataset, prompt_pairs, rename_map, target_layer=10, num_samples=30, device="cuda"):
#     """
#     CAA 提取流程：使用 Monkey Patching 確保 100% 擷取到 Activation
#     """
#     # 1. 定位到真正的層
#     try:
#         target_layer_module = policy.model.vlm_with_expert.lm_expert.layers[target_layer]
#         print(f"[✓] 成功定位到目標層：policy.model.vlm_with_expert.lm_expert.layers[{target_layer}]")
#     except Exception as e:
#         raise ValueError(f"[X] 定位目標層失敗，請檢查模型結構。錯誤訊息: {e}")

#     # 2. 準備閉包變數來儲存擷取到的 Activation
#     captured_activation = [None]

#     # 3. 定義 Hook 函數
#     def hook_fn(module, inputs, outputs):
#         if isinstance(outputs, tuple):
#             hidden_states = outputs[0]
#         else:
#             hidden_states = outputs
            
#         # 將資料複製並移至 CPU，避免顯存爆炸
#         captured_activation[0] = hidden_states.detach().cpu()

#     # 4 & 5. 註冊 Forward Hook 至該層的 mlp 子模組
#     if hasattr(target_layer_module, "mlp"):
#         hook_handle = target_layer_module.mlp.register_forward_hook(hook_fn)
#         print(f"[*] 成功將 Forward Hook 註冊至 Layer {target_layer} 的 mlp 子模組。")
#     else:
#         # 備用方案：萬一結構命名不同（SmolVLM/Llama 結構標準命名皆為 mlp）
#         raise AttributeError(f"[X] 在 Layer {target_layer} 中找不到 'mlp' 子模組，請檢查模型結構。")

#     diff_accumulated = []
#     num_samples = min(num_samples, len(dataset))
#     print(f"[*] 開始從資料集中提取 {num_samples} 個觀測幀進行 CAA 差值計算...")

#     try:
#         for frame_idx in range(num_samples):
#             sample_frame = dataset[frame_idx]
            
#             # 建立標準的觀測字典 
#             observation_frame = {}
#             for k, v in sample_frame.items():
#                 if isinstance(v, str):
#                     continue
                    
#                 target_key = rename_map.get(k, k)
                
#                 if isinstance(v, torch.Tensor):
#                     if "image" in k:
#                         img_np = v.detach().cpu().numpy().transpose(1, 2, 0)
#                         if img_np.dtype == np.float32 and img_np.max() <= 1.0:
#                             img_np = (img_np * 255).astype(np.uint8)
#                         observation_frame[target_key] = img_np
#                     else:
#                         observation_frame[target_key] = v.detach().cpu().numpy()
#                 else:
#                     observation_frame[target_key] = v

#             # 遍歷多組對比 Prompt (改成 Left vs Right)
#             for pair_idx, (prompt_left, prompt_right) in enumerate(prompt_pairs):
                
#                 # --- Left Prompt (取代原本的 High) ---
#                 captured_activation[0] = None  # 重置
#                 policy.reset()

#                 _ = predict_action(
#                     observation=observation_frame,
#                     policy=policy,
#                     preprocessor=preprocessor,
#                     postprocessor=postprocessor,
#                     use_amp=policy.config.use_amp,
#                     device=device,
#                     task=prompt_left,
#                 )
                
#                 if captured_activation[0] is None:
#                     print(f"[!] 警告: Layer {target_layer} 在 Left Prompt 推理時未能成功擷取到 Activation。")
#                     continue
                    
#                 act_left = captured_activation[0]
#                 if frame_idx == 0 and pair_idx == 0:
#                     print(f"    --> [診斷] 成功擷取到 Activation! 原始 Shape: {act_left.shape}")
                
#                 # 預設對 sequence 長度維度 (通常是 dim=1) 做平均
#                 h_left = act_left.mean(dim=1) 

#                 # --- Right Prompt (取代原本的 Low) ---
#                 captured_activation[0] = None  # 重置
#                 policy.reset()

#                 _ = predict_action(
#                     observation=observation_frame,
#                     policy=policy,
#                     preprocessor=preprocessor,
#                     postprocessor=postprocessor,
#                     use_amp=policy.config.use_amp,
#                     device=device,
#                     task=prompt_right,
#                 )
                
#                 if captured_activation[0] is None:
#                     print(f"[!] 警告: Layer {target_layer} 在 Right Prompt 推理時未能成功擷取到 Activation。")
#                     continue
                    
#                 h_right = captured_activation[0].mean(dim=1)  

#                 # 計算方向差值：Left - Right
#                 diff = h_left - h_right
#                 diff_accumulated.append(diff)

#     finally:
#         # 6. 無論成功或失敗，最後一定要移除 Hook
#         hook_handle.remove()
#         print("[*] 已安全移除 Forward Hook。")

#     if not diff_accumulated:
#         print("[X] 錯誤: 未能收集到任何激活值差值，提取失敗。")
#         return

#     mean_diff = torch.cat(diff_accumulated, dim=0).mean(dim=0, keepdim=True)  
#     v_steer = torch.nn.functional.normalize(mean_diff, p=2, dim=-1)

#     # 💡 這裡將檔名改為包含 left_right，避免覆蓋你原本的高低向量檔案
#     output_filename = f"steering_vector_L{target_layer}_left_right_caa.pt"
#     torch.save(v_steer, output_filename)
#     print(f"\n[✓] CAA Steering Vector 成功儲存至: {output_filename}")
#     print(f"    Vector Shape: {v_steer.shape}")

# def main():
#     # --- 1. 實驗參數與路徑 ---
#     policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_direction"
#     dataset_repo_id = "ethanCSL/svla_koch_pick_n_place_vla_steering_direction"
    
#     target_layer = 14
#     num_samples = 60  

#     # 💡 修改點：將提示詞改為 左/右 的對比
#     prompt_pairs = [
#         ("Put the red cube in the box moving to the left.", "Put the red cube in the box moving to the right."),
#         ("Put the red cube in the left box.", "Put the red cube in the right box."),
#         ("Move the red cube more to the left.", "Move the red cube more to the right."),
#         ("Shift the object to the left side.", "Shift the object to the right side."),
#         ("Use a leftward trajectory.", "Use a rightward trajectory.")
#     ]
    
#     device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[*] 當前執行環境設備: {device}")

#     # --- 2. 載入資料集 ---
#     print(f"[*] 正在載入資料集: {dataset_repo_id}")
#     dataset = LeRobotDataset(dataset_repo_id)
#     print(f"    --> [診斷] 資料集總幀數: {len(dataset)}")

#     if len(dataset) > 0:
#         print(f"    --> [診斷] 資料集包含的原始特徵 Keys:\n        {list(dataset[0].keys())}")

#     RENAME_MAP = {
#         "observation.images.front": "observation.images.camera1",
#         "observation.images.top": "observation.images.camera2",
#         "observation.images.wrist": "observation.images.camera3",
#     }

#     # 建立一個修補過的 meta 複本，替換掉不符合的 key
#     patched_meta = copy.deepcopy(dataset.meta)
#     for old_key, new_key in RENAME_MAP.items():
#         if old_key in patched_meta.features:
#             patched_meta.features[new_key] = patched_meta.features.pop(old_key)
#         if hasattr(patched_meta, "stats") and patched_meta.stats and old_key in patched_meta.stats:
#             patched_meta.stats[new_key] = patched_meta.stats.pop(old_key)

#     # --- 3. 載入 VLA Policy ---
#     print(f"[*] 正在載入 Policy 設定與模型權重: {policy_path}")
#     policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
#     policy_cfg.device = str(device)
    
#     policy = make_policy(policy_cfg, ds_meta=patched_meta)
#     policy.eval()

#     preprocessor, postprocessor = make_pre_post_processors(
#         policy_cfg=policy_cfg,
#         pretrained_path=policy_path,
#         dataset_stats=patched_meta.stats, 
#     )
    
#     if hasattr(preprocessor, "keys"):
#         print(f"    --> [診斷] Policy (Preprocessor) 預期接收的輸入 Keys:\n        {list(preprocessor.keys())}")

#     # --- 4. 執行提取 ---
#     extract_steering_vector_caa(
#         policy=policy,
#         preprocessor=preprocessor,
#         postprocessor=postprocessor,
#         dataset=dataset,
#         prompt_pairs=prompt_pairs,
#         rename_map=RENAME_MAP, 
#         target_layer=target_layer,
#         num_samples=num_samples,
#         device=device
#     )

# if __name__ == "__main__":
#     main()


# green and red

import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.control_utils import predict_action
import copy

class ActivationCapturer:
    def __init__(self):
        self.captured = None

    def hook_fn(module, inputs, outputs):
        # 如果已經抓到了，就不覆蓋 (確保我們拿到的是包含整個 prompt 的那一次 forward)
        if captured_activation[0] is not None:
            return 
            
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
            
        captured_activation[0] = hidden_states.detach().cpu()

def extract_steering_vector_caa(policy, preprocessor, postprocessor, dataset, prompt_pairs, rename_map, target_layer=10, num_samples=30, device="cuda"):
    """
    CAA 提取流程：使用 Monkey Patching 確保 100% 擷取到 Activation
    """
    # 1. 定位到真正的層
    try:
        target_layer_module = policy.model.vlm_with_expert.lm_expert.layers[target_layer]
        print(f"[✓] 成功定位到目標層：policy.model.vlm_with_expert.lm_expert.layers[{target_layer}]")
    except Exception as e:
        raise ValueError(f"[X] 定位目標層失敗，請檢查模型結構。錯誤訊息: {e}")

    # 2. 準備閉包變數來儲存擷取到的 Activation
    captured_activation = [None]

    # 3. 定義 Hook 函數
    def hook_fn(module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
            
        # 將資料複製並移至 CPU，避免顯存爆炸
        captured_activation[0] = hidden_states.detach().cpu()

    # 4 & 5. 註冊 Forward Hook 至該層的 mlp 子模組
    if hasattr(target_layer_module, "mlp"):
        hook_handle = target_layer_module.mlp.register_forward_hook(hook_fn)
        print(f"[*] 成功將 Forward Hook 註冊至 Layer {target_layer} 的 mlp 子模組。")
    else:
        # 備用方案：萬一結構命名不同（SmolVLM/Llama 結構標準命名皆為 mlp）
        raise AttributeError(f"[X] 在 Layer {target_layer} 中找不到 'mlp' 子模組，請檢查模型結構。")

    diff_accumulated = []
    num_samples = min(num_samples, len(dataset))
    print(f"[*] 開始從資料集中提取 {num_samples} 個觀測幀進行 CAA 差值計算...")

    try:
        for frame_idx in range(num_samples):
            sample_frame = dataset[frame_idx]
            
            # 建立標準的觀測字典 
            observation_frame = {}
            for k, v in sample_frame.items():
                if isinstance(v, str):
                    continue
                    
                target_key = rename_map.get(k, k)
                
                if isinstance(v, torch.Tensor):
                    if "image" in k:
                        img_np = v.detach().cpu().numpy().transpose(1, 2, 0)
                        if img_np.dtype == np.float32 and img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        observation_frame[target_key] = img_np
                    else:
                        observation_frame[target_key] = v.detach().cpu().numpy()
                else:
                    observation_frame[target_key] = v

            # 遍歷多組對比 Prompt (改成 Left vs Right)
            for pair_idx, (prompt_red, prompt_green) in enumerate(prompt_pairs):
                
                # --- Left Prompt (取代原本的 High) ---
                captured_activation[0] = None  # 重置
                policy.reset()

                _ = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    device=device,
                    task=prompt_red,
                )
                
                if captured_activation[0] is None:
                    print(f"[!] 警告: Layer {target_layer} 在 red Prompt 推理時未能成功擷取到 Activation。")
                    continue
                    
                act_left = captured_activation[0]
                if frame_idx == 0 and pair_idx == 0:
                    print(f"    --> [診斷] 成功擷取到 Activation! 原始 Shape: {act_left.shape}")
                
                # 預設對 sequence 長度維度 (通常是 dim=1) 做平均
                h_left = act_left.mean(dim=1) 

                # --- Right Prompt (取代原本的 Low) ---
                captured_activation[0] = None  # 重置
                policy.reset()

                _ = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    device=device,
                    task=prompt_green,
                )
                
                if captured_activation[0] is None:
                    print(f"[!] 警告: Layer {target_layer} 在 green Prompt 推理時未能成功擷取到 Activation。")
                    continue
                    
                h_right = captured_activation[0].mean(dim=1)  

                # 計算方向差值：Left - Right
                diff = h_left - h_right
                diff_accumulated.append(diff)

    finally:
        # 6. 無論成功或失敗，最後一定要移除 Hook
        hook_handle.remove()
        print("[*] 已安全移除 Forward Hook。")

    if not diff_accumulated:
        print("[X] 錯誤: 未能收集到任何激活值差值，提取失敗。")
        return

    mean_diff = torch.cat(diff_accumulated, dim=0).mean(dim=0, keepdim=True)  
    v_steer = torch.nn.functional.normalize(mean_diff, p=2, dim=-1)

    # 💡 這裡將檔名改為包含 left_right，避免覆蓋你原本的高低向量檔案
    output_filename = f"steering_vector_L{target_layer}_red_green_caa.pt"
    torch.save(v_steer, output_filename)
    print(f"\n[✓] CAA Steering Vector 成功儲存至: {output_filename}")
    print(f"    Vector Shape: {v_steer.shape}")

def main():
    # --- 1. 實驗參數與路徑 ---
    policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_color"
    dataset_repo_id = "ethanCSL/svla_koch_pick_n_place_vla_steering_color"
    
    target_layer = 14
    num_samples = 60  

    # 💡 修改點：將提示詞改為 左/右 的對比
    prompt_pairs = [
        ("Put the red cube in the box.", "Put the green cube in the box"),
    ]
    
    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 當前執行環境設備: {device}")

    # --- 2. 載入資料集 ---
    print(f"[*] 正在載入資料集: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    print(f"    --> [診斷] 資料集總幀數: {len(dataset)}")

    if len(dataset) > 0:
        print(f"    --> [診斷] 資料集包含的原始特徵 Keys:\n        {list(dataset[0].keys())}")

    RENAME_MAP = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
        "observation.images.wrist": "observation.images.camera3",
    }

    # 建立一個修補過的 meta 複本，替換掉不符合的 key
    patched_meta = copy.deepcopy(dataset.meta)
    for old_key, new_key in RENAME_MAP.items():
        if old_key in patched_meta.features:
            patched_meta.features[new_key] = patched_meta.features.pop(old_key)
        if hasattr(patched_meta, "stats") and patched_meta.stats and old_key in patched_meta.stats:
            patched_meta.stats[new_key] = patched_meta.stats.pop(old_key)

    # --- 3. 載入 VLA Policy ---
    print(f"[*] 正在載入 Policy 設定與模型權重: {policy_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = str(device)
    
    policy = make_policy(policy_cfg, ds_meta=patched_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=patched_meta.stats, 
    )
    
    if hasattr(preprocessor, "keys"):
        print(f"    --> [診斷] Policy (Preprocessor) 預期接收的輸入 Keys:\n        {list(preprocessor.keys())}")

    # --- 4. 執行提取 ---
    extract_steering_vector_caa(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        prompt_pairs=prompt_pairs,
        rename_map=RENAME_MAP, 
        target_layer=target_layer,
        num_samples=num_samples,
        device=device
    )

if __name__ == "__main__":
    main()