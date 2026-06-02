import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import sys
from transformers import AutoProcessor, AutoModelForImageTextToText
import torchvision.transforms as T

# 自動相容不同版本的 LeRobot 匯入路徑 (v2.x / v3.x)
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ==========================================
# 1. 配置與模型載入
# ==========================================
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
DATASET_ID = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    _attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
).to(DEVICE)

# ==========================================
# 2. 使用官方 LeRobotDataset 載入資料集
# ==========================================
print(f"Loading LeRobot dataset: {DATASET_ID}...")
# LeRobotDataset 會動態連結並自動解開底層的 MP4 影片串流
dataset = LeRobotDataset(DATASET_ID)

# ==========================================
# 3. 自動偵測解碼後的真實影像與 Episode 欄位
# ==========================================
# 透過讀取第 0 幀來檢查 LeRobotDataset 解碼後實際包含的所有特徵鍵名
first_sample = dataset[0]
image_key = None
for k in first_sample.keys():
    if any(w in k.lower() for w in ["image", "cam", "rgb", "view", "pic"]):
        image_key = k
        break

if not image_key:
    print(f"\n[ERROR] 找不到任何影像欄位！解碼後的特徵有：\n👉 {list(first_sample.keys())}")
    sys.exit(1)

print(f"[INFO] 成功在 LeRobot 樣本中識別到影像欄位: '{image_key}'")

# 取得底層的表格結構，用來做快速的 Episode 分組
hf_ds = dataset.hf_dataset
episode_key = "episode_index" if "episode_index" in hf_ds.column_names else "episode_id"
if episode_key not in hf_ds.column_names:
    print(f"\n[ERROR] 找不到 episode 分組欄位。資料欄位為: {hf_ds.column_names}")
    sys.exit(1)

print(f"Using episode key: '{episode_key}' and image key: '{image_key}'")

# ==========================================
# 4. 定義處理函數
# ==========================================
def process_episode(sampled_images, num_frames=5):
    prompt_text = (
        "Analyze this sequence of robot arm movements. The prompt is 'put the red cube in the box'. "
        "Focus on the vertical height of the gripper during the transfer phase. "
        "Is the trajectory 'High' (significant clearance from table) or 'Low' (stays close to the table)? "
        "Answer with only one word: High or Low."
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(num_frames)] + [{"type": "text", "text": prompt_text}]
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=sampled_images, return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=10)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    prediction = result.split("assistant\n")[-1].strip().capitalize()
    return prediction

# ==========================================
# 5. 執行按 Episode 分組的批次處理
# ==========================================
print("Structuring dataset into episodes...")
episode_to_indices = defaultdict(list)
for idx, ep_id in enumerate(hf_ds[episode_key]):
    episode_to_indices[ep_id].append(idx)

results = []
num_frames = 5
to_pil = T.ToPILImage()

print(f"Starting inference on {len(episode_to_indices)} episodes...")
for episode_id, indices in tqdm(episode_to_indices.items()):
    try:
        total_frames = len(indices)
        if total_frames < num_frames:
            print(f"Skipping episode {episode_id}: too few frames ({total_frames})")
            continue
            
        # 均勻抽樣 5 幀的全局索引
        sampled_indices = [indices[int(i * (total_frames - 1) / (num_frames - 1))] for i in range(num_frames)]
        
        # 讀取並解碼影像
        sampled_images = []
        for idx in sampled_indices:
            img = dataset[idx][image_key]
            
            # 防禦機制：LeRobot 影片解碼後通常為 PyTorch Tensor (C, H, W)
            # 我們需要將其轉回 PIL Image，以便對齊 SmolVLM 的輸入要求
            if isinstance(img, torch.Tensor):
                img = to_pil(img.cpu())
            sampled_images.append(img)
        
        # 送入模型預測
        label = process_episode(sampled_images, num_frames=num_frames)
        
        results.append({
            "episode_id": episode_id,
            "prediction": label
        })
    except Exception as e:
        print(f"Error processing episode {episode_id}: {e}")

# ==========================================
# 6. 儲存結果
# ==========================================
if not results:
    print("\n[ERROR] No episodes were successfully classified.")
else:
    df = pd.DataFrame(results)
    df.to_csv("vla_trajectory_classification.csv", index=False)
    print("\nClassification complete. Results saved to 'vla_trajectory_classification.csv'.")
    print("\nSummary of results:")
    print(df['prediction'].value_counts())