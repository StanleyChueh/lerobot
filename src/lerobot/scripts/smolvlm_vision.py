import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. 配置與模型載入
# ==========================================
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
DATASET_ID = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 載入處理器與模型 (建議開啟 bfloat16 或 4bit 節省顯存)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    _attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
).to(DEVICE)

# ==========================================
# 2. 載入資料集
# ==========================================
# 注意：請確保你已經透過 huggingface-cli login 登入，如果有權限限制的話
print(f"Loading dataset: {DATASET_ID}...")
ds = load_dataset(DATASET_ID, split="train")

# ==========================================
# 3. 定義處理函數
# ==========================================
def process_episode(sample, num_frames=5):
    """
    從資料集中抽取影像序列並讓 SmolVLM 判斷
    """
    # 假設資料集結構中包含 'image' 欄位且為序列
    # 如果資料集是平鋪的，則需要根據 episode_id 進行分組（此處假設 sample 已是一段序列）
    all_images = sample['image'] 
    
    # 均勻抽樣：確保模型看到動作的開始、中間與結束
    total_frames = len(all_images)
    indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    sampled_images = [all_images[i] for i in indices]

    # 設定 Prompt
    # 我們明確告訴模型觀察夾爪與桌面的距離
    prompt_text = (
        "Analyze this sequence of robot arm movements. The prompt is 'put the red cube in the box'. "
        "Focus on the vertical height of the gripper during the transfer phase. "
        "Is the trajectory 'High' (significant clearance from table) or 'Low' (stays close to the table)? "
        "Answer with only one word: High or Low."
    )

    # 建立多圖對話格式
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(num_frames)] + [{"type": "text", "text": prompt_text}]
        }
    ]

    # 準備輸入
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=sampled_images, return_tensors="pt").to(DEVICE)

    # 生成預測
    generated_ids = model.generate(**inputs, max_new_tokens=10)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 清理輸出字串（只取最後的回覆部分）
    prediction = result.split("assistant\n")[-1].strip().capitalize()
    return prediction

# ==========================================
# 4. 執行批次處理
# ==========================================
results = []

print("Starting inference...")
# 為了示範，我們先處理前 100 筆資料，你可以移除 [:100] 處理全部
for i in tqdm(range(len(ds))):
    try:
        episode_data = ds[i]
        label = process_episode(episode_data)
        
        results.append({
            "index": i,
            "prediction": label,
            "episode_id": episode_data.get("episode_id", "N/A")
        })
    except Exception as e:
        print(f"Error processing index {i}: {e}")

# ==========================================
# 5. 儲存結果
# ==========================================
df = pd.DataFrame(results)
df.to_csv("vla_trajectory_classification.csv", index=False)
print("Classification complete. Results saved to 'vla_trajectory_classification.csv'.")

# 統計一下高低的比例
print("\nSummary of results:")
print(df['prediction'].value_counts())