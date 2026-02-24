import cv2
import torch
import textwrap
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import os 

# Configuration
IMAGE_PATH = "top_crop.png"

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device detected: {DEVICE}")

# Load model
print(f"Loading {MODEL_ID}...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=DEVICE
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

if hasattr(processor, "image_processor"):
    processor.image_processor.do_resize = True
    processor.image_processor.size = {"height": 384, "width": 384}

print("Model Loaded! Starting Image Processing...")

prompt_text = input("Enter prompt: ").strip()
if not prompt_text:
    prompt_text = "Describe the image briefly."

# Open image
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print("Error: Cannot open image.")
    exit()

with torch.inference_mode():
    resized = cv2.resize(frame, (384, 384))
    save_dir = "debug_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 使用 prompt 作為檔名的一部分（移除特殊字元），方便比對
    safe_prompt = "".join(x for x in prompt_text[:20] if x.isalnum())
    save_path = os.path.join(save_dir, f"input_{safe_prompt}.png")
    
    cv2.imwrite(save_path, resized)
    print(f"處理後的圖片已儲存至: {save_path}")
    
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(DEVICE, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    last_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"AI Says: {last_text}")

    del inputs, generated_ids
    torch.cuda.empty_cache()
