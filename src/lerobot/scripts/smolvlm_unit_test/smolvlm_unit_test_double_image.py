import cv2
import torch
import textwrap
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np

# Configuration
IMAGE_PATH_1 = "single_cam_sorting_front.png"
IMAGE_PATH_2 = "single_cam_sorting_top.png"

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

# Open images
frame_1 = cv2.imread(IMAGE_PATH_1)
frame_2 = cv2.imread(IMAGE_PATH_2)
if frame_1 is None or frame_2 is None:
    print("Error: Cannot open one or both images.")
    exit()

with torch.inference_mode():
    resized_1 = cv2.resize(frame_1, (384, 384))
    resized_2 = cv2.resize(frame_2, (384, 384))
    rgb_1 = cv2.cvtColor(resized_1, cv2.COLOR_BGR2RGB)
    rgb_2 = cv2.cvtColor(resized_2, cv2.COLOR_BGR2RGB)
    pil_image_1 = Image.fromarray(rgb_1)
    pil_image_2 = Image.fromarray(rgb_2)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image_1},
                {"type": "image", "image": pil_image_2},
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

    generated_ids = model.generate(**inputs, max_new_tokens=30)
    last_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"AI Says: {last_text}")

    del inputs, generated_ids
    torch.cuda.empty_cache()
