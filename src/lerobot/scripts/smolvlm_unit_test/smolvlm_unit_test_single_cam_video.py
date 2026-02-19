import cv2
import torch
import textwrap
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np

# Configuration
VIDEO_PATH = "single_cam_sorting.mp4"
OUTPUT_VIDEO_PATH = "output_with_overlay.mp4"

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

print("Model Loaded! Starting Video Processing...")

prompt_text = input("Enter prompt: ").strip()
if not prompt_text:
    prompt_text = "Describe the image briefly."

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

frame_interval = 1 #1  # Process every Nth frame
current_frame = 0
last_text = ""

with torch.inference_mode():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video reached.")
            break

        if current_frame % frame_interval == 0:
            resized = cv2.resize(frame, (384, 384))
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

            generated_ids = model.generate(**inputs, max_new_tokens=30,do_sample=False)
            last_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"\rAI Says: {last_text}", end=" " * 20)

            del inputs, generated_ids
            torch.cuda.empty_cache()

        display_frame = frame.copy()
        cv2.rectangle(display_frame, (0, 0), (frame_width, 150), (0, 0, 0), -1)
        wrapped = textwrap.wrap(last_text, width=50)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Change these values:
        font_scale = 0.5  # Increase for larger text (e.g., 1.0, 1.2), decrease for smaller (e.g., 0.5, 0.6)
        thickness = 1     # Increase for bolder text (e.g., 3, 4)
        line_spacing = 35 # Increase for more space between lines if using larger font

        for i, line in enumerate(wrapped):
            y = 40 + i * line_spacing
            cv2.putText(display_frame, line, (20, y), font, font_scale, (0, 255, 0), thickness)

        out.write(display_frame)

        current_frame += 1

cap.release()
out.release()

print("Video processing completed. Output saved to:", OUTPUT_VIDEO_PATH)