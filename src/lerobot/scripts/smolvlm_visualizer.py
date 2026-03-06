import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForImageTextToText

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the local image file")
    parser.add_argument("--prompt", type=str, required=True, help="Text instruction for the model")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 載入模型與處理器
    print(f"[*] Loading model {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager" 
    ).to(device)
    model.eval()

    # 2. 讀取影像
    raw_image = Image.open(args.image_path).convert("RGB")

    # 3. 準備輸入
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # 針對 Video 模型，images 需要是嵌套列表 [[Image]]
    inputs = processor(text=text, images=[[raw_image]], return_tensors="pt").to(device)

    # 4. 推理
    print(f"[*] Running inference and capturing attention...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True
        )

    # 5. 分析注意力
    # 取得生成的 Tokens
    generated_ids = outputs.sequences[0][inputs.input_ids.shape[-1]:]
    tokens = [processor.tokenizer.decode(tid) for tid in generated_ids]

    last_step_attn = outputs.attentions[-1][-1] # (1, heads, seq_len, seq_len)
    total_seq_len = last_step_attn.shape[-1]
    
    # 2. 自動尋找視覺 Token 的範圍
    # 在 SmolVLM2 中，視覺 Token 的權重通常顯著大於 Padding 或 Special Tokens
    # 我們取平均注意力圖來尋找非零區域
    full_avg_attn = last_step_attn.mean(dim=1)[0, -1, :] # 取得最後一個 token 對全體的注意力
    
    # 尋找權重最大的連續區域，或者根據 SmolVLM2 慣例：
    # 通常序列佈局為: <BOS> + [System Tokens] + [Vision Tokens] + [Prompt Tokens]
    # 我們可以透過排除 Prompt 和已生成的 Tokens 來回推
    prompt_len = inputs.input_ids.shape[-1]
    # 視覺 Token 數量
    num_vision_tokens = total_seq_len - prompt_len - (len(outputs.attentions) - 1)

    print(f"[*] Debug Info: Total Seq={total_seq_len}, Prompt={prompt_len}, Est Vision={num_vision_tokens}")

    vision_weights = []
    for i, step_attn in enumerate(outputs.attentions):
        # 取得當前步驟的最後一層注意力
        # Shape: (1, heads, 1, current_total_seq)
        curr_attn = step_attn[-1].mean(dim=1).squeeze()
        
        # 修正：嘗試不同的切片範圍
        # 如果視覺 Token 被放在開頭，使用 [:num_vision_tokens]
        # 如果有 BOS，則可能是 [1:num_vision_tokens+1]
        v_sum = curr_attn[:num_vision_tokens].sum().item()
        
        # 如果 v_sum 依然是 0，代表視覺 Token 在中間，我們輸出最大權重位置來 Debug
        if v_sum == 0:
            max_idx = curr_attn.argmax().item()
            # 強制捕捉最大權重周圍的區域
            v_sum = curr_attn[max(0, max_idx-50):max_idx+50].sum().item()

        vision_weights.append(v_sum)

    # 6. 繪圖
    plt.figure(figsize=(16, 6))
    plt.plot(vision_weights, marker='o', color='#d62728', linewidth=1.5)
    
    clean_tokens = [t.replace(' ', ' ') for t in tokens]
    plt.xticks(range(len(clean_tokens)), clean_tokens, rotation=70, fontsize=8)
    plt.ylabel("Visual Attention Weight Sum")
    plt.title(f"Attention Analysis | Image: {os.path.basename(args.image_path)}")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("smolvlm_attention_final.png")
    print("[+] Success! Plot saved as smolvlm_attention_final.png")
    plt.show()

if __name__ == "__main__":
    main()