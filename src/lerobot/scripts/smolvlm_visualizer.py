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

    last_step_attn = outputs.attentions[-1][-1] # (1, heads, seq_len, seq_len)
    total_seq_len = last_step_attn.shape[-1]

    # 自動尋找視覺 Token 的範圍
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

    print("\n=== Input Token → Vision Attention ===")

    input_ids = inputs.input_ids[0]
    input_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

    prompt_attn = outputs.attentions[0]  # prompt forward pass
    last_layer = prompt_attn[-1]         # 最後一層

    # (1, heads, seq, seq)
    avg_attn = last_layer.mean(dim=1)[0]

    input_vision_weights = []

    for i in range(prompt_len):
        token_attn = avg_attn[i]
        v_sum = token_attn[:num_vision_tokens].sum().item()
        input_vision_weights.append(v_sum)

    for t, w in zip(input_tokens, input_vision_weights):
        print(f"{t:12s} | attention={w:.4f}")

    print("\n=== Generated Token → Vision Attention ===")

    generated_ids = outputs.sequences[0][inputs.input_ids.shape[-1]:]
    tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids)

    output_vision_weights = []

    for step_attn in outputs.attentions:
        # 最後一層
        last_layer = step_attn[-1]

        # (1, heads, 1, seq)
        curr_attn = last_layer.mean(dim=1).squeeze()

        v_sum = curr_attn[:num_vision_tokens].sum().item()
        output_vision_weights.append(v_sum)

    for t, w in zip(tokens, output_vision_weights):
        print(f"{t:12s} | attention={w:.4f}")
    
    all_tokens = processor.tokenizer.convert_ids_to_tokens(outputs.sequences[0])

    print("\n=== Full Token Layout ===")
    for i, tok in enumerate(all_tokens):
        print(i, tok)

    # 6. 繪圖
    plt.figure(figsize=(16,6))

    plt.plot(input_vision_weights, marker='o')

    plt.xticks(
        range(len(input_tokens)),
        input_tokens,
        rotation=70,
        fontsize=8
    )

    plt.ylabel("Vision Attention Weight")
    plt.title("Input Tokens → Vision Attention")
    plt.tight_layout()

    plt.savefig("input_attention.png")

    plt.figure(figsize=(16,6))

    plt.plot(output_vision_weights, marker='o')

    plt.xticks(
        range(len(tokens)),
        tokens,
        rotation=70,
        fontsize=8
    )

    plt.ylabel("Vision Attention Weight")
    plt.title("Generated Tokens → Vision Attention")
    plt.tight_layout()

    plt.savefig("output_attention.png")

if __name__ == "__main__":
    main()