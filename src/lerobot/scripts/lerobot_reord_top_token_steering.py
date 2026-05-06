import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

@torch.no_grad()
def extract_semantic_embeddings_from_policy(policy, top_k_tokens=10, device=None):
    """
    從已經載入的 policy 中萃取 FFN value vectors 並計算對應的 Top Tokens 與 Logits。
    這支函數現在可以直接吃 policy 物件，方便我們在「修改權重前後」重複呼叫。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vlm_model = policy.model.vlm_with_expert.vlm
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    W_out = vlm_model.lm_head.weight.detach().to(device=device, dtype=torch.float32)

    semantic_embeddings = []
    metadata = []
    global_id = 0
    num_layers = len(text_model.layers)

    for layer_idx in range(num_layers):
        # 取得該層的 down_proj 權重
        W_value = text_model.layers[layer_idx].mlp.down_proj.weight.detach().to(
            device=device, dtype=torch.float32
        ) 

        # 計算 Logits: [vocab_size, d_ff]
        token_logits = W_out @ W_value

        # 取 Top-k tokens
        top_logits, top_token_ids = torch.topk(token_logits, k=top_k_tokens, dim=0)

        top_logits_t = top_logits.transpose(0, 1).contiguous()
        top_token_ids_t = top_token_ids.transpose(0, 1).contiguous()
        weights = torch.softmax(top_logits_t, dim=1)  
        token_embs = W_out[top_token_ids_t]

        # Semantic embedding
        e_sem = (weights.unsqueeze(-1) * token_embs).sum(dim=1)
        e_sem = torch.nn.functional.normalize(e_sem, p=2, dim=1)

        e_sem_np = e_sem.cpu().numpy()
        top_token_ids_np = top_token_ids_t.cpu().numpy()
        top_logits_np = top_logits_t.cpu().numpy()

        for neuron_idx in range(W_value.shape[1]):
            token_ids = top_token_ids_np[neuron_idx].tolist()
            decoded_tokens = [
                tokenizer.decode([tok_id]).replace("\n", " ").strip()
                for tok_id in token_ids
            ]

            # 計算該神經元的 L2 Norm
            neuron_norm = W_value[:, neuron_idx].norm().item()

            semantic_embeddings.append(e_sem_np[neuron_idx])
            metadata.append(
                {
                    "global_id": global_id,
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "top_token_ids": token_ids,
                    "top_tokens": decoded_tokens,
                    "top_logits": top_logits_np[neuron_idx].tolist(),
                    "l2_norm": neuron_norm
                }
            )
            global_id += 1

    semantic_embeddings = np.asarray(semantic_embeddings, dtype=np.float32)
    return semantic_embeddings, metadata, tokenizer


def print_steered_neurons_info(metadata, config, title="Target Neurons Info"):
    """
    針對 multi_layer_steering_config 中指定的神經元，
    印出其 L2 Norm、Max Logit 以及 Top Tokens。
    """
    print("\n" + "=" * 115)
    print(title)
    print("=" * 115)
    print(f"{'Layer':<6} | {'Neuron':<7} | {'Config Target':<14} | {'Norm (L2)':<10} | {'Max Logit':<10} | {'Top Tokens'}")
    print("-" * 115)

    # 建立查找表，方便快速抓取特定層與神經元的資訊
    lookup = {(m["layer"], m["neuron"]): m for m in metadata}

    for layer_idx, neurons in sorted(config.items()):
        for neuron_idx, strength in sorted(neurons.items()):
            info = lookup.get((layer_idx, neuron_idx))
            if info:
                tokens_str = ", ".join(info["top_tokens"])
                max_logit = info["top_logits"][0]
                l2_norm = info["l2_norm"]
                print(f"L{layer_idx:<5} | {neuron_idx:<7} | {strength:<14.2f} | {l2_norm:<10.4f} | {max_logit:<10.4f} | [{tokens_str}]")
            else:
                print(f"L{layer_idx:<5} | {neuron_idx:<7} | {strength:<14.2f} | Not found in metadata")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_height"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Loading policy: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.eval()
    policy.to(device)

    # --- ⚡ FULL-MODEL VLA STEERING SETUP ⚡ ---
    print("\n--- ⚡ FULL-MODEL VLA STEERING SETUP ⚡ ---")
    red_strength = 0  #-16
    green_strength = 0

    multi_layer_steering_config = {
            1: {1986: red_strength, 67: red_strength, 798: green_strength},
            4: {688: red_strength},
            7: {230: green_strength},
            8: {216: red_strength},
            9: {863: green_strength},
            11:{1182: green_strength},
            13:{268: red_strength},
    }
    # ---------------------------------------------------------
    # 1. 取得干預前 (BEFORE) 的特徵與 Logits
    # ---------------------------------------------------------
    print("[*] Extracting embeddings BEFORE steering...")
    _, metadata_before, _ = extract_semantic_embeddings_from_policy(policy, top_k_tokens=5, device=device)
    
    # 顯示干預前的結果
    print_steered_neurons_info(metadata_before, multi_layer_steering_config, title="[BEFORE] Steered Neurons: Logits & Top Tokens")

    # ---------------------------------------------------------
    # 2. 進行權重干預
    # ---------------------------------------------------------
    if hasattr(policy, "apply_steering_vector"):
        policy.apply_steering_vector(
            multi_layer_clusters=multi_layer_steering_config
        )
        print("[INFO] apply_steering_vector executed successfully.")
    else:
        print("[WARN] Policy does not have 'apply_steering_vector' method.")

    # ---------------------------------------------------------
    # 3. 取得干預後 (AFTER) 的特徵與 Logits
    # ---------------------------------------------------------
    print("[*] Extracting embeddings AFTER steering...")
    _, metadata_after, _ = extract_semantic_embeddings_from_policy(policy, top_k_tokens=5, device=device)
    
    # 顯示干預後的結果
    print_steered_neurons_info(metadata_after, multi_layer_steering_config, title="[AFTER] Steered Neurons: Logits & Top Tokens")