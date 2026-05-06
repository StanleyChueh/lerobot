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
    policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2_unfrozen"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Loading policy: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.eval()
    policy.to(device)

    multi_layer_steering_config = {
        1: {1222: 10.0},
        2: {826: 0.0},
        3: {369: 0.0, 2003: 10.0},
        5: {1877: 10.0, 1904: 10.0, 2102: 0.0},
        7: {1151: 0.0},
        9: {2554: 0.0},
        13:{414: 0.0},
        15:{1157: 10.0},
    }

    print("[*] Extracting embeddings BEFORE steering...")
    _, metadata_before, _ = extract_semantic_embeddings_from_policy(policy, top_k_tokens=5, device=device)
    
    # 顯示干預前的結果
    print_steered_neurons_info(metadata_before, multi_layer_steering_config, title="[BEFORE] Steered Neurons: Logits & Top Tokens")


    # 1. 萃取干預前的特徵 (這會給我們基礎的 Semantic Embeddings)
    print("[*] Extracting base semantic embeddings...")
    sem_embs, metadata_before, _ = extract_semantic_embeddings_from_policy(policy, top_k_tokens=10, device=device)
    
    # 建立 lookup table: (layer, neuron) -> index_in_sem_embs
    lookup = {(m["layer"], m["neuron"]): i for i, m in enumerate(metadata_before)}

    # 3. 核心邏輯：將「強度」轉換為「縮放後的向量」
    prepared_steering_data = {}
    for layer_idx, neurons in multi_layer_steering_config.items():
        prepared_steering_data[layer_idx] = {}
        for neuron_idx, strength in neurons.items():
            idx = lookup.get((layer_idx, neuron_idx))
            if idx is not None:
                # 取得該神經元的語義方向 (單位向量)
                base_vec = torch.from_numpy(sem_embs[idx]).to(device)
                
                # 根據強度縮放：
                # 如果 strength > 0, 向量變強
                # 如果 strength < 0, 向量反轉縮小 (如你的需求)
                # 如果 strength == 0, 這裡可以決定是要設為 0 還是維持原樣
                multiplier = strength if strength >= 0 else (1.0 / abs(strength))
                
                prepared_steering_data[layer_idx][neuron_idx] = base_vec * multiplier
            else:
                print(f"[!] Warning: Neuron L{layer_idx}_N{neuron_idx} not found.")

    # 4. 執行權重干預 (修正關鍵字參數名稱)
    if hasattr(policy, "apply_steering_vector"):
        policy.apply_steering_vector(
            steering_data=prepared_steering_data  # 這裡名稱要對應函式定義
        )
    else:
        print("[WARN] Policy does not have 'apply_steering_vector' method.")

    # 5. 驗證干預後的結果
    print("[*] Extracting embeddings AFTER steering...")
    _, metadata_after, _ = extract_semantic_embeddings_from_policy(policy, top_k_tokens=5, device=device)
    print_steered_neurons_info(metadata_after, multi_layer_steering_config, title="[AFTER] Steered Neurons")