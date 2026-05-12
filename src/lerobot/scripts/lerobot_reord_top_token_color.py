import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import re
import math

@torch.no_grad()
def load_smolvla_and_extract_semantic_embeddings(
    policy_path="ethanCSL/svla_koch_pick_n_place_vla_steering_color_unfrozen",
    top_k_tokens=5,
    device=None,
):
    """
    Paper-like semantic embedding extraction:
      1) take FFN value vectors from down_proj columns
      2) project each value vector into output token space via lm_head
      3) take top-k tokens
      4) build a semantic embedding by softmax-weighted averaging
         the corresponding output token embeddings
    """
    print(f"[*] Loading policy: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    vlm_model = policy.model.vlm_with_expert.vlm
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    # Output embedding / LM head weight: [vocab_size, hidden_dim]
    W_out = vlm_model.lm_head.weight.detach().to(device=device, dtype=torch.float32)

    semantic_embeddings = []
    metadata = []

    global_id = 0
    num_layers = len(text_model.layers)

    print(f"[*] Extracting value-vector semantic embeddings from {num_layers} layers...")

    for layer_idx in range(num_layers):
        # PyTorch Linear weight shape: [out_features, in_features]
        # down_proj: [hidden_dim, intermediate_dim]
        # each COLUMN is one FFN value vector
        W_value = text_model.layers[layer_idx].mlp.down_proj.weight.detach().to(
            device=device, dtype=torch.float32
        )  # [d_model, d_ff]

        # Token logits induced by each value vector:
        # [vocab_size, d_ff]
        token_logits = W_out @ W_value

        # Top-k tokens per neuron/value vector
        top_logits, top_token_ids = torch.topk(token_logits, k=top_k_tokens, dim=0)
        # shapes: [k, d_ff], [k, d_ff]

        # Rearrange to [d_ff, k]
        top_logits_t = top_logits.transpose(0, 1).contiguous()
        top_token_ids_t = top_token_ids.transpose(0, 1).contiguous()

        # Softmax weights over the top-k tokens for each value vector
        weights = torch.softmax(top_logits_t, dim=1)  # [d_ff, k]

        # Output token embeddings of the top-k tokens
        # W_out[top_token_ids_t] -> [d_ff, k, d_model]
        token_embs = W_out[top_token_ids_t]

        # Paper-like semantic embedding:
        # e_sem^(i) = sum_j softmax(logit_j) * W_j
        e_sem = (weights.unsqueeze(-1) * token_embs).sum(dim=1)  # [d_ff, d_model]

        # Normalize for cosine comparisons
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

            semantic_embeddings.append(e_sem_np[neuron_idx])
            metadata.append(
                {
                    "global_id": global_id,
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "top_token_ids": token_ids,
                    "top_tokens": decoded_tokens,
                    "top_logits": top_logits_np[neuron_idx].tolist(),
                }
            )
            global_id += 1

    semantic_embeddings = np.asarray(semantic_embeddings, dtype=np.float32)
    print(f"[*] Extracted {len(semantic_embeddings)} semantic embeddings.")

    return {
        "policy": policy,
        "tokenizer": tokenizer,
        "W_out": W_out,
        "semantic_embeddings": semantic_embeddings,
        "metadata": metadata,
    }


@torch.no_grad()
def build_concept_embedding(concept_text, tokenizer, W_out, device):
    """
    Tokenize the concept word/phrase and average output embeddings.
    """
    enc = tokenizer(concept_text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"][0].to(device)

    if input_ids.numel() == 0:
        raise ValueError(f"Tokenizer produced no tokens for concept: {concept_text}")

    concept_vec = W_out[input_ids].mean(dim=0)
    concept_vec = torch.nn.functional.normalize(concept_vec, p=2, dim=0)

    return concept_vec.cpu().numpy()

def print_top_neurons_overall_by_logit(metadata, top_n=15):
    """
    不依賴任何輸入概念。
    直接掃描模型中「所有層、所有神經元」，根據神經元對其 Top 1 Token 的 Logit 進行全局排序。
    Logit 越高，代表該神經元對特定詞彙的推動力量越強、語意越鮮明。
    """
    print("\n" + "=" * 90)
    print(f"Top {top_n} Strongest Neurons Overall (By Max Token Logit)")
    print("=" * 90)

    # 複製一份 metadata，包含模型中所有的神經元資訊
    all_neurons = list(metadata)

    # 依照每個神經元的最高 logit (即 top_logits 陣列的第一個值) 進行降冪排序
    all_neurons.sort(key=lambda x: x["top_logits"][0], reverse=True)

    print(f"{'Global Rank':<12} | {'Max Logit':<10} | {'Layer':<5} | {'Neuron':<6} | {'Top Tokens'}")
    print("-" * 90)
    
    for i, neuron in enumerate(all_neurons[:top_n]):
        tokens_str = ", ".join(neuron["top_tokens"])
        max_logit = neuron["top_logits"][0]
        print(f"{i+1:<12} | {max_logit:.4f}    | L{neuron['layer']:<4} | {neuron['neuron']:<6} | [{tokens_str}]")


def rank_neurons_by_keyword_frequency(metadata, keywords_dict, top_n=6):
    pos_keywords = [kw.lower() for kw in keywords_dict.get("pos", [])]
    neg_keywords = [kw.lower() for kw in keywords_dict.get("neg", [])]

    scored_neurons = []
    for neuron in metadata:
        match_count = 0
        for token in neuron["top_tokens"]:
            # Strip punctuation and whitespace to handle tokens like "Red," or " red"
            token_clean = token.lower().strip(" .,!?;:\"'()[]{}")
            
            # 1. Negative filter (keep as is)
            if any(neg_kw in token_clean for neg_kw in neg_keywords):
                continue 

            # 2. STRICT Match: Use equality instead of 'in'
            # This prevents "swered" or "paralleled" from matching "red"
            if any(token_clean == pos_kw for pos_kw in pos_keywords):
                match_count += 1
        
        scored_neurons.append({
            "layer": neuron["layer"],
            "neuron": neuron["neuron"],
            "match_count": match_count,
            "top_tokens": neuron["top_tokens"],
            "max_logit": neuron["top_logits"][0] 
        })

    # 排序邏輯：優先比較 match_count (降冪)，若次數相同，則比較 max_logit (降冪)
    scored_neurons.sort(key=lambda x: (x["match_count"], x["max_logit"]), reverse=True)

    # 提取前 top_n 名
    ranked_results = []
    for rank, neuron in enumerate(scored_neurons[:top_n]):
        ranked_results.append({
            "rank": rank + 1,
            "match_count": neuron["match_count"],
            "layer": neuron["layer"],
            "neuron": neuron["neuron"],
            "top_tokens": neuron["top_tokens"]
        })
    
    return ranked_results

def run_keyword_based_ranking(metadata, concept_keywords_map, top_n=6):
    """
    處理多個概念的關鍵字搜尋並彙整結果。
    """
    results = {}
    for concept_name, keywords_dict in concept_keywords_map.items():
        ranked_neurons = rank_neurons_by_keyword_frequency(
            metadata=metadata,
            keywords_dict=keywords_dict,  # 這裡傳入包含 pos 和 neg 的字典
            top_n=top_n
        )
        results[concept_name] = ranked_neurons
    return results


def print_keyword_ranking_summary(results):
    print("\n" + "=" * 95)
    print("Top Neurons by Keyword Frequency (Paper Section C.3 Baseline)")
    print("=" * 95)

    for concept, ranked_neurons in results.items():
        print(f"\n[INTERVENTION TASK] {concept}")
        print(f"{'Rank':<5} | {'Match Count':<11} | {'Layer':<5} | {'Neuron':<6} | {'Top Tokens'}")
        print("-" * 95)
        for neuron in ranked_neurons:
            tokens_str = ", ".join(neuron["top_tokens"])
            print(f"{neuron['rank']:<5} | {neuron['match_count']:<11} | L{neuron['layer']:<4} | {neuron['neuron']:<6} | [{tokens_str}]")

def calculate_purity(top_tokens, pos_keywords):
    """
    Calculates the 'Purity': What percentage of top tokens are strictly the concept.
    """
    clean_tokens = [t.lower().strip(" .,!?;:\"'()[]{}") for t in top_tokens]
    matches = sum(1 for t in clean_tokens if any(kw == t for kw in pos_keywords))
    return matches / len(top_tokens)

def run_pure_neuron_search(bundle, concept_map, top_n=6):
    metadata = bundle["metadata"]
    semantic_embeddings = bundle["semantic_embeddings"]
    tokenizer = bundle["tokenizer"]
    W_out = bundle["W_out"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}

    for concept_name, config in concept_map.items():
        # Get vector for just the first word (e.g., 'Red')
        concept_vec = build_concept_embedding(concept_name.split()[0], tokenizer, W_out, device)
        
        scored_neurons = []
        for i, neuron in enumerate(metadata):
            # LAYER PRUNING: Focus on the 'semantic belly' (L4 to L14)
            # This removes shallow L0-L1 noise and final action-bin noise.
            if not (4 <= neuron["layer"] <= 14):
                continue
                
            # Cosine Similarity
            sim = float(semantic_embeddings[i] @ concept_vec)
            # Purity Score
            purity = calculate_purity(neuron["top_tokens"], config["pos"])
            
            # Combined Score: Similarity weighted heavily by Purity
            combined_score = sim * (purity ** 2) 
            
            if purity > 0.1: # Threshold to filter out random noise
                scored_neurons.append({
                    "rank": 0, # Placeholder
                    "match_count": int(purity * len(neuron["top_tokens"])),
                    "layer": neuron["layer"],
                    "neuron": neuron["neuron"],
                    "top_tokens": neuron["top_tokens"],
                    "score": combined_score,
                    "purity": purity
                })

        # Sort by the new combined score
        scored_neurons.sort(key=lambda x: x["score"], reverse=True)
        for idx, n in enumerate(scored_neurons[:top_n]):
            n["rank"] = idx + 1
        results[concept_name] = scored_neurons[:top_n]
        
    return results

if __name__ == "__main__":
    policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_color_unfrozen"
    
    bundle = load_smolvla_and_extract_semantic_embeddings(
        policy_path=policy_path,
        top_k_tokens=10, # Keeping k=10 is better for finding pure clusters
    )
    
    intervention_keywords_map = {
        "Red Color": {
            "pos": ["red", "crimson", "scarlet", "RED", "Red"],
            "neg": ["predict", "ordered", "hundred"]
        },
        "Green Color": {
            "pos": ["green", "emerald", "lime", "GREEN", "Green"],
            "neg": ["agreement"]
        },
        "Blue Color": {
            "pos": ["blue", "azure", "navy", "BLUE", "Blue"],
            "neg": ["value"]
        }
    }
    
    # NEW: Run the Pure Search
    pure_results = run_pure_neuron_search(
        bundle=bundle,
        concept_map=intervention_keywords_map,
        top_n=6
    )

    # Print using your existing summary function
    print_keyword_ranking_summary(pure_results)
    print_top_neurons_overall_by_logit(bundle["metadata"], top_n=15)
