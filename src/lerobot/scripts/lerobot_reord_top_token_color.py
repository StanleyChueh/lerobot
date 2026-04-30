import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import re

def normalize_token(token):
    token = token.strip().lower()
    token = token.replace("Ġ", "")
    token = token.replace("▁", "")
    token = token.strip(".,;:!?()[]{}'\"`-_/\\")
    return token

@torch.no_grad()
def load_smolvla_and_extract_semantic_embeddings(
    policy_path="ethanCSL/svla_koch_sorting_n_stacking_vla_steering",
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


def rank_color_neurons(
    metadata,
    target_color,
    other_colors=None,
    top_n=10,
    max_color_rank=3,
    min_color_hits=1,
    min_clean_ratio=0.20,
):
    """
    Stricter color-neuron selector.

    Keeps neurons only if:
      1. target color appears in top max_color_rank tokens
      2. target color appears at least min_color_hits times
      3. no other color appears
      4. top tokens are not too noisy
    """
    if other_colors is None:
        other_colors = []

    target_color = target_color.lower()
    other_colors = set([c.lower() for c in other_colors])

    # Optional synonyms. Add more if useful.
    color_aliases = {
        "red": {"red", "reddish", "scarlet", "crimson"},
        "black": {"black", "blackish"},
    }

    target_words = color_aliases.get(target_color, {target_color})

    candidates = []

    for neuron in metadata:
        top_tokens = neuron["top_tokens"]
        top_logits = neuron["top_logits"]

        target_hits = []
        other_hits = []
        clean_token_count = 0

        for rank, (token, logit) in enumerate(zip(top_tokens, top_logits)):
            token_clean = normalize_token(token)

            if token_clean == "":
                continue

            # Count readable word-like tokens.
            if token_clean.isalpha() and len(token_clean) >= 3:
                clean_token_count += 1

            if token_clean in target_words:
                target_hits.append({
                    "rank": rank,
                    "token": token,
                    "logit": float(logit),
                })

            if token_clean in other_colors:
                other_hits.append({
                    "rank": rank,
                    "token": token,
                    "logit": float(logit),
                })

        if len(target_hits) == 0:
            continue

        best_hit = min(target_hits, key=lambda x: x["rank"])
        best_rank = best_hit["rank"] + 1
        best_logit = best_hit["logit"]

        clean_ratio = clean_token_count / max(len(top_tokens), 1)

        # Hard filters.
        if best_rank > max_color_rank:
            continue

        if len(target_hits) < min_color_hits:
            continue

        if len(other_hits) > 0:
            continue

        if clean_ratio < min_clean_ratio:
            continue

        # Strongly prefer:
        # - color at rank 1
        # - repeated color hits
        # - high logit
        # - cleaner token list
        color_score = (
            best_logit * (1.0 / best_rank)
            + 0.25 * len(target_hits)
            + 0.25 * clean_ratio
        )

        candidates.append({
            "layer": neuron["layer"],
            "neuron": neuron["neuron"],
            "target_color": target_color,
            "best_color_rank": best_rank,
            "best_color_logit": best_logit,
            "num_color_hits": len(target_hits),
            "num_other_color_hits": len(other_hits),
            "clean_ratio": clean_ratio,
            "color_score": color_score,
            "matched_tokens": [h["token"] for h in target_hits],
            "other_color_tokens": [h["token"] for h in other_hits],
            "top_tokens": top_tokens,
        })

    candidates.sort(
        key=lambda x: (
            x["best_color_rank"] == 1,
            x["num_color_hits"],
            x["color_score"],
            x["clean_ratio"],
        ),
        reverse=True,
    )

    return candidates[:top_n]

def print_color_ranking_summary(results):
    print("\n" + "=" * 110)
    print("Top Neurons by Color Token Ranking")
    print("=" * 110)

    for concept, ranked_neurons in results.items():
        print(f"\n[INTERVENTION TASK] {concept}")
        print(
            f"{'Rank':<5} | "
            f"{'Color Score':<12} | "
            f"{'Color Rank':<10} | "
            f"{'Clean':<8} | "
            f"{'Logit':<8} | "
            f"{'Layer':<5} | "
            f"{'Neuron':<6} | "
            f"{'Matched':<12} | "
            f"{'Top Tokens'}"
        )
        print("-" * 110)

        for rank, neuron in enumerate(ranked_neurons, start=1):
            tokens_str = ", ".join(neuron["top_tokens"])
            matched_str = ", ".join(neuron["matched_tokens"])

            print(
                f"{rank:<5} | "
                f"{neuron['color_score']:<12.4f} | "
                f"{neuron['best_color_rank']:<10} | "
                f"{neuron['clean_ratio']:<8.2f} | "
                f"{neuron['best_color_logit']:<8.4f} | "
                f"L{neuron['layer']:<4} | "
                f"{neuron['neuron']:<6} | "
                f"{matched_str:<12} | "
                f"[{tokens_str}]"
            )

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

if __name__ == "__main__":
    policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_height"
    
    bundle = load_smolvla_and_extract_semantic_embeddings(
        policy_path=policy_path,
        top_k_tokens=10, 
    )

    red_neurons = rank_color_neurons(
        metadata=bundle["metadata"],
        target_color="red",
        other_colors=["black", "blue", "green", "yellow", "white"],
        top_n=10,
        max_color_rank=2,
        min_color_hits=2,
        min_clean_ratio=0.50,
    )

    black_neurons = rank_color_neurons(
        metadata=bundle["metadata"],
        target_color="black",
        other_colors=["red", "blue", "green", "yellow", "white"],
        top_n=10,
        max_color_rank=3,
        min_color_hits=1,
        min_clean_ratio=0.40,
    )

    color_ranking_results = {
        "Red Object": red_neurons,
        "Black Object": black_neurons,
    }

    print_color_ranking_summary(color_ranking_results)
    
