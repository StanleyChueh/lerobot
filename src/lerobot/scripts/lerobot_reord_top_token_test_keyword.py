import torch
import numpy as np
import json

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

@torch.no_grad()
def load_smolvla_and_extract_semantic_embeddings(
    policy_path="ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2_unfrozen",
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


def normalize_token(token: str) -> str:
    """
    Normalize tokenizer artifacts and case for keyword matching.
    """
    token = token.strip()
    token = token.replace("Ġ", "")
    token = token.replace("▁", "")
    token = token.replace("Ċ", "")
    token = token.strip()
    return token


def token_matches_keyword(token: str, keyword: str) -> bool:
    """
    Conservative matching:
      - exact match
      - simple morphological prefix match for low/high/fast/slow variants
      - avoids arbitrary substring matching unless the keyword is non-ASCII
    """
    token_norm = normalize_token(token)
    token_lower = token_norm.lower()
    keyword_lower = keyword.lower()

    # Non-ASCII tokens, e.g. 高
    if any(ord(ch) > 127 for ch in keyword):
        return keyword in token_norm

    if token_lower == keyword_lower:
        return True

    # Allow simple morphological variants.
    # Example: high -> higher/highest/highs, low -> lower/lowest/lows
    allowed_prefix_roots = {
        "low": ("low", "lower", "lowest", "lows"),
        "high": ("high", "higher", "highest", "highs", "hight"),
        "slow": ("slow", "slowly", "slowest"),
        "fast": ("fast", "faster", "fastest"),
        "safe": ("safe", "safely", "safety", "safer"),
        "risk": ("risk", "risky", "risks"),
    }

    if keyword_lower in allowed_prefix_roots:
        return token_lower in allowed_prefix_roots[keyword_lower]

    return False


def score_neuron_by_keywords(neuron, keywords_dict):
    pos = keywords_dict.get("pos", {})
    soft_pos = keywords_dict.get("soft_pos", {})
    neg = keywords_dict.get("neg", {})

    # Backward compatibility: allow list format.
    if isinstance(pos, list):
        pos = {kw: 1.0 for kw in pos}
    if isinstance(soft_pos, list):
        soft_pos = {kw: 0.5 for kw in soft_pos}
    if isinstance(neg, list):
        neg = {kw: 1.0 for kw in neg}

    score = 0.0
    hard_match_count = 0
    soft_match_count = 0
    neg_match_count = 0
    matched_tokens = []
    negative_tokens = []

    for token in neuron["top_tokens"]:
        token_norm = normalize_token(token)
        token_lower = token_norm.lower()

        # Negative penalty first.
        neg_hit = False
        for neg_kw, neg_weight in neg.items():
            # For negative filters, substring matching is useful:
            # "yellow" contains "low", "thigh" contains "high".
            if neg_kw.lower() in token_lower:
                score -= float(neg_weight)
                neg_match_count += 1
                negative_tokens.append(token)
                neg_hit = True
                break

        if neg_hit:
            continue

        for kw, weight in pos.items():
            if token_matches_keyword(token_norm, kw):
                score += float(weight)
                hard_match_count += 1
                matched_tokens.append(token)
                break

        for kw, weight in soft_pos.items():
            if token_matches_keyword(token_norm, kw):
                score += float(weight)
                soft_match_count += 1
                matched_tokens.append(token)
                break

    return {
        "score": score,
        "hard_match_count": hard_match_count,
        "soft_match_count": soft_match_count,
        "neg_match_count": neg_match_count,
        "matched_tokens": matched_tokens,
        "negative_tokens": negative_tokens,
    }


def rank_neurons_by_keyword_frequency(
    metadata,
    keywords_dict,
    top_n=6,
    min_score=1.0,
    min_hard_match_count=2,
    min_purity=0.20,
    max_neg_match_count=0,
    require_hard_match=True,
):
    """
    Paper C.3-style keyword selection, stricter version.

    Goal:
      - Select neurons whose top-k tokens are semantically concentrated.
      - Reject neurons that contain opposite-concept / negative tokens.
      - Avoid exporting weak or noisy neurons.

    Args:
        metadata:
            List of neuron metadata from value-vector top-token projection.

        keywords_dict:
            Dict with:
              - pos: weighted hard-positive keywords
              - soft_pos: weighted weak-positive keywords
              - neg: weighted negative / opposite keywords

        top_n:
            Number of neurons to return. Paper UR5-style setting uses top_n=6.

        min_score:
            Minimum weighted score.

        min_hard_match_count:
            Minimum number of direct concept matches.
            For low/high, start with 2.
            For stricter selection, use 3.

        min_purity:
            hard_match_count / number_of_top_tokens.
            With top_k_tokens=10:
              0.20 means at least 2/10 direct matches.
              0.30 means at least 3/10 direct matches.

        max_neg_match_count:
            Maximum allowed negative/opposite matches.
            For low/high, use 0.

        require_hard_match:
            If True, soft_pos alone cannot select a neuron.
    """
    scored_neurons = []

    for neuron in metadata:
        score_info = score_neuron_by_keywords(neuron, keywords_dict)

        top_tokens = neuron["top_tokens"]
        top_k = len(top_tokens)
        purity = score_info["hard_match_count"] / max(top_k, 1)

        # 1. Basic weighted-score threshold.
        if score_info["score"] < min_score:
            continue

        # 2. Require real concept tokens, not only soft semantic words.
        if require_hard_match and score_info["hard_match_count"] == 0:
            continue

        # 3. Require enough direct semantic evidence.
        if score_info["hard_match_count"] < min_hard_match_count:
            continue

        # 4. Reject mixed/opposite neurons.
        # Example:
        #   Low neuron with "high" token should be rejected.
        #   High neuron with "low/lower" token should be rejected.
        if score_info["neg_match_count"] > max_neg_match_count:
            continue

        # 5. Require semantic purity among top-k tokens.
        if purity < min_purity:
            continue

        scored_neurons.append({
            "layer": int(neuron["layer"]),
            "neuron": int(neuron["neuron"]),
            "score": float(score_info["score"]),
            "purity": float(purity),
            "hard_match_count": int(score_info["hard_match_count"]),
            "soft_match_count": int(score_info["soft_match_count"]),
            "neg_match_count": int(score_info["neg_match_count"]),
            "matched_tokens": score_info["matched_tokens"],
            "negative_tokens": score_info["negative_tokens"],
            "top_tokens": top_tokens,
            "max_logit": float(neuron["top_logits"][0]),
        })

    # Prefer:
    #   1. more hard concept matches
    #   2. higher purity
    #   3. higher weighted score
    #   4. stronger value-vector logit
    scored_neurons.sort(
        key=lambda x: (
            x["hard_match_count"],
            x["purity"],
            x["score"],
            x["soft_match_count"],
            x["max_logit"],
        ),
        reverse=True,
    )

    ranked_results = []
    for rank, neuron in enumerate(scored_neurons[:top_n]):
        ranked_results.append({
            "rank": rank + 1,
            "score": neuron["score"],
            "purity": neuron["purity"],
            "hard_match_count": neuron["hard_match_count"],
            "soft_match_count": neuron["soft_match_count"],
            "neg_match_count": neuron["neg_match_count"],
            "layer": neuron["layer"],
            "neuron": neuron["neuron"],
            "matched_tokens": neuron["matched_tokens"],
            "negative_tokens": neuron["negative_tokens"],
            "top_tokens": neuron["top_tokens"],
        })

    return ranked_results

def run_keyword_based_ranking(
    metadata,
    concept_keywords_map,
    top_n=6,
    min_score=1.0,
    min_hard_match_count=2,
    min_purity=0.20,
    max_neg_match_count=0,
    require_hard_match=True,
):
    """
    Run weighted keyword ranking for multiple intervention concepts.
    """
    results = {}

    for concept_name, keywords_dict in concept_keywords_map.items():
        ranked_neurons = rank_neurons_by_keyword_frequency(
            metadata=metadata,
            keywords_dict=keywords_dict,
            top_n=top_n,
            min_score=min_score,
            min_hard_match_count=min_hard_match_count,
            min_purity=min_purity,
            max_neg_match_count=max_neg_match_count,
            require_hard_match=require_hard_match,
        )
        results[concept_name] = ranked_neurons

    return results


def has_negative_match(neuron, keywords_dict):
    """
    Check whether a neuron contains any negative / opposite-concept token.
    Example:
      Low Transport should reject tokens containing high/higher/highest.
      High Transport should reject tokens containing low/lower/lowest.
    """
    neg = keywords_dict.get("neg", {})

    if isinstance(neg, list):
        neg = {kw: 1.0 for kw in neg}

    for token in neuron["top_tokens"]:
        token_lower = normalize_token(token).lower()
        for neg_kw in neg.keys():
            if neg_kw.lower() in token_lower:
                return True

    return False


def build_seed_indices_from_keyword_results(metadata, keyword_results_for_concept):
    """
    Convert strict keyword-selected neurons into metadata indices.

    keyword_results_for_concept format:
      [
        {"layer": 5, "neuron": 1877, ...},
        {"layer": 10, "neuron": 2349, ...}
      ]
    """
    seed_pairs = {
        (int(item["layer"]), int(item["neuron"]))
        for item in keyword_results_for_concept
    }

    seed_indices = []
    for idx, item in enumerate(metadata):
        pair = (int(item["layer"]), int(item["neuron"]))
        if pair in seed_pairs:
            seed_indices.append(idx)

    return seed_indices


def seeded_knn_expand_concept(
    semantic_embeddings,
    metadata,
    keyword_results_for_concept,
    keywords_dict,
    top_n=6,
    candidate_pool_size=100,
    min_seed_similarity=0.30,
    require_no_negative=True,
):
    """
    Seeded kNN neuron selection.

    Logic:
      1. Use strict keyword-selected neurons as clean semantic seeds.
      2. Average seed semantic embeddings into one centroid.
      3. Rank all neurons by cosine similarity to the seed centroid.
      4. Reject neurons with opposite/negative tokens.
      5. Return top_n neurons for steering.

    This is useful when strict keyword selection finds fewer than 6 clean neurons.
    """
    seed_indices = build_seed_indices_from_keyword_results(
        metadata=metadata,
        keyword_results_for_concept=keyword_results_for_concept,
    )

    if len(seed_indices) == 0:
        raise ValueError(
            "No strict keyword seeds found. "
            "Lower min_hard_match_count/min_purity first, or use pure concept-kNN."
        )

    seed_embs = semantic_embeddings[seed_indices]
    centroid = seed_embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    # semantic_embeddings are already L2-normalized in your extraction function,
    # so dot product is cosine similarity.
    similarities = semantic_embeddings @ centroid
    sorted_indices = np.argsort(similarities)[::-1]

    ranked = []
    seen_pairs = set()

    for idx in sorted_indices[:candidate_pool_size]:
        idx = int(idx)
        neuron = metadata[idx]
        layer = int(neuron["layer"])
        neuron_id = int(neuron["neuron"])
        pair = (layer, neuron_id)

        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        sim = float(similarities[idx])

        if sim < min_seed_similarity:
            continue

        if require_no_negative and has_negative_match(neuron, keywords_dict):
            continue

        score_info = score_neuron_by_keywords(neuron, keywords_dict)

        top_tokens = neuron["top_tokens"]
        top_k = len(top_tokens)
        purity = score_info["hard_match_count"] / max(top_k, 1)

        ranked.append({
            "layer": layer,
            "neuron": neuron_id,
            "seed_similarity": sim,
            "keyword_score": float(score_info["score"]),
            "purity": float(purity),
            "hard_match_count": int(score_info["hard_match_count"]),
            "soft_match_count": int(score_info["soft_match_count"]),
            "neg_match_count": int(score_info["neg_match_count"]),
            "matched_tokens": score_info["matched_tokens"],
            "negative_tokens": score_info["negative_tokens"],
            "top_tokens": neuron["top_tokens"],
        })

    ranked.sort(
        key=lambda x: (
            x["seed_similarity"],
            x["hard_match_count"],
            x["keyword_score"],
            x["purity"],
        ),
        reverse=True,
    )

    output = []
    for rank, item in enumerate(ranked[:top_n]):
        output.append({
            "rank": rank + 1,
            **item,
        })

    return output


def run_seeded_knn_selection(
    semantic_embeddings,
    metadata,
    keyword_ranking_results,
    concept_keywords_map,
    top_n=6,
    candidate_pool_size=100,
    min_seed_similarity=0.30,
):
    """
    Run seeded kNN for all concepts using strict keyword results as seeds.
    """
    results = {}

    for concept_name, keyword_results_for_concept in keyword_ranking_results.items():
        keywords_dict = concept_keywords_map[concept_name]

        results[concept_name] = seeded_knn_expand_concept(
            semantic_embeddings=semantic_embeddings,
            metadata=metadata,
            keyword_results_for_concept=keyword_results_for_concept,
            keywords_dict=keywords_dict,
            top_n=top_n,
            candidate_pool_size=candidate_pool_size,
            min_seed_similarity=min_seed_similarity,
            require_no_negative=True,
        )

    return results


def print_seeded_knn_summary(results):
    print("\n" + "=" * 140)
    print("Top Neurons by Seeded kNN Expansion")
    print("=" * 140)

    for concept, ranked_neurons in results.items():
        print(f"\n[INTERVENTION TASK] {concept}")
        print(
            f"{'Rank':<5} | {'Sim':<7} | {'KScore':<7} | {'Purity':<7} | "
            f"{'Hard':<5} | {'Soft':<5} | {'Neg':<4} | {'Layer':<5} | {'Neuron':<7} | {'Matched Tokens'}"
        )
        print("-" * 140)

        if len(ranked_neurons) == 0:
            print("No neurons passed seeded-kNN threshold.")
            continue

        for neuron in ranked_neurons:
            matched = ", ".join(neuron["matched_tokens"])
            print(
                f"{neuron['rank']:<5} | "
                f"{neuron['seed_similarity']:<7.3f} | "
                f"{neuron['keyword_score']:<7.2f} | "
                f"{neuron['purity']:<7.2f} | "
                f"{neuron['hard_match_count']:<5} | "
                f"{neuron['soft_match_count']:<5} | "
                f"{neuron['neg_match_count']:<4} | "
                f"L{neuron['layer']:<4} | "
                f"{neuron['neuron']:<7} | "
                f"[{matched}]"
            )


def print_keyword_ranking_summary(results):
    print("\n" + "=" * 130)
    print("Top Neurons by Strict Weighted Keyword Score (Paper C.3-style Selection)")
    print("=" * 130)

    for concept, ranked_neurons in results.items():
        print(f"\n[INTERVENTION TASK] {concept}")
        print(
            f"{'Rank':<5} | {'Score':<7} | {'Purity':<7} | {'Hard':<5} | "
            f"{'Soft':<5} | {'Neg':<4} | {'Layer':<5} | {'Neuron':<7} | {'Matched Tokens'}"
        )
        print("-" * 130)

        if len(ranked_neurons) == 0:
            print("No neurons passed the strict keyword threshold.")
            continue

        for neuron in ranked_neurons:
            matched = ", ".join(neuron["matched_tokens"])
            print(
                f"{neuron['rank']:<5} | "
                f"{neuron['score']:<7.2f} | "
                f"{neuron['purity']:<7.2f} | "
                f"{neuron['hard_match_count']:<5} | "
                f"{neuron['soft_match_count']:<5} | "
                f"{neuron['neg_match_count']:<4} | "
                f"L{neuron['layer']:<4} | "
                f"{neuron['neuron']:<7} | "
                f"[{matched}]"
            )

###############################################################################
'''
    Save neurons list as json for easier real-time steering
'''

def convert_keyword_results_to_steering_dict(keyword_results):
    output = {}
    for concept_name, ranked_neurons in keyword_results.items():
        steering_dict = {}
        for item in ranked_neurons:
            layer = int(item["layer"])
            neuron = int(item["neuron"])
            steering_dict.setdefault(layer, []).append(neuron)

        output[concept_name] = {
            str(layer): sorted(set(neurons))
            for layer, neurons in sorted(steering_dict.items())
        }

    return output

###############################################################################

if __name__ == "__main__":
    #policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2"
    policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2_unfrozen"
    
    bundle = load_smolvla_and_extract_semantic_embeddings(
        policy_path=policy_path,
        top_k_tokens=10, 
    )
    
    # 升級版的字典結構：加入 neg (負向排除字)
    intervention_keywords_map = {
        "Low Transport": {
            "pos": {
                "low": 3.0,
                "lower": 3.0,
                "lowest": 3.0,
                "lows": 2.0,
            },
            "soft_pos": {
                "down": 0.5,
                "downward": 0.75,
                "below": 0.75,
            },
            "neg": {
                # false substring
                "follow": 5.0,
                "allow": 5.0,
                "slow": 5.0,
                "blow": 5.0,
                "glow": 5.0,
                "yellow": 5.0,
                "hollow": 5.0,

                # opposite concept
                "high": 6.0,
                "higher": 6.0,
                "highest": 6.0,
                "up": 3.0,
                "above": 3.0,
                "tall": 3.0,
            },
        },

        "High Transport": {
            "pos": {
                "high": 3.0,
                "higher": 3.0,
                "highest": 3.0,
                "highs": 2.0,
                "hight": 1.5,
                "高": 1.5,
            },
            "soft_pos": {
                "up": 0.5,
                "upward": 0.75,
                "above": 0.75,
                "raise": 0.75,
                "raised": 0.75,
                "elevated": 0.75,
            },
            "neg": {
                # false substring
                "thigh": 5.0,
                "highway": 5.0,
                "highlight": 5.0,
                "highschool": 5.0,
                "highly": 4.0,

                # opposite concept
                "low": 6.0,
                "lower": 6.0,
                "lowest": 6.0,
                "down": 3.0,
                "below": 3.0,
                "floor": 3.0,
                "ground": 3.0,
            },
        },
    }
        
    # 執行關鍵字頻率排名
    # 1. Strict keyword seeds.
    # This is the closest to paper C.3 physical-robot keyword selection.
    keyword_ranking_results = run_keyword_based_ranking(
        metadata=bundle["metadata"],
        concept_keywords_map=intervention_keywords_map,
        top_n=6,
        min_score=1.0,
        min_hard_match_count=3,
        min_purity=0.30,
        max_neg_match_count=0,
        require_hard_match=True,
    )

    print_keyword_ranking_summary(keyword_ranking_results)
    print_top_neurons_overall_by_logit(bundle["metadata"], top_n=15)

    # Save strict keyword results.
    keyword_steering_neurons = convert_keyword_results_to_steering_dict(keyword_ranking_results)

    with open("keyword_strict_steering_neurons.json", "w", encoding="utf-8") as f:
        json.dump(keyword_steering_neurons, f, indent=2, ensure_ascii=False)

    with open("keyword_strict_steering_neurons_debug.json", "w", encoding="utf-8") as f:
        json.dump(keyword_ranking_results, f, indent=2, ensure_ascii=False)

    print("[*] Saved strict keyword neurons to keyword_strict_steering_neurons.json")
    print("[*] Saved strict keyword debug to keyword_strict_steering_neurons_debug.json")

    # 2. Seeded kNN expansion.
    # Use clean keyword neurons as seeds, then expand to top_n=6 via semantic embedding similarity.
    seeded_knn_results = run_seeded_knn_selection(
        semantic_embeddings=bundle["semantic_embeddings"],
        metadata=bundle["metadata"],
        keyword_ranking_results=keyword_ranking_results,
        concept_keywords_map=intervention_keywords_map,
        top_n=6,
        candidate_pool_size=100,
        min_seed_similarity=0.30,
    )

    print_seeded_knn_summary(seeded_knn_results)

    seeded_knn_steering_neurons = convert_keyword_results_to_steering_dict(seeded_knn_results)

    with open("seeded_knn_steering_neurons.json", "w", encoding="utf-8") as f:
        json.dump(seeded_knn_steering_neurons, f, indent=2, ensure_ascii=False)

    with open("seeded_knn_steering_neurons_debug.json", "w", encoding="utf-8") as f:
        json.dump(seeded_knn_results, f, indent=2, ensure_ascii=False)

    print("[*] Saved seeded-kNN neurons to seeded_knn_steering_neurons.json")
    print("[*] Saved seeded-kNN debug to seeded_knn_steering_neurons_debug.json")

