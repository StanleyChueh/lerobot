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


# @torch.no_grad()
# def build_concept_embedding(concept_text, tokenizer, W_out, device):
#     """
#     Paper says:
#       tokenize the concept word/phrase using the model tokenizer,
#       embed it using the language modeling head.
#     For multi-token phrases, we average their output embeddings.
#     """
#     enc = tokenizer(concept_text, add_special_tokens=False, return_tensors="pt")
#     input_ids = enc["input_ids"][0].to(device)

#     if input_ids.numel() == 0:
#         raise ValueError(f"Tokenizer produced no tokens for concept: {concept_text}")

#     concept_vec = W_out[input_ids].mean(dim=0)
#     concept_vec = torch.nn.functional.normalize(concept_vec, p=2, dim=0)

#     return concept_vec.cpu().numpy()


# def get_pget_partition_indicesartition_indices(num_vectors, partition="full"):
#     if partition == "full":
#         return np.arange(num_vectors)
#     elif partition == "early":
#         return np.arange(num_vectors // 2)
#     elif partition == "late":
#         return np.arange(num_vectors // 2, num_vectors)
#     else:
#         raise ValueError("partition must be one of: full, early, late")


# def find_best_knn_cluster_for_concept(
#     semantic_embeddings,
#     metadata,
#     concept_vec,
#     k=20,
#     partition="full",
# ):
#     """
#     Faithful approximation of the paper's Appendix B.3:
#       - cosine kNN over semantic embeddings
#       - each neighborhood acts as a candidate cluster
#       - centroid = average embedding of cluster members
#       - choose cluster whose centroid is most similar to concept embedding
#     """
#     all_idx = get_partition_indices(len(semantic_embeddings), partition=partition)
#     X = semantic_embeddings[all_idx]

#     # Make sure k is valid
#     k = min(k, len(X))
#     if k < 1:
#         raise ValueError("No vectors available in this partition.")

#     nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
#     nbrs.fit(X)

#     distances, neighbors = nbrs.kneighbors(X)  # neighbors shape [N, k]

#     # Candidate cluster centroids
#     centroids = X[neighbors].mean(axis=1)
#     centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

#     # Concept-centroid cosine similarity
#     scores = centroids @ concept_vec
#     best_local = int(np.argmax(scores))
#     best_score = float(scores[best_local])

#     member_local_idx = neighbors[best_local]
#     member_global_idx = all_idx[member_local_idx]

#     cluster_metadata = [metadata[int(i)] for i in member_global_idx.tolist()]

#     return {
#         "partition": partition,
#         "k": k,
#         "best_score": best_score,
#         "member_indices": member_global_idx.tolist(),
#         "members": cluster_metadata,
#     }


# def run_paper_like_concept_clustering(
#     policy_path,
#     concepts=("slow", "fast", "high", "low"),
#     k_values=(10, 20, 40),
#     partition="full",
#     top_k_tokens=5,
#     device=None,
# ):
#     bundle = load_smolvla_and_extract_semantic_embeddings(
#         policy_path=policy_path,
#         top_k_tokens=top_k_tokens,
#         device=device,
#     )

#     tokenizer = bundle["tokenizer"]
#     W_out = bundle["W_out"]
#     semantic_embeddings = bundle["semantic_embeddings"]
#     metadata = bundle["metadata"]

#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     results = {}

#     for concept in concepts:
#         concept_vec = build_concept_embedding(
#             concept_text=concept,
#             tokenizer=tokenizer,
#             W_out=W_out,
#             device=device,
#         )

#         best_overall = None
#         for k in k_values:
#             candidate = find_best_knn_cluster_for_concept(
#                 semantic_embeddings=semantic_embeddings,
#                 metadata=metadata,
#                 concept_vec=concept_vec,
#                 k=k,
#                 partition=partition,
#             )
#             candidate["concept"] = concept

#             if (best_overall is None) or (candidate["best_score"] > best_overall["best_score"]):
#                 best_overall = candidate

#         results[concept] = best_overall

#     return results


# def print_cluster_summary(results, top_members_to_show=10):
#     print("\n" + "=" * 80)
#     print("Paper-like Concept Cluster Summary")
#     print("=" * 80)

#     for concept, info in results.items():
#         print(f"\n[CONCEPT] {concept}")
#         print(f"partition   : {info['partition']}")
#         print(f"k           : {info['k']}")
#         print(f"best_score  : {info['best_score']:.4f}")
#         print(f"cluster size: {len(info['member_indices'])}")

#         print("top members:")
#         for member in info["members"][:top_members_to_show]:
#             print(
#                 f"  layer={member['layer']:>2}, neuron={member['neuron']:>5}, "
#                 f"tokens={member['top_tokens']}"
#             )


# def rank_individual_neurons_by_concept(
#     semantic_embeddings,
#     metadata,
#     concept_vec,
#     top_n=20
# ):
#     """
#     計算每個神經元的語意嵌入與概念嵌入的餘弦相似度，並依相似度由高至低排序。
#     這能直接反映哪些神經元 (Value Vectors) 對特定概念最敏感。
#     """
#     # 由於 semantic_embeddings 與 concept_vec 先前都已經過 L2 正規化 (p=2)
#     # 因此直接計算內積 (Dot Product) 即等於餘弦相似度 (Cosine Similarity)
#     similarities = semantic_embeddings @ concept_vec

#     # 取得相似度由大到小的索引
#     sorted_indices = np.argsort(similarities)[::-1]

#     ranked_results = []
#     for rank, idx in enumerate(sorted_indices[:top_n]):
#         member = metadata[idx]
#         score = similarities[idx]
#         ranked_results.append({
#             "rank": rank + 1,
#             "score": float(score),
#             "layer": member["layer"],
#             "neuron": member["neuron"],
#             "top_tokens": member["top_tokens"]
#         })

#     return ranked_results

# def run_neuron_ranking(
#     policy_path,
#     concepts=("red", "green", "stack", "sort"), # 可替換為你測試夾取任務的關鍵詞彙
#     top_k_tokens=5,
#     top_n_neurons=20,
#     device=None,
# ):
#     bundle = load_smolvla_and_extract_semantic_embeddings(
#         policy_path=policy_path,
#         top_k_tokens=top_k_tokens,
#         device=device,
#     )

#     tokenizer = bundle["tokenizer"]
#     W_out = bundle["W_out"]
#     semantic_embeddings = bundle["semantic_embeddings"]
#     metadata = bundle["metadata"]

#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     results = {}

#     for concept in concepts:
#         # 1. 將概念詞彙轉換為概念嵌入向量
#         concept_vec = build_concept_embedding(
#             concept_text=concept,
#             tokenizer=tokenizer,
#             W_out=W_out,
#             device=device,
#         )

#         # 2. 對所有神經元進行相似度計算與排序
#         ranked_neurons = rank_individual_neurons_by_concept(
#             semantic_embeddings=semantic_embeddings,
#             metadata=metadata,
#             concept_vec=concept_vec,
#             top_n=top_n_neurons
#         )
        
#         results[concept] = ranked_neurons

#     return results

# def print_neuron_ranking_summary(results):
#     print("\n" + "=" * 80)
#     print("Top Influential Neurons per Concept Ranking")
#     print("=" * 80)

#     for concept, ranked_neurons in results.items():
#         print(f"\n[CONCEPT] {concept}")
#         print(f"{'Rank':<5} | {'Score':<7} | {'Layer':<5} | {'Neuron':<6} | {'Top Tokens'}")
#         print("-" * 80)
#         for neuron in ranked_neurons:
#             # 格式化輸出以便於觀察
#             tokens_str = ", ".join(neuron["top_tokens"])
#             print(f"{neuron['rank']:<5} | {neuron['score']:.4f}  | L{neuron['layer']:<4} | {neuron['neuron']:<6} | [{tokens_str}]")


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
    """
    根據神經元的 top_tokens 中包含目標關鍵字的頻率進行排序。
    新增了正向(pos)與負向(neg)關鍵字過濾，以避免 substring 誤判 (例如 "low" 觸發 "following")。
    """
    # 支援新版的 dict 格式 (區分 pos 和 neg)，也相容舊版的 list 格式
    if isinstance(keywords_dict, list):
        pos_keywords = [kw.lower() for kw in keywords_dict]
        neg_keywords = []
    else:
        pos_keywords = [kw.lower() for kw in keywords_dict.get("pos", [])]
        neg_keywords = [kw.lower() for kw in keywords_dict.get("neg", [])]

    scored_neurons = []
    for neuron in metadata:
        match_count = 0
        # 遍歷該神經元所有的 top tokens
        for token in neuron["top_tokens"]:
            token_lower = token.lower()
            
            # 1. 檢查是否觸發負向關鍵字 (排除條件)
            # 如果 token 包含 follow, allow, slow 等字，直接略過，不計入 low 的分數
            if any(neg_kw in token_lower for neg_kw in neg_keywords):
                continue 

            # 2. 檢查是否包含正向關鍵字
            if any(pos_kw in token_lower for pos_kw in pos_keywords):
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

if __name__ == "__main__":
    #policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2"
    policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_color_unfrozen"
    
    bundle = load_smolvla_and_extract_semantic_embeddings(
        policy_path=policy_path,
        top_k_tokens=10, 
    )
    
    # 升級版的字典結構：加入 neg (負向排除字)
    # 將會導致字串誤判的字詞放入 neg 列表中
    # intervention_keywords_map = {
    #     "Low Transport": {
    #         "pos": ["low"],
    #         "neg": ["follow", "allow", "slow", "blow", "glow", "yellow", "hollow"] # 排除包含 low 的無關字
    #     },
    #     "High Transport": {
    #         "pos": ["high"],
    #         "neg": ["thigh"] # 排除大腿 (雖然不一定會出現，但作為防呆示範)
    #     },
    #     "Slow Transport": {
    #         "pos": ["slow", "safe"],
    #         "neg": []
    #     },
    #     "Fast Transport": {
    #         "pos": ["fast", "risk"],
    #         "neg": ["breakfast"] # 排除早餐
    #     },
        
    # }
    intervention_keywords_map = {
        "Red Color": {
            "pos": ["red", "crimson", "scarlet"],
            "neg": ["predict", "ordered", "hundred", "tired", "shared", "ingredients", "reduction"]
        },
        "Green Color": {
            "pos": ["green", "emerald", "lime"],
            "neg": ["agreement", "screen", "greensboro"]
        },
        "Blue Color": {
            "pos": ["blue", "azure", "navy"],
            "neg": ["value", "blueprint", "blues"]
        },
        "Yellow Color": {
            "pos": ["yellow", "gold", "lemon"],
            "neg": ["mellow", "bellow"]
        }
    }
    
    # 執行關鍵字頻率排名
    keyword_ranking_results = run_keyword_based_ranking(
        metadata=bundle["metadata"],
        concept_keywords_map=intervention_keywords_map,
        top_n=6
    )

    # 印出結果
    print_keyword_ranking_summary(keyword_ranking_results)
    print_top_neurons_overall_by_logit(bundle["metadata"], top_n=15)




