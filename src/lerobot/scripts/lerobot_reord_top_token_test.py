import torch
import numpy as np

# Cluster & visualization
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Output selective-neurons with their top tokens
import csv
import re

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


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

def knn_expand_concept_clusters(
    semantic_embeddings,
    metadata,
    keyword_ranking_results,
    n_neighbors=20,
    max_results_per_concept=20,
):
    """
    VLA-style semantic cluster expansion.

    Step:
      1) Use keyword-ranked neurons as seed neurons.
      2) Use semantic_embeddings to find nearest neighboring neurons.
      3) Return a cluster-like neighbor set for each intervention concept.

    Note:
      This is not unsupervised clustering.
      It is seed-based semantic retrieval / cluster expansion.
    """
    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="cosine"
    )
    nn.fit(semantic_embeddings)

    concept_clusters = {}

    # Build lookup from (layer, neuron) to global index
    neuron_to_global_id = {
        (m["layer"], m["neuron"]): m["global_id"]
        for m in metadata
    }

    for concept, seed_neurons in keyword_ranking_results.items():
        neighbor_scores = {}

        for seed in seed_neurons:
            key = (seed["layer"], seed["neuron"])
            if key not in neuron_to_global_id:
                continue

            seed_id = neuron_to_global_id[key]
            query_vec = semantic_embeddings[seed_id : seed_id + 1]

            distances, indices = nn.kneighbors(query_vec)

            for dist, idx in zip(distances[0], indices[0]):
                similarity = 1.0 - float(dist)

                # Keep best similarity if the same neuron is reached by multiple seeds
                if idx not in neighbor_scores or similarity > neighbor_scores[idx]["similarity"]:
                    neighbor_scores[idx] = {
                        "similarity": similarity,
                        "seed_layer": seed["layer"],
                        "seed_neuron": seed["neuron"],
                    }

        ranked_neighbors = sorted(
            neighbor_scores.items(),
            key=lambda x: x[1]["similarity"],
            reverse=True
        )

        cluster = []
        for idx, info in ranked_neighbors[:max_results_per_concept]:
            m = metadata[idx]
            cluster.append({
                "global_id": m["global_id"],
                "layer": m["layer"],
                "neuron": m["neuron"],
                "similarity": info["similarity"],
                "seed_layer": info["seed_layer"],
                "seed_neuron": info["seed_neuron"],
                "top_tokens": m["top_tokens"],
                "max_logit": m["top_logits"][0],
            })

        concept_clusters[concept] = cluster

    return concept_clusters

def print_knn_concept_clusters(concept_clusters):
    print("\n" + "=" * 110)
    print("KNN-Expanded Semantic Concept Clusters")
    print("=" * 110)

    for concept, cluster in concept_clusters.items():
        print(f"\n[CONCEPT CLUSTER] {concept}")
        print(
            f"{'Rank':<5} | {'Sim':<8} | {'Layer':<5} | {'Neuron':<7} | "
            f"{'Seed':<12} | {'Top Tokens'}"
        )
        print("-" * 110)

        for rank, item in enumerate(cluster, start=1):
            tokens_str = ", ".join(item["top_tokens"])
            seed_str = f"L{item['seed_layer']}:N{item['seed_neuron']}"
            print(
                f"{rank:<5} | {item['similarity']:.4f}   | "
                f"L{item['layer']:<4} | {item['neuron']:<7} | "
                f"{seed_str:<12} | [{tokens_str}]"
            )

def cluster_neurons_with_kmeans(
    semantic_embeddings,
    metadata,
    n_clusters=50,
    batch_size=4096,
    random_state=42,
):
    """
    Unsupervised clustering over all neuron semantic embeddings.

    This clusters neurons based on their semantic embedding, not based on keyword matching.
    MiniBatchKMeans is used because the number of FFN neurons can be large.
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto",
    )

    labels = kmeans.fit_predict(semantic_embeddings)

    clustered = []
    for idx, m in enumerate(metadata):
        clustered.append({
            "cluster": int(labels[idx]),
            "global_id": m["global_id"],
            "layer": m["layer"],
            "neuron": m["neuron"],
            "top_tokens": m["top_tokens"],
            "max_logit": m["top_logits"][0],
        })

    return clustered, kmeans

def print_kmeans_cluster_summary(
    clustered,
    top_clusters=20,
    samples_per_cluster=8,
):
    """
    Print a compact summary of each cluster:
      - cluster size
      - most frequent top tokens
      - sample neurons with high max logit
    """
    clusters = defaultdict(list)

    for item in clustered:
        clusters[item["cluster"]].append(item)

    # Sort clusters by size, descending
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print("\n" + "=" * 120)
    print("KMeans Semantic Cluster Summary")
    print("=" * 120)

    for cluster_id, items in sorted_clusters[:top_clusters]:
        token_counter = Counter()

        for item in items:
            for tok in item["top_tokens"]:
                tok = tok.strip()
                if tok:
                    token_counter[tok] += 1

        common_tokens = [tok for tok, _ in token_counter.most_common(12)]

        # Show strongest neurons in this cluster
        items_sorted = sorted(
            items,
            key=lambda x: x["max_logit"],
            reverse=True
        )

        print(f"\n[CLUSTER {cluster_id}] size={len(items)}")
        print(f"Common tokens: {common_tokens}")
        print(f"{'Rank':<5} | {'Layer':<5} | {'Neuron':<7} | {'Max Logit':<10} | {'Top Tokens'}")
        print("-" * 120)

        for rank, item in enumerate(items_sorted[:samples_per_cluster], start=1):
            tokens_str = ", ".join(item["top_tokens"])
            print(
                f"{rank:<5} | L{item['layer']:<4} | {item['neuron']:<7} | "
                f"{item['max_logit']:.4f}     | [{tokens_str}]"
            )

def run_tsne_for_cluster_visualization(
    semantic_embeddings,
    metadata,
    clustered_neurons=None,
    max_points=5000,
    perplexity=30,
    random_state=42,
    save_path="tsne_neuron_clusters.png",
):
    """
    t-SNE visualization for neuron semantic embeddings.

    Important:
      t-SNE is for visualization, not reliable clustering.
      Cluster labels should come from KMeans/DBSCAN/etc., not from t-SNE itself.
    """
    import matplotlib.pyplot as plt

    num_points = semantic_embeddings.shape[0]

    if num_points > max_points:
        # Prefer visually meaningful neurons: high max token logit
        strengths = np.array(
            [m["top_logits"][0] for m in metadata],
            dtype=np.float32
        )
        selected_indices = np.argsort(strengths)[-max_points:]
    else:
        selected_indices = np.arange(num_points)

    X = semantic_embeddings[selected_indices]

    print(f"[*] Running t-SNE on {len(selected_indices)} neurons...")

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    coords = tsne.fit_transform(X)

    if clustered_neurons is not None:
        labels = np.array(
            [clustered_neurons[idx]["cluster"] for idx in selected_indices]
        )
    else:
        labels = np.zeros(len(selected_indices), dtype=int)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab20",
        s=6,
        alpha=0.85,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("KMeans Cluster ID")
    plt.title("t-SNE Visualization of FFN Neuron Semantic Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[*] Saved t-SNE plot to: {save_path}")

    return coords, selected_indices

def run_tsne_for_concept_cluster_visualization(
    semantic_embeddings,
    metadata,
    concept_clusters,
    max_background_points=4000,
    perplexity=30,
    random_state=42,
    save_path="tsne_concept_clusters.png",
):
    """
    t-SNE visualization where colors correspond to intervention concepts.

    Unlike the previous t-SNE function, this does NOT color by KMeans cluster ID.
    It colors neurons from knn_concept_clusters by concept name.
    """

    concept_color_map = {
        "Low Transport": "tab:blue",
        "High Transport": "tab:brown",
        "Slow Transport": "tab:orange",
        "Fast Transport": "tab:purple",
        "Green": "tab:green",
        "Red": "tab:red",
    }

    num_points = semantic_embeddings.shape[0]

    # Collect all concept-related neuron global IDs
    concept_to_ids = {}
    all_concept_ids = set()

    for concept, cluster in concept_clusters.items():
        ids = [item["global_id"] for item in cluster]
        concept_to_ids[concept] = set(ids)
        all_concept_ids.update(ids)

    # Select background points by high max-token logit
    strengths = np.array(
        [m["top_logits"][0] for m in metadata],
        dtype=np.float32,
    )

    background_candidates = np.argsort(strengths)[-max_background_points:]
    background_ids = set(background_candidates.tolist())

    # Force concept neurons to appear in the plot
    selected_indices = sorted(background_ids.union(all_concept_ids))
    X = semantic_embeddings[selected_indices]

    print(f"[*] Running concept-colored t-SNE on {len(selected_indices)} neurons...")

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    coords = tsne.fit_transform(X)

    global_id_to_row = {
        global_id: row_idx
        for row_idx, global_id in enumerate(selected_indices)
    }

    plt.figure(figsize=(11, 8))

    # Plot background neurons first
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c="lightgray",
        s=6,
        alpha=0.25,
        label="Other neurons",
    )

    # Plot concept clusters on top
    for concept, ids in concept_to_ids.items():
        rows = [
            global_id_to_row[gid]
            for gid in ids
            if gid in global_id_to_row
        ]

        if len(rows) == 0:
            continue

        color = concept_color_map.get(concept, "tab:gray")

        plt.scatter(
            coords[rows, 0],
            coords[rows, 1],
            c=color,
            s=35,
            alpha=0.95,
            label=concept,
            edgecolors="black",
            linewidths=0.3,
        )

    plt.title("t-SNE of FFN Semantic Embeddings Colored by Intervention Concept")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.5, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[*] Saved concept-colored t-SNE plot to: {save_path}")

    return coords, selected_indices

def analyze_concept_overlap_with_kmeans(concept_clusters, clustered_neurons):
    """
    Check which KMeans clusters contain the KNN-expanded concept neurons.
    This tells you whether a concept aligns with one or more unsupervised KMeans clusters.
    """
    global_id_to_cluster = {
        item["global_id"]: item["cluster"]
        for item in clustered_neurons
    }

    print("\n" + "=" * 100)
    print("Concept-to-KMeans Cluster Overlap")
    print("=" * 100)

    for concept, neurons in concept_clusters.items():
        cluster_counter = Counter()

        for neuron in neurons:
            gid = neuron["global_id"]
            if gid in global_id_to_cluster:
                cluster_counter[global_id_to_cluster[gid]] += 1

        total = sum(cluster_counter.values())

        print(f"\n[CONCEPT] {concept}")
        print(f"Total concept neurons: {total}")
        print(f"{'KMeans Cluster':<15} | {'Count':<7} | {'Percent'}")
        print("-" * 45)

        for cluster_id, count in cluster_counter.most_common():
            percent = 100.0 * count / max(total, 1)
            print(f"{cluster_id:<15} | {count:<7} | {percent:.1f}%")


def normalize_token_for_audit(token):
    """
    Normalize decoded tokenizer tokens for matching/auditing.
    Handles common tokenizer artifacts.
    """
    if token is None:
        return ""

    token = str(token)
    token = token.replace("Ġ", "")
    token = token.replace("▁", "")
    token = token.replace("</w>", "")
    token = token.replace("\n", " ")
    token = token.strip().lower()

    return token


def is_gibberish_or_token_artifact(token):
    """
    Heuristic classifier for gibberish / token artifacts.

    This is intentionally conservative:
      - empty tokens
      - pure punctuation
      - very short non-alphanumeric fragments
      - tokens with replacement characters
      - tokens with mostly symbols
    """
    raw = "" if token is None else str(token)
    norm = normalize_token_for_audit(raw)

    if norm == "":
        return True

    if "�" in raw:
        return True

    # Pure punctuation / symbols.
    if re.fullmatch(r"[^a-zA-Z0-9]+", norm):
        return True

    # Single-character non-alphanumeric or odd fragments.
    if len(norm) <= 1 and not norm.isalnum():
        return True

    # Mostly non-alphanumeric.
    alnum_count = sum(ch.isalnum() for ch in norm)
    if len(norm) > 0 and alnum_count / len(norm) < 0.5:
        return True

    return False


def classify_token_for_concept(token, concept_keywords):
    """
    Classify one top token relative to a target concept.

    Categories:
      exact_pos      : exactly equals one of the positive keywords
      contains_pos   : contains a positive keyword but is not exact
      neg            : matches or contains a negative keyword
      gibberish      : likely tokenizer artifact / unreadable token
      unrelated      : readable token but not matched to concept
    """
    norm = normalize_token_for_audit(token)

    pos_keywords = [
        normalize_token_for_audit(kw)
        for kw in concept_keywords.get("pos", [])
    ]
    neg_keywords = [
        normalize_token_for_audit(kw)
        for kw in concept_keywords.get("neg", [])
    ]

    if is_gibberish_or_token_artifact(token):
        return "gibberish"

    if any(norm == neg_kw for neg_kw in neg_keywords):
        return "neg"

    if any(neg_kw and neg_kw in norm for neg_kw in neg_keywords):
        return "neg"

    if any(norm == pos_kw for pos_kw in pos_keywords):
        return "exact_pos"

    if any(pos_kw and pos_kw in norm for pos_kw in pos_keywords):
        return "contains_pos"

    return "unrelated"


def audit_selected_concept_neurons_to_csv(
    concept_clusters,
    concept_keywords_map,
    output_neuron_csv="selected_concept_neurons_audit.csv",
    output_token_csv="selected_concept_tokens_audit.csv",
):
    """
    Export selected concept neurons to CSV for manual inspection.

    Output 1:
      selected_concept_neurons_audit.csv
      One row per selected neuron.

    Output 2:
      selected_concept_tokens_audit.csv
      One row per top token per selected neuron.

    This helps inspect:
      - how many top tokens exactly match the concept keywords
      - how many are partial/synonym matches
      - how many are negative/conflicting words
      - how many are gibberish/token artifacts
      - how many are readable but unrelated
    """
    neuron_rows = []
    token_rows = []

    duplicate_tracker = defaultdict(list)

    for concept, cluster in concept_clusters.items():
        concept_keywords = concept_keywords_map[concept]

        for rank, item in enumerate(cluster, start=1):
            global_id = item.get("global_id")
            layer = item.get("layer")
            neuron = item.get("neuron")
            top_tokens = item.get("top_tokens", [])
            top_logits = item.get("top_logits", None)

            duplicate_tracker[(layer, neuron)].append(concept)

            counts = Counter()
            classified_tokens = defaultdict(list)

            for token_rank, token in enumerate(top_tokens, start=1):
                token_class = classify_token_for_concept(
                    token=token,
                    concept_keywords=concept_keywords,
                )

                counts[token_class] += 1
                classified_tokens[token_class].append(token)

                token_rows.append({
                    "concept": concept,
                    "rank_in_concept_cluster": rank,
                    "global_id": global_id,
                    "layer": layer,
                    "neuron": neuron,
                    "token_rank": token_rank,
                    "token_raw": token,
                    "token_normalized": normalize_token_for_audit(token),
                    "token_class": token_class,
                    "similarity_to_seed": item.get("similarity", ""),
                    "seed_layer": item.get("seed_layer", ""),
                    "seed_neuron": item.get("seed_neuron", ""),
                    "max_logit": item.get("max_logit", ""),
                })

            total_tokens = max(len(top_tokens), 1)
            concept_match_count = counts["exact_pos"] + counts["contains_pos"]
            noisy_count = counts["gibberish"] + counts["unrelated"] + counts["neg"]

            neuron_rows.append({
                "concept": concept,
                "rank_in_concept_cluster": rank,
                "global_id": global_id,
                "layer": layer,
                "neuron": neuron,
                "similarity_to_seed": item.get("similarity", ""),
                "seed_layer": item.get("seed_layer", ""),
                "seed_neuron": item.get("seed_neuron", ""),
                "max_logit": item.get("max_logit", ""),
                "top_tokens_joined": " | ".join(map(str, top_tokens)),
                "exact_pos_match_count": counts["exact_pos"],
                "contains_pos_match_count": counts["contains_pos"],
                "concept_match_count": concept_match_count,
                "neg_conflict_count": counts["neg"],
                "gibberish_count": counts["gibberish"],
                "unrelated_count": counts["unrelated"],
                "noisy_or_unrelated_count": noisy_count,
                "concept_match_ratio": concept_match_count / total_tokens,
                "noisy_or_unrelated_ratio": noisy_count / total_tokens,
                "exact_pos_tokens": " | ".join(classified_tokens["exact_pos"]),
                "contains_pos_tokens": " | ".join(classified_tokens["contains_pos"]),
                "neg_conflict_tokens": " | ".join(classified_tokens["neg"]),
                "gibberish_tokens": " | ".join(classified_tokens["gibberish"]),
                "unrelated_tokens": " | ".join(classified_tokens["unrelated"]),
            })

    # Mark neurons that appear in more than one concept.
    duplicate_map = {
        key: concepts
        for key, concepts in duplicate_tracker.items()
        if len(set(concepts)) > 1
    }

    for row in neuron_rows:
        key = (row["layer"], row["neuron"])
        overlapping_concepts = sorted(set(duplicate_map.get(key, [])))

        row["appears_in_multiple_concepts"] = len(overlapping_concepts) > 1
        row["overlapping_concepts"] = " | ".join(overlapping_concepts)

    # Write neuron-level CSV.
    if len(neuron_rows) > 0:
        neuron_fieldnames = list(neuron_rows[0].keys())
        with open(output_neuron_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=neuron_fieldnames)
            writer.writeheader()
            writer.writerows(neuron_rows)

    # Write token-level CSV.
    if len(token_rows) > 0:
        token_fieldnames = list(token_rows[0].keys())
        with open(output_token_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=token_fieldnames)
            writer.writeheader()
            writer.writerows(token_rows)

    print("\n" + "=" * 100)
    print("Selected Concept Neuron Audit CSV Export")
    print("=" * 100)
    print(f"Neuron-level CSV saved to: {output_neuron_csv}")
    print(f"Token-level CSV saved to : {output_token_csv}")
    print(f"Total selected neuron rows: {len(neuron_rows)}")
    print(f"Total selected token rows : {len(token_rows)}")

    return neuron_rows, token_rows

def export_concept_tsne_points_to_csv(
    semantic_embeddings,
    metadata,
    concept_clusters,
    max_background_points=4000,
    perplexity=30,
    random_state=42,
    output_csv="selected_concept_tsne_points.csv",
):
    """
    Export concept-colored t-SNE coordinates to CSV.

    This lets the Streamlit dashboard show an interactive t-SNE scatter plot.
    """
    import csv

    num_points = semantic_embeddings.shape[0]

    concept_to_ids = {}
    all_concept_ids = set()

    for concept, cluster in concept_clusters.items():
        ids = [item["global_id"] for item in cluster]
        concept_to_ids[concept] = set(ids)
        all_concept_ids.update(ids)

    # Background neurons: high max-token logit neurons.
    strengths = np.array(
        [m["top_logits"][0] for m in metadata],
        dtype=np.float32,
    )

    background_candidates = np.argsort(strengths)[-max_background_points:]
    background_ids = set(background_candidates.tolist())

    selected_indices = sorted(background_ids.union(all_concept_ids))
    X = semantic_embeddings[selected_indices]

    print(f"[*] Running t-SNE for CSV export on {len(selected_indices)} neurons...")

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    coords = tsne.fit_transform(X)

    # Reverse lookup: global_id -> concept.
    global_id_to_concepts = defaultdict(list)
    for concept, ids in concept_to_ids.items():
        for gid in ids:
            global_id_to_concepts[gid].append(concept)

    rows = []
    for row_idx, global_id in enumerate(selected_indices):
        m = metadata[global_id]
        concepts = global_id_to_concepts.get(global_id, [])

        if len(concepts) == 0:
            concept_label = "Other neurons"
            is_selected_concept = False
        else:
            concept_label = " | ".join(sorted(concepts))
            is_selected_concept = True

        rows.append({
            "global_id": global_id,
            "layer": m["layer"],
            "neuron": m["neuron"],
            "concept": concept_label,
            "is_selected_concept": is_selected_concept,
            "tsne_x": float(coords[row_idx, 0]),
            "tsne_y": float(coords[row_idx, 1]),
            "max_logit": float(m["top_logits"][0]),
            "top_tokens_joined": " | ".join(map(str, m["top_tokens"])),
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[*] Saved selected concept t-SNE points to: {output_csv}")

    return rows

def export_kmeans_tsne_points_to_csv(
    semantic_embeddings,
    metadata,
    clustered_neurons,
    max_points=5000,
    perplexity=30,
    random_state=42,
    output_csv="neuron_kmeans_tsne_points.csv",
):
    """
    Export global KMeans/t-SNE neuron map to CSV.

    This is the interactive version of tsne_neuron_clusters.png.

    Each row includes:
      - t-SNE coordinates
      - KMeans cluster ID
      - layer / neuron / global_id
      - top tokens
      - max token logit
    """
    import csv

    num_points = semantic_embeddings.shape[0]

    if num_points > max_points:
        strengths = np.array(
            [m["top_logits"][0] for m in metadata],
            dtype=np.float32,
        )
        selected_indices = np.argsort(strengths)[-max_points:]
    else:
        selected_indices = np.arange(num_points)

    X = semantic_embeddings[selected_indices]

    print(f"[*] Running KMeans-neuron t-SNE CSV export on {len(selected_indices)} neurons...")

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    coords = tsne.fit_transform(X)

    global_id_to_cluster = {
        item["global_id"]: item["cluster"]
        for item in clustered_neurons
    }

    rows = []

    for row_idx, global_id in enumerate(selected_indices):
        m = metadata[int(global_id)]

        rows.append({
            "global_id": int(global_id),
            "layer": int(m["layer"]),
            "neuron": int(m["neuron"]),
            "kmeans_cluster": int(global_id_to_cluster.get(int(global_id), -1)),
            "tsne_x": float(coords[row_idx, 0]),
            "tsne_y": float(coords[row_idx, 1]),
            "max_logit": float(m["top_logits"][0]),
            "top_tokens_joined": " | ".join(map(str, m["top_tokens"])),
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[*] Saved KMeans neuron t-SNE points to: {output_csv}")

    return rows
    
if __name__ == "__main__":
    policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_height"
    
    bundle = load_smolvla_and_extract_semantic_embeddings(
        policy_path=policy_path,
        top_k_tokens=10, 
    )
    
    # 升級版的字典結構：加入 neg (負向排除字)
    # 將會導致字串誤判的字詞放入 neg 列表中
    intervention_keywords_map = {
        "Low Transport": {
            "pos": ["low"],
            "neg": ["follow", "allow", "slow", "blow", "glow", "yellow", "hollow"] # 排除包含 low 的無關字
        },
        "High Transport": {
            "pos": ["high"],
            "neg": ["thigh"] # 排除大腿 (雖然不一定會出現，但作為防呆示範)
        },
        "Slow Transport": {
            "pos": ["slow", "safe"],
            "neg": []
        },
        "Fast Transport": {
            "pos": ["fast", "risk"],
            "neg": ["breakfast"] # 排除早餐
        },
        "Green" : {
            "pos": ["green","verdant","leafy"],
            "neg": ["red","yellow","blue","pink"]
        },
        "Red" : {
            "pos": ["red","maroon","vermilion", "crimson", "scarlet"],
            "neg": ["green","yellow","blue","pink"]
        }
    }
    
    # 執行關鍵字頻率排名
    keyword_ranking_results = run_keyword_based_ranking(
        metadata=bundle["metadata"],
        concept_keywords_map=intervention_keywords_map,
        top_n=6
    )

    # KNN semantic cluster expansion from keyword-selected seed neurons
    knn_concept_clusters = knn_expand_concept_clusters(
        semantic_embeddings=bundle["semantic_embeddings"],
        metadata=bundle["metadata"],
        keyword_ranking_results=keyword_ranking_results,
        n_neighbors=30,
        max_results_per_concept=20, #20 for each concept
    )

    neuron_audit_rows, token_audit_rows = audit_selected_concept_neurons_to_csv(
        concept_clusters=knn_concept_clusters,
        concept_keywords_map=intervention_keywords_map,
        output_neuron_csv="selected_concept_neurons_audit.csv",
        output_token_csv="selected_concept_tokens_audit.csv",
    )

    run_tsne_for_concept_cluster_visualization(
        semantic_embeddings=bundle["semantic_embeddings"],
        metadata=bundle["metadata"],
        concept_clusters=knn_concept_clusters,
        max_background_points=4000,
        perplexity=30,
        save_path="tsne_concept_clusters.png",
    )

    clustered_neurons, kmeans_model = cluster_neurons_with_kmeans(
        semantic_embeddings=bundle["semantic_embeddings"],
        metadata=bundle["metadata"],
        n_clusters=50,
    )

    analyze_concept_overlap_with_kmeans(
        concept_clusters=knn_concept_clusters,
        clustered_neurons=clustered_neurons,
    )

    print_kmeans_cluster_summary(
        clustered=clustered_neurons,
        top_clusters=20,
        samples_per_cluster=8,
    )

    run_tsne_for_cluster_visualization(
        semantic_embeddings=bundle["semantic_embeddings"],
        metadata=bundle["metadata"],
        clustered_neurons=clustered_neurons,
        max_points=5000,
        perplexity=30,
        save_path="tsne_neuron_clusters.png",
    )

    export_concept_tsne_points_to_csv(
        semantic_embeddings=bundle["semantic_embeddings"],
        metadata=bundle["metadata"],
        concept_clusters=knn_concept_clusters,
        max_background_points=4000,
        perplexity=30,
        output_csv="selected_concept_tsne_points.csv",
    )

    print_top_neurons_overall_by_logit(bundle["metadata"], top_n=15)


