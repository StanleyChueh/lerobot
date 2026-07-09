import argparse
import csv
import json
import re

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@torch.no_grad()
def load_model_bundle(policy_family, policy_path, device=None):
    """
    Load a policy and return (text_model, lm_head_weight, tokenizer) in a
    common shape, regardless of which VLA/VLM backbone it wraps.

    text_model must expose `.layers[i].mlp.down_proj.weight` (Llama/Gemma-style
    SwiGLU/GEGLU MLP), matching the paper's FFN(x) = f_theta(x)^T W_theta
    formalism (arXiv:2509.00328, Eq. 1-2).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if policy_family == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(policy_path)
        policy.eval()
        policy.to(device)

        vlm_with_expert = policy.model.vlm_with_expert
        text_model = vlm_with_expert.get_vlm_model().text_model
        tokenizer = vlm_with_expert.processor.tokenizer
        lm_head_weight = vlm_with_expert.vlm.lm_head.weight

    elif policy_family == "pi0":
        from transformers import AutoTokenizer

        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        policy = PI0Policy.from_pretrained(policy_path)
        policy.eval()
        policy.to(device)

        paligemma = policy.model.paligemma_with_expert.paligemma
        text_model = paligemma.language_model
        lm_head_weight = paligemma.lm_head.weight
        # PaliGemma is gated on the Hub; requires `huggingface-cli login` +
        # accepting the license once for google/paligemma-3b-pt-224.
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    elif policy_family == "openvla":
        # OpenVLA has no lerobot wrapper; load the HF checkpoint directly.
        # Assumes the Prismatic/OpenVLA wrapper exposes its Llama2 backbone at
        # `.language_model` (LlamaForCausalLM) -- verify this path against the
        # actual openvla-7b checkpoint before trusting results; adjust if a
        # release renames the attribute.
        from transformers import AutoModelForVision2Seq, AutoProcessor

        policy = AutoModelForVision2Seq.from_pretrained(
            policy_path, trust_remote_code=True, torch_dtype=torch.float32
        )
        policy.eval()
        policy.to(device)

        processor = AutoProcessor.from_pretrained(policy_path, trust_remote_code=True)
        text_model = policy.language_model.model
        lm_head_weight = policy.language_model.lm_head.weight
        tokenizer = processor.tokenizer

    else:
        raise ValueError(f"Unknown policy_family: {policy_family}")

    lm_head_weight = lm_head_weight.detach().to(device=device, dtype=torch.float32)
    return policy, text_model, lm_head_weight, tokenizer, device


@torch.no_grad()
def extract_semantic_embeddings(text_model, lm_head_weight, tokenizer, top_k_tokens=5, device="cpu"):
    """
    Paper-faithful semantic embedding extraction (arXiv:2509.00328, Appendix B.3):
      1) take FFN value vectors from down_proj columns
      2) project each value vector into output token space via the LM head
      3) take top-k tokens
      4) build a semantic embedding by softmax-weighted averaging
         the corresponding output token embeddings
    """
    W_out = lm_head_weight  # [vocab_size, hidden_dim]

    semantic_embeddings = []
    metadata = []

    global_id = 0
    num_layers = len(text_model.layers)
    print(f"[*] Extracting value-vector semantic embeddings from {num_layers} layers...")

    for layer_idx in range(num_layers):
        # PyTorch Linear weight shape: [out_features, in_features]
        # down_proj: [hidden_dim, intermediate_dim]; each COLUMN is one FFN value vector
        W_value = text_model.layers[layer_idx].mlp.down_proj.weight.detach().to(
            device=device, dtype=torch.float32
        )  # [d_model, d_ff]

        token_logits = W_out @ W_value  # [vocab_size, d_ff]

        top_logits, top_token_ids = torch.topk(token_logits, k=top_k_tokens, dim=0)
        top_logits_t = top_logits.transpose(0, 1).contiguous()  # [d_ff, k]
        top_token_ids_t = top_token_ids.transpose(0, 1).contiguous()  # [d_ff, k]

        weights = torch.softmax(top_logits_t, dim=1)  # [d_ff, k]
        token_embs = W_out[top_token_ids_t]  # [d_ff, k, d_model]
        e_sem = (weights.unsqueeze(-1) * token_embs).sum(dim=1)  # [d_ff, d_model]
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
        "tokenizer": tokenizer,
        "W_out": W_out,
        "semantic_embeddings": semantic_embeddings,
        "metadata": metadata,
    }


@torch.no_grad()
def build_concept_embedding(concept_text, tokenizer, W_out, device):
    """
    Paper Appendix B.3 step 4: tokenize the concept word/phrase using the model
    tokenizer, embed it using the language modeling head. For multi-token
    phrases, average their output embeddings.
    """
    enc = tokenizer(concept_text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"][0].to(device)

    if input_ids.numel() == 0:
        raise ValueError(f"Tokenizer produced no tokens for concept: {concept_text}")

    concept_vec = W_out[input_ids].mean(dim=0)
    concept_vec = torch.nn.functional.normalize(concept_vec, p=2, dim=0)

    return concept_vec.cpu().numpy()


def get_partition_indices(num_vectors, partition="full"):
    if partition == "full":
        return np.arange(num_vectors)
    elif partition == "early":
        return np.arange(num_vectors // 2)
    elif partition == "late":
        return np.arange(num_vectors // 2, num_vectors)
    else:
        raise ValueError("partition must be one of: full, early, late")


def build_knn_candidate_clusters(semantic_embeddings, k_values=(10, 20, 40), partition="full"):
    """
    Faithful approximation of the paper's Appendix B.3 steps 3-4, fitted once
    per partition and reused across every k and every concept:
      - cosine kNN over semantic embeddings
      - each neighborhood acts as a candidate cluster
      - centroid = average embedding of cluster members

    sklearn's cosine metric has no tree index, so `.kneighbors()` is an
    O(N^2) brute-force pass over all ~tens-of-thousands of value vectors --
    running it once per (concept, k) pair instead of once per partition is
    what makes `--mode all` slow.
    """
    all_idx = get_partition_indices(len(semantic_embeddings), partition=partition)
    X = semantic_embeddings[all_idx]

    max_k = min(max(k_values), len(X))
    if max_k < 1:
        raise ValueError("No vectors available in this partition.")

    nbrs = NearestNeighbors(n_neighbors=max_k, metric="cosine")
    nbrs.fit(X)
    _, neighbors = nbrs.kneighbors(X, n_neighbors=max_k)  # [N, max_k], the one expensive call

    clusters_by_k = {}
    for k in k_values:
        k = min(k, max_k)
        neighbors_k = neighbors[:, :k]
        centroids = X[neighbors_k].mean(axis=1)
        centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        clusters_by_k[k] = {"all_idx": all_idx, "neighbors": neighbors_k, "centroids": centroids}

    return clusters_by_k


def find_best_knn_cluster_for_concept(clusters_by_k, metadata, concept_vec, partition="full"):
    best_overall = None
    for k, cluster in clusters_by_k.items():
        scores = cluster["centroids"] @ concept_vec
        best_local = int(np.argmax(scores))
        best_score = float(scores[best_local])

        if best_overall is None or best_score > best_overall["best_score"]:
            member_global_idx = cluster["all_idx"][cluster["neighbors"][best_local]]
            best_overall = {
                "partition": partition,
                "k": k,
                "best_score": best_score,
                "centroid": cluster["centroids"][best_local],
                "member_indices": member_global_idx.tolist(),
                "members": [metadata[int(i)] for i in member_global_idx.tolist()],
            }

    return best_overall


def run_concept_clustering(
    bundle,
    concepts,
    k_values=(10, 20, 40),
    partition="full",
    device="cpu",
):
    tokenizer = bundle["tokenizer"]
    W_out = bundle["W_out"]
    semantic_embeddings = bundle["semantic_embeddings"]
    metadata = bundle["metadata"]

    clusters_by_k = build_knn_candidate_clusters(semantic_embeddings, k_values=k_values, partition=partition)

    results = {}
    for concept in concepts:
        concept_vec = build_concept_embedding(concept, tokenizer, W_out, device)
        best_overall = find_best_knn_cluster_for_concept(clusters_by_k, metadata, concept_vec, partition=partition)
        best_overall["concept"] = concept
        results[concept] = best_overall

    return results


def print_cluster_summary(results, top_members_to_show=10):
    print("\n" + "=" * 80)
    print("Paper-like Concept Cluster Summary (Appendix B.3)")
    print("=" * 80)

    for concept, info in results.items():
        print(f"\n[CONCEPT] {concept}")
        print(f"partition   : {info['partition']}")
        print(f"k           : {info['k']}")
        print(f"best_score  : {info['best_score']:.4f}")
        print(f"cluster size: {len(info['member_indices'])}")

        print("top members:")
        for member in info["members"][:top_members_to_show]:
            print(
                f"  layer={member['layer']:>2}, neuron={member['neuron']:>5}, "
                f"tokens={member['top_tokens']}"
            )


def compute_cluster_purity(members, concept_keywords):
    """
    Fraction of top-token slots across a cluster's member neurons that
    literally match the concept's keyword family (case-insensitive substring).
    A high-purity cluster (paper's Fig 9 "up" cluster is ~90%+) means the
    kNN step actually isolated the concept; a low-purity cluster means the
    neighborhood is a grab-bag that happened to score well on cosine
    similarity alone.
    """
    total, matched = 0, 0
    for member in members:
        for tok in member["top_tokens"]:
            total += 1
            if any(kw.lower() in tok.lower() for kw in concept_keywords):
                matched += 1
    return matched / total if total else 0.0


def summarize_concept_results(results, keyword_map):
    """
    Builds a paper-ready table: one row per concept with its cluster's
    best_score (cosine sim to concept embedding), purity %, and cluster size.
    `keyword_map` is {concept: [keyword, ...]} used only for the purity check
    (independent of the kNN clustering itself).
    """
    rows = []
    for concept, info in results.items():
        keywords = keyword_map.get(concept, [concept])
        purity = compute_cluster_purity(info["members"], keywords)
        rows.append({
            "concept": concept,
            "best_score": info["best_score"],
            "purity": purity,
            "cluster_size": len(info["member_indices"]),
            "k": info["k"],
        })
    return rows


def print_concept_summary_table(rows):
    print("\n" + "=" * 60)
    print("Concept Cluster Quality Summary")
    print("=" * 60)
    print(f"{'Concept':<10} | {'Cos.Sim':<8} | {'Purity %':<9} | {'Size':<5} | {'k':<4}")
    print("-" * 60)
    for row in rows:
        print(
            f"{row['concept']:<10} | {row['best_score']:.4f}   | "
            f"{row['purity'] * 100:>6.1f}%  | {row['cluster_size']:<5} | {row['k']:<4}"
        )


def concept_centroid_similarity_matrix(results):
    """
    Pairwise cosine similarity between each concept's winning cluster
    centroid. Antonym pairs (e.g. high/low, fast/slow) landing near 1.0
    means the model failed to separate them into distinct directions.
    """
    concepts = list(results.keys())
    centroids = np.stack([results[c]["centroid"] for c in concepts])
    sims = centroids @ centroids.T
    return concepts, sims


def plot_concept_summary(results, keyword_map, output_prefix="concept_analysis"):
    """
    Saves two paper-ready PNGs: a purity bar chart and an antonym centroid
    cosine-similarity heatmap.
    """
    import matplotlib.pyplot as plt

    rows = summarize_concept_results(results, keyword_map)
    concepts = [r["concept"] for r in rows]
    purities = [r["purity"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(concepts, purities, color="#4C72B0")
    ax.set_ylabel("Cluster purity (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Fraction of top-token slots matching the target concept")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_purity.png", dpi=200)
    plt.close(fig)

    concept_names, sims = concept_centroid_similarity_matrix(results)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(sims, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(concept_names)))
    ax.set_yticks(range(len(concept_names)))
    ax.set_xticklabels(concept_names, rotation=45, ha="right")
    ax.set_yticklabels(concept_names)
    for i in range(len(concept_names)):
        for j in range(len(concept_names)):
            ax.text(j, i, f"{sims[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title("Concept cluster centroid similarity")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_centroid_similarity.png", dpi=200)
    plt.close(fig)

    print(f"[*] Saved {output_prefix}_purity.png and {output_prefix}_centroid_similarity.png")


def export_results_json(cluster_results, summary_rows, policy_family, policy_path, output_path):
    """
    Saves this run's per-concept metrics + winning cluster centroid so results
    from separate smolvla/pi0/openvla runs can be combined into cross-policy
    comparison charts later (--mode combine) without re-running the model.
    """
    purity_by_concept = {row["concept"]: row["purity"] for row in summary_rows}
    payload = {
        "policy_family": policy_family,
        "policy_path": policy_path,
        "concepts": {
            concept: {
                "best_score": info["best_score"],
                "purity": purity_by_concept.get(concept),
                "cluster_size": len(info["member_indices"]),
                "k": info["k"],
                "centroid": info["centroid"].tolist(),
            }
            for concept, info in cluster_results.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[*] Saved combinable results to {output_path}")


def load_multi_policy_results(paths):
    results = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        results[data["policy_family"]] = data
    return results


def plot_multi_policy_purity(results, concepts, output_path):
    """
    One grouped bar chart: x-axis = concept, one bar per policy, so all
    policies' cleanliness scores sit side by side instead of one chart per
    policy.
    """
    import matplotlib.pyplot as plt

    policies = list(results.keys())
    x = np.arange(len(concepts))
    width = 0.8 / max(len(policies), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, policy in enumerate(policies):
        values = [(results[policy]["concepts"].get(c) or {}).get("purity", 0.0) * 100 for c in concepts]
        ax.bar(x + i * width, values, width=width, label=policy)

    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(concepts)
    ax.set_ylabel("Cluster purity (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cluster purity by concept and policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[*] Saved {output_path}")


def plot_multi_policy_antonym_similarity(results, antonym_pairs, output_path):
    """
    One grouped bar chart: x-axis = antonym pair, one bar per policy. Each
    similarity is computed within a single policy's own embedding space
    (centroids across policies live in different dimensions and are never
    compared directly), so cross-policy bars remain a valid comparison.
    """
    import matplotlib.pyplot as plt

    policies = list(results.keys())
    pair_labels = [f"{a}/{b}" for a, b in antonym_pairs]
    x = np.arange(len(antonym_pairs))
    width = 0.8 / max(len(policies), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, policy in enumerate(policies):
        concepts = results[policy]["concepts"]
        values = []
        for a, b in antonym_pairs:
            if a in concepts and b in concepts:
                ca = np.array(concepts[a]["centroid"])
                cb = np.array(concepts[b]["centroid"])
                sim = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-8))
            else:
                sim = 0.0
            values.append(sim)
        ax.bar(x + i * width, values, width=width, label=policy)

    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel("Cosine similarity between antonym cluster centroids")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Antonym cluster entanglement by policy (higher = worse separation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[*] Saved {output_path}")


def print_top_neurons_overall_by_logit(metadata, top_n=15):
    """
    不依賴任何輸入概念。
    直接掃描模型中「所有層、所有神經元」，根據神經元對其 Top 1 Token 的 Logit 進行全局排序。
    Logit 越高，代表該神經元對特定詞彙的推動力量越強、語意越鮮明。
    """
    print("\n" + "=" * 90)
    print(f"Top {top_n} Strongest Neurons Overall (By Max Token Logit)")
    print("=" * 90)

    all_neurons = list(metadata)
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
    if isinstance(keywords_dict, list):
        pos_keywords = [kw.lower() for kw in keywords_dict]
        neg_keywords = []
    else:
        pos_keywords = [kw.lower() for kw in keywords_dict.get("pos", [])]
        neg_keywords = [kw.lower() for kw in keywords_dict.get("neg", [])]

    scored_neurons = []
    for neuron in metadata:
        match_count = 0
        for token in neuron["top_tokens"]:
            token_lower = token.lower()

            if any(neg_kw in token_lower for neg_kw in neg_keywords):
                continue

            if any(pos_kw in token_lower for pos_kw in pos_keywords):
                match_count += 1

        scored_neurons.append({
            "layer": neuron["layer"],
            "neuron": neuron["neuron"],
            "match_count": match_count,
            "top_tokens": neuron["top_tokens"],
            "max_logit": neuron["top_logits"][0]
        })

    scored_neurons.sort(key=lambda x: (x["match_count"], x["max_logit"]), reverse=True)

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
            keywords_dict=keywords_dict,
            top_n=top_n
        )
        results[concept_name] = ranked_neurons
    return results


def print_keyword_ranking_summary(results):
    print("\n" + "=" * 95)
    print("Top Neurons by Keyword Frequency (Paper Appendix C.3 hardware steering-selection method)")
    print("=" * 95)

    for concept, ranked_neurons in results.items():
        print(f"\n[INTERVENTION TASK] {concept}")
        print(f"{'Rank':<5} | {'Match Count':<11} | {'Layer':<5} | {'Neuron':<6} | {'Top Tokens'}")
        print("-" * 95)
        for neuron in ranked_neurons:
            tokens_str = ", ".join(neuron["top_tokens"])
            print(f"{neuron['rank']:<5} | {neuron['match_count']:<11} | L{neuron['layer']:<4} | {neuron['neuron']:<6} | [{tokens_str}]")


def export_top_tokens_for_classification(metadata, output_csv_path, top_n=30):
    """
    Dumps top-N tokens per value vector to CSV for manual or LLM-judge
    classification into semantic/non-semantic/none, replicating the paper's
    Appendix B.1 methodology (human classification of >=4/30 top tokens
    sharing a pattern) rather than approximating it with a brittle regex.
    """
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_id", "layer", "neuron", "top_tokens", "label"])
        for neuron in metadata:
            tokens_str = " | ".join(neuron["top_tokens"][:top_n])
            writer.writerow([neuron["global_id"], neuron["layer"], neuron["neuron"], tokens_str, ""])
    print(f"[*] Exported {len(metadata)} neurons to {output_csv_path} for classification.")


def compute_semantic_fraction_by_layer(metadata, labels_by_global_id):
    """
    Given external labels {global_id: "semantic"|"non-semantic"|"none"},
    reproduces the Figure 2a statistic: fraction meaningful (semantic +
    non-semantic) and fraction semantic, per layer.
    """
    by_layer = {}
    for neuron in metadata:
        label = labels_by_global_id.get(neuron["global_id"])
        if label is None:
            continue
        layer = neuron["layer"]
        by_layer.setdefault(layer, {"total": 0, "meaningful": 0, "semantic": 0})
        by_layer[layer]["total"] += 1
        if label in ("semantic", "non-semantic"):
            by_layer[layer]["meaningful"] += 1
        if label == "semantic":
            by_layer[layer]["semantic"] += 1

    stats = {}
    for layer, counts in sorted(by_layer.items()):
        stats[layer] = {
            "fraction_meaningful": counts["meaningful"] / counts["total"],
            "fraction_semantic": counts["semantic"] / counts["total"],
            "n": counts["total"],
        }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and analyze FFN semantic neurons across VLA policies (smolvla, pi0, openvla)."
    )
    parser.add_argument("--policy_family", choices=["smolvla", "pi0", "openvla"], default=None)
    parser.add_argument("--policy_path", default=None, help="HF repo id or local checkpoint path")
    parser.add_argument("--top_k_tokens", type=int, default=5)
    parser.add_argument("--mode", choices=["keyword", "knn", "export", "logit", "all", "combine"], default="all")
    parser.add_argument("--keywords_json", default=None, help="Path to JSON: {concept: {pos:[...], neg:[...]}}")
    parser.add_argument("--concepts", nargs="*", default=["fast", "slow", "up", "down"])
    parser.add_argument("--knn_k_values", nargs="*", type=int, default=[10, 20, 40])
    parser.add_argument("--partition", choices=["full", "early", "late"], default="full")
    parser.add_argument("--export_csv", default=None)
    parser.add_argument("--top_n_logit", type=int, default=15)
    parser.add_argument(
        "--plot_prefix",
        default=None,
        help="If set, save a purity bar chart and centroid-similarity heatmap to <prefix>_purity.png / <prefix>_centroid_similarity.png",
    )
    parser.add_argument(
        "--purity_keywords_json",
        default=None,
        help="Path to JSON: {concept: [keyword, ...]} for the purity metric; defaults to using the concept name itself",
    )
    parser.add_argument(
        "--results_json",
        default=None,
        help="Path to save this run's per-concept metrics+centroids as JSON, for later --mode combine",
    )
    parser.add_argument(
        "--combine_inputs",
        nargs="*",
        default=None,
        help="--mode combine only: paths to --results_json outputs from separate policy runs",
    )
    parser.add_argument(
        "--combine_output_prefix",
        default="multi_policy",
        help="--mode combine only: output prefix for the combined comparison charts",
    )
    parser.add_argument(
        "--antonym_pairs",
        nargs="*",
        default=["fast:slow", "high:low"],
        help="--mode combine only: antonym pairs as 'a:b', used for the entanglement chart",
    )
    args = parser.parse_args()

    if args.mode == "combine":
        if not args.combine_inputs:
            parser.error("--mode combine requires --combine_inputs <results.json> <results.json> ...")

        multi_results = load_multi_policy_results(args.combine_inputs)
        all_concepts = sorted({c for data in multi_results.values() for c in data["concepts"]})

        plot_multi_policy_purity(multi_results, all_concepts, f"{args.combine_output_prefix}_purity.png")
        antonym_pairs = [tuple(pair.split(":")) for pair in args.antonym_pairs]
        plot_multi_policy_antonym_similarity(
            multi_results, antonym_pairs, f"{args.combine_output_prefix}_antonym_similarity.png"
        )
        raise SystemExit(0)

    if not args.policy_family or not args.policy_path:
        parser.error("--policy_family and --policy_path are required unless --mode combine")

    policy, text_model, lm_head_weight, tokenizer, device = load_model_bundle(
        args.policy_family, args.policy_path
    )
    bundle = extract_semantic_embeddings(
        text_model, lm_head_weight, tokenizer, top_k_tokens=args.top_k_tokens, device=device
    )

    if args.mode in ("keyword", "all") and args.keywords_json:
        with open(args.keywords_json) as f:
            concept_keywords_map = json.load(f)
        keyword_results = run_keyword_based_ranking(bundle["metadata"], concept_keywords_map, top_n=6)
        print_keyword_ranking_summary(keyword_results)

    if args.mode in ("knn", "all"):
        cluster_results = run_concept_clustering(
            bundle, args.concepts, k_values=tuple(args.knn_k_values), partition=args.partition, device=device
        )
        print_cluster_summary(cluster_results)

        if args.purity_keywords_json:
            with open(args.purity_keywords_json) as f:
                purity_keyword_map = json.load(f)
        else:
            purity_keyword_map = {c: [c] for c in args.concepts}

        summary_rows = summarize_concept_results(cluster_results, purity_keyword_map)
        print_concept_summary_table(summary_rows)

        if args.plot_prefix:
            plot_concept_summary(cluster_results, purity_keyword_map, output_prefix=args.plot_prefix)

        if args.results_json:
            export_results_json(
                cluster_results, summary_rows, args.policy_family, args.policy_path, args.results_json
            )

    if args.export_csv or args.mode == "export":
        export_top_tokens_for_classification(bundle["metadata"], args.export_csv or "top_tokens_for_labeling.csv")

    if args.mode in ("logit", "all"):
        print_top_neurons_overall_by_logit(bundle["metadata"], top_n=args.top_n_logit)
