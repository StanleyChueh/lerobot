import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@torch.no_grad()
def load_smolvla_and_extract_semantic_embeddings(
    policy_path="ethanCSL/svla_koch_pick_n_place_vla_steering_height",
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
    Paper says:
      tokenize the concept word/phrase using the model tokenizer,
      embed it using the language modeling head.
    For multi-token phrases, we average their output embeddings.
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


def find_best_knn_cluster_for_concept(
    semantic_embeddings,
    metadata,
    concept_vec,
    k=20,
    partition="full",
):
    """
    Faithful approximation of the paper's Appendix B.3:
      - cosine kNN over semantic embeddings
      - each neighborhood acts as a candidate cluster
      - centroid = average embedding of cluster members
      - choose cluster whose centroid is most similar to concept embedding
    """
    all_idx = get_partition_indices(len(semantic_embeddings), partition=partition)
    X = semantic_embeddings[all_idx]

    # Make sure k is valid
    k = min(k, len(X))
    if k < 1:
        raise ValueError("No vectors available in this partition.")

    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
    nbrs.fit(X)

    distances, neighbors = nbrs.kneighbors(X)  # neighbors shape [N, k]

    # Candidate cluster centroids
    centroids = X[neighbors].mean(axis=1)
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    # Concept-centroid cosine similarity
    scores = centroids @ concept_vec
    best_local = int(np.argmax(scores))
    best_score = float(scores[best_local])

    member_local_idx = neighbors[best_local]
    member_global_idx = all_idx[member_local_idx]

    cluster_metadata = [metadata[int(i)] for i in member_global_idx.tolist()]

    return {
        "partition": partition,
        "k": k,
        "best_score": best_score,
        "member_indices": member_global_idx.tolist(),
        "members": cluster_metadata,
    }


def run_paper_like_concept_clustering(
    policy_path,
    concepts=("slow", "fast", "high", "low"),
    k_values=(10, 20, 40),
    partition="full",
    top_k_tokens=5,
    device=None,
):
    bundle = load_smolvla_and_extract_semantic_embeddings(
        policy_path=policy_path,
        top_k_tokens=top_k_tokens,
        device=device,
    )

    tokenizer = bundle["tokenizer"]
    W_out = bundle["W_out"]
    semantic_embeddings = bundle["semantic_embeddings"]
    metadata = bundle["metadata"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    for concept in concepts:
        concept_vec = build_concept_embedding(
            concept_text=concept,
            tokenizer=tokenizer,
            W_out=W_out,
            device=device,
        )

        best_overall = None
        for k in k_values:
            candidate = find_best_knn_cluster_for_concept(
                semantic_embeddings=semantic_embeddings,
                metadata=metadata,
                concept_vec=concept_vec,
                k=k,
                partition=partition,
            )
            candidate["concept"] = concept

            if (best_overall is None) or (candidate["best_score"] > best_overall["best_score"]):
                best_overall = candidate

        results[concept] = best_overall

    return results


def print_cluster_summary(results, top_members_to_show=10):
    print("\n" + "=" * 80)
    print("Paper-like Concept Cluster Summary")
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


if __name__ == "__main__":
    results = run_paper_like_concept_clustering(
        policy_path="ethanCSL/svla_koch_pick_n_place_vla_steering_height",
        concepts=("slow", "fast", "high", "low"),
        k_values=(10, 20, 40),
        partition="full",       # "full", "early", "late"
        top_k_tokens=5,         # matches the paper's Appendix B.3
    )

    print_cluster_summary(results, top_members_to_show=12)