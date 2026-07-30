import argparse
import csv
import json
import re

import numpy as np
import torch


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

        # PaliGemma plus its Gemma action-expert twin together already take
        # ~14.6GB in float32 -- on a 16GB-class GPU that leaves no headroom
        # for this script's own extraction buffers (even chunked), so this
        # branch runs on CPU like the openvla branch below.
        device = "cpu"
        policy = PI0Policy.from_pretrained(policy_path)
        policy.eval()
        policy.to(device)

        paligemma = policy.model.paligemma_with_expert.paligemma
        text_model = paligemma.language_model
        lm_head_weight = paligemma.lm_head.weight
        # PaliGemma is gated on the Hub; requires `huggingface-cli login` +
        # accepting the license once for google/paligemma-3b-pt-224.
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        # The gemma_expert (action-expert twin of the LM, same param count)
        # and vision_tower are never read by extract_semantic_embeddings --
        # freeing them here is the difference between fitting in RAM and
        # OOMing once the extraction buffers are also allocated.
        del policy.model.paligemma_with_expert.gemma_expert
        del paligemma.model.vision_tower
        import gc

        gc.collect()

    elif policy_family == "openvla":
        # OpenVLA has no lerobot wrapper; load the HF checkpoint directly.
        # Assumes the Prismatic/OpenVLA wrapper exposes its Llama2 backbone at
        # `.language_model` (LlamaForCausalLM) -- verify this path against the
        # actual openvla-7b checkpoint before trusting results; adjust if a
        # release renames the attribute.
        # openvla-7b's config.auto_map only registers under the now-removed
        # "AutoModelForVision2Seq" key, so no AutoModelFor* class in current
        # transformers matches it via the normal dispatch -- resolve the
        # custom class directly from the dynamic module instead.
        from transformers import AutoConfig, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(policy_path, trust_remote_code=True)
        auto_map = getattr(config, "auto_map", {}) or {}
        model_class_ref = (
            auto_map.get("AutoModelForVision2Seq")
            or auto_map.get("AutoModelForImageTextToText")
            or auto_map.get("AutoModel")
        )
        if model_class_ref is None:
            raise RuntimeError(f"No usable model class in config.auto_map for {policy_path}: {auto_map}")
        model_cls = get_class_from_dynamic_module(model_class_ref, policy_path)

        # openvla-7b's modeling_prismatic.py predates transformers' current
        # attention-implementation dispatch, which now expects every
        # PreTrainedModel subclass to declare `_supports_sdpa`; the vendored
        # class never sets it, so __init__ crashes probing for SDPA support.
        # Declaring it (and passing attn_implementation="eager" so the
        # SDPA/flash candidates are never probed at all) works around the
        # version skew without touching the vendored file.
        if not hasattr(model_cls, "_supports_sdpa"):
            model_cls._supports_sdpa = False

        # The checkpoint is ~15GB on disk for 7B params (~2 bytes/param), i.e.
        # already stored in bfloat16 -- requesting float32 here would double
        # it to ~28GB, which fits neither typical consumer GPU VRAM nor,
        # often, system RAM. Load at the checkpoint's native precision and
        # keep it on CPU: even at half precision the weights alone leave no
        # headroom for compute buffers on a 16GB-class GPU. Per-layer slices
        # are upcast to float32 for the actual projection math in
        # extract_semantic_embeddings, so this doesn't cost extraction precision.
        # openvla-7b's remote code only runs against transformers==4.40.1,
        # which still uses the pre-rename `torch_dtype=` kwarg.
        device = "cpu"
        policy = model_cls.from_pretrained(
            policy_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        policy.eval()
        policy.to(device)

        processor = AutoProcessor.from_pretrained(policy_path, trust_remote_code=True)
        text_model = policy.language_model.model
        lm_head_weight = policy.language_model.lm_head.weight
        tokenizer = processor.tokenizer

        # We only ever read the Llama2 backbone's weights; the vision tower
        # (~0.73B params) and projector (~0.07B params) are never touched but
        # otherwise stay resident for no reason -- on a memory-constrained
        # host, freeing them is the difference between fitting and OOMing.
        del policy.vision_backbone
        del policy.projector
        import gc

        gc.collect()

    else:
        raise ValueError(f"Unknown policy_family: {policy_family}")

    lm_head_weight = lm_head_weight.detach().to(device=device, dtype=torch.float32)
    return policy, text_model, lm_head_weight, tokenizer, device


def decode_top_tokens(tokenizer, token_ids):
    return [tokenizer.decode([tok_id]).replace("\n", " ").strip() for tok_id in token_ids]


@torch.no_grad()
def extract_semantic_embeddings(text_model, lm_head_weight, tokenizer, top_k_tokens=5, device="cpu"):
    """
    Paper-faithful semantic embedding extraction (arXiv:2509.00328, Appendix B.3):
      1) take FFN value vectors from down_proj columns
      2) project each value vector into output token space via the LM head
      3) take top-k tokens
      4) build a semantic embedding by softmax-weighted averaging
         the corresponding output token embeddings

    Token strings are NOT decoded here -- metadata only stores token ids, and
    callers decode lazily (see decode_top_tokens) for just the handful of
    neurons they actually display. For OpenVLA-scale models (~350K value
    vectors), decoding all of them up front is both slow and memory-heavy
    for no benefit in modes (e.g. --mode knn) that only ever look at a
    winning cluster's ~10-40 members.
    """
    W_out = lm_head_weight  # [vocab_size, hidden_dim]

    num_layers = len(text_model.layers)
    d_ff = text_model.layers[0].mlp.down_proj.weight.shape[1]
    d_model = W_out.shape[1]
    total_neurons = num_layers * d_ff

    # Preallocated once and filled by layer slice. Building this via a
    # Python list of per-neuron numpy rows + a final np.asarray conversion
    # transiently holds both representations at once (~2x the final size)
    # right when the resident model already occupies most of memory --
    # for OpenVLA-scale models that spike is enough to OOM.
    semantic_embeddings = np.empty((total_neurons, d_model), dtype=np.float32)
    metadata = []

    print(f"[*] Extracting value-vector semantic embeddings from {num_layers} layers...")

    global_id = 0
    for layer_idx in range(num_layers):
        # Each large intermediate is `del`ed the moment it's no longer needed
        # instead of left bound to its loop variable until the next
        # reassignment -- for OpenVLA-scale dims (vocab~32K, d_ff~11K) each of
        # token_logits/token_embs is ~1GB, and letting the old and new
        # iteration's tensors briefly coexist was enough extra peak memory to
        # tip an already-tight system into OOM.

        # PyTorch Linear weight shape: [out_features, in_features]
        # down_proj: [hidden_dim, intermediate_dim]; each COLUMN is one FFN value vector
        W_value = text_model.layers[layer_idx].mlp.down_proj.weight.detach().to(
            device=device, dtype=torch.float32
        )  # [d_model, d_ff]
        if W_value.shape[1] != d_ff:
            raise ValueError(f"Layer {layer_idx} has d_ff={W_value.shape[1]}, expected {d_ff}")

        # W_out @ W_value materializes [vocab_size, d_ff]. For large-vocab
        # backbones (e.g. Gemma/PaliGemma's 256K-token vocab used by pi0)
        # that single matmul is tens of GB for one layer alone -- OOMs
        # regardless of device. Chunking along d_ff (the topk reduction is
        # over dim=0/vocab, so it's unaffected by how the d_ff columns are
        # grouped) bounds peak size to vocab_size * chunk_size * 4 bytes,
        # independent of d_ff, at ~1GiB/chunk.
        vocab_size = W_out.shape[0]
        chunk_size = max(1, min(d_ff, (1 << 28) // max(vocab_size, 1)))

        top_logits_chunks = []
        top_token_ids_chunks = []
        for start in range(0, d_ff, chunk_size):
            end = min(start + chunk_size, d_ff)
            token_logits_chunk = W_out @ W_value[:, start:end]  # [vocab_size, chunk]
            top_logits_chunk, top_token_ids_chunk = torch.topk(token_logits_chunk, k=top_k_tokens, dim=0)
            del token_logits_chunk
            top_logits_chunks.append(top_logits_chunk)
            top_token_ids_chunks.append(top_token_ids_chunk)
        del W_value

        top_logits = torch.cat(top_logits_chunks, dim=1)  # [k, d_ff]
        top_token_ids = torch.cat(top_token_ids_chunks, dim=1)  # [k, d_ff]
        del top_logits_chunks, top_token_ids_chunks

        top_logits_t = top_logits.transpose(0, 1).contiguous()  # [d_ff, k]
        top_token_ids_t = top_token_ids.transpose(0, 1).contiguous()  # [d_ff, k]
        del top_logits, top_token_ids

        weights = torch.softmax(top_logits_t, dim=1)  # [d_ff, k]
        token_embs = W_out[top_token_ids_t]  # [d_ff, k, d_model]
        e_sem = (weights.unsqueeze(-1) * token_embs).sum(dim=1)  # [d_ff, d_model]
        del weights, token_embs
        e_sem = torch.nn.functional.normalize(e_sem, p=2, dim=1)

        layer_start = layer_idx * d_ff
        semantic_embeddings[layer_start:layer_start + d_ff] = e_sem.cpu().numpy()
        del e_sem

        top_token_ids_np = top_token_ids_t.cpu().numpy()
        top_logits_np = top_logits_t.cpu().numpy()
        del top_token_ids_t, top_logits_t

        for neuron_idx in range(d_ff):
            metadata.append(
                {
                    "global_id": global_id,
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "top_token_ids": top_token_ids_np[neuron_idx].tolist(),
                    "top_logits": top_logits_np[neuron_idx].tolist(),
                }
            )
            global_id += 1

        print(f"[*]   layer {layer_idx + 1}/{num_layers} done ({len(metadata)} neurons so far)")

    print(f"[*] Extracted {len(metadata)} semantic embeddings.")

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


def build_knn_candidate_clusters(semantic_embeddings, k_values=(10, 20, 40), partition="full", chunk_size=4000):
    """
    Faithful approximation of the paper's Appendix B.3 steps 3-4, fitted once
    per partition and reused across every k and every concept:
      - cosine kNN over semantic embeddings
      - each neighborhood acts as a candidate cluster
      - centroid = average embedding of cluster members

    This computes cosine similarity as a plain dot product, chunked over rows,
    rather than via sklearn's NearestNeighbors(metric="cosine") -- semantic_embeddings
    is already L2-normalized (see extract_semantic_embeddings), so cosine
    similarity IS the dot product, and sklearn's generic brute-force path
    doesn't need to be involved at all. This matters because sklearn's
    cosine kNN was measured (via journalctl -k, RSS at OOM-kill) to hold
    ~120GB+ resident at OpenVLA's ~350K-vector scale regardless of
    OMP_NUM_THREADS, i.e. it wasn't chunking tightly enough for this vector
    count. This version's peak is bounded to chunk_size * N * 4 bytes
    (~5.6GB at the default chunk_size on OpenVLA's largest partition),
    independent of thread count.
    """
    all_idx = get_partition_indices(len(semantic_embeddings), partition=partition)
    X = semantic_embeddings[all_idx]  # already L2-normalized, float32

    num_vectors = len(X)
    max_k = min(max(k_values), num_vectors)
    if max_k < 1:
        raise ValueError("No vectors available in this partition.")

    X_t = torch.from_numpy(X)  # zero-copy view over X
    neighbors = np.empty((num_vectors, max_k), dtype=np.int64)

    for start in range(0, num_vectors, chunk_size):
        end = min(start + chunk_size, num_vectors)
        sims = X_t[start:end] @ X_t.T  # [chunk, N] cosine similarities (dot products of unit vectors)
        _, top_idx = torch.topk(sims, k=max_k, dim=1)
        neighbors[start:end] = top_idx.numpy()
        del sims, top_idx

    clusters_by_k = {}
    for k in k_values:
        k = min(k, max_k)
        neighbors_k = neighbors[:, :k]
        # X[neighbors_k] would fancy-index into a new (N, k, d_model) array before
        # .mean() ever runs -- for OpenVLA's N=352,256 that's ~58GB at k=10 and
        # ~231GB at k=40, which is what was actually OOM-killing this step (the
        # chunked similarity matmul above stayed flat at ~18GB the whole time).
        # Accumulating one neighbor-slot at a time bounds peak memory to O(N x
        # d_model), the same order as X itself, regardless of k.
        centroid_sum = np.zeros((num_vectors, X.shape[1]), dtype=np.float32)
        for j in range(k):
            centroid_sum += X[neighbors_k[:, j]]
        centroids = centroid_sum / k
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


def print_cluster_summary(results, tokenizer, top_members_to_show=10):
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
            tokens = decode_top_tokens(tokenizer, member["top_token_ids"])
            print(f"  layer={member['layer']:>2}, neuron={member['neuron']:>5}, tokens={tokens}")


def compute_cluster_purity(members, tokenizer, concept_keywords):
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
        for tok in decode_top_tokens(tokenizer, member["top_token_ids"]):
            total += 1
            if any(kw.lower() in tok.lower() for kw in concept_keywords):
                matched += 1
    return matched / total if total else 0.0


def summarize_concept_results(results, tokenizer, keyword_map):
    """
    Builds a paper-ready table: one row per concept with its cluster's
    best_score (cosine sim to concept embedding), purity %, and cluster size.
    `keyword_map` is {concept: [keyword, ...]} used only for the purity check
    (independent of the kNN clustering itself).
    """
    rows = []
    for concept, info in results.items():
        keywords = keyword_map.get(concept, [concept])
        purity = compute_cluster_purity(info["members"], tokenizer, keywords)
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


def plot_concept_summary(results, tokenizer, keyword_map, output_prefix="concept_analysis"):
    """
    Saves two paper-ready PNGs: a purity bar chart and an antonym centroid
    cosine-similarity heatmap.
    """
    import matplotlib.pyplot as plt

    rows = summarize_concept_results(results, tokenizer, keyword_map)
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


def print_top_neurons_overall_by_logit(metadata, tokenizer, top_n=15):
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
        tokens_str = ", ".join(decode_top_tokens(tokenizer, neuron["top_token_ids"]))
        max_logit = neuron["top_logits"][0]
        print(f"{i+1:<12} | {max_logit:.4f}    | L{neuron['layer']:<4} | {neuron['neuron']:<6} | [{tokens_str}]")


def rank_neurons_by_keyword_frequency(metadata, tokenizer, keywords_dict, top_n=6):
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
        top_tokens = decode_top_tokens(tokenizer, neuron["top_token_ids"])
        match_count = 0
        for token in top_tokens:
            token_lower = token.lower()

            if any(neg_kw in token_lower for neg_kw in neg_keywords):
                continue

            if any(pos_kw in token_lower for pos_kw in pos_keywords):
                match_count += 1

        scored_neurons.append({
            "layer": neuron["layer"],
            "neuron": neuron["neuron"],
            "match_count": match_count,
            "top_tokens": top_tokens,
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


def run_keyword_based_ranking(metadata, tokenizer, concept_keywords_map, top_n=6):
    """
    處理多個概念的關鍵字搜尋並彙整結果。
    """
    results = {}
    for concept_name, keywords_dict in concept_keywords_map.items():
        ranked_neurons = rank_neurons_by_keyword_frequency(
            metadata=metadata,
            tokenizer=tokenizer,
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


def export_keyword_results_json(keyword_results, policy_family, policy_path, top_k_tokens, output_path):
    """
    Saves per-concept keyword-frequency metrics for cross-policy --mode combine_keyword
    comparison, mirroring export_results_json's role for the kNN method.

    keyword_purity = mean(match_count)/top_k_tokens across the top-N ranked neurons for
    that concept -- i.e. what fraction of the top-k token slots, averaged over exactly the
    neuron population the paper's Appendix C.3 steering-selection method would pick
    (highest keyword-frequency value vectors), literally match the target concept. This is
    the keyword-method analog of the kNN method's cluster purity, so the two are on a
    comparable 0-1 scale even though they're computed from different neuron populations.
    """
    payload = {
        "policy_family": policy_family,
        "policy_path": policy_path,
        "top_k_tokens": top_k_tokens,
        "concepts": {},
    }
    for concept, ranked_neurons in keyword_results.items():
        if not ranked_neurons:
            continue
        match_counts = [n["match_count"] for n in ranked_neurons]
        payload["concepts"][concept] = {
            "keyword_purity": (sum(match_counts) / len(match_counts)) / top_k_tokens,
            "top1_match_count": ranked_neurons[0]["match_count"],
            "num_neurons_considered": len(ranked_neurons),
            "top_members": ranked_neurons,
        }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[*] Saved combinable keyword results to {output_path}")


def load_multi_policy_keyword_results(paths):
    results = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        results[data["policy_family"]] = data
    return results


def plot_multi_policy_keyword_purity(results, concepts, output_path):
    """
    One grouped bar chart: x-axis = concept, one bar per policy -- the keyword-method
    (Appendix C.3) analog of plot_multi_policy_purity, which does this for the kNN method.
    """
    import matplotlib.pyplot as plt

    policies = list(results.keys())
    x = np.arange(len(concepts))
    width = 0.8 / max(len(policies), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, policy in enumerate(policies):
        values = [(results[policy]["concepts"].get(c) or {}).get("keyword_purity", 0.0) * 100 for c in concepts]
        ax.bar(x + i * width, values, width=width, label=policy)

    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(concepts)
    ax.set_ylabel("Keyword-match purity (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Keyword-frequency purity by concept and policy (Appendix C.3 method)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[*] Saved {output_path}")


def export_top_tokens_for_classification(metadata, tokenizer, output_csv_path, top_n=30):
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
            tokens_str = " | ".join(decode_top_tokens(tokenizer, neuron["top_token_ids"])[:top_n])
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
    parser.add_argument(
        "--mode",
        choices=["keyword", "knn", "export", "logit", "all", "combine", "combine_keyword"],
        default="all",
    )
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
        "--keyword_results_json",
        default=None,
        help="Path to save this run's per-concept keyword-frequency metrics as JSON, for later --mode combine_keyword",
    )
    parser.add_argument(
        "--combine_inputs",
        nargs="*",
        default=None,
        help="--mode combine/combine_keyword only: paths to --results_json/--keyword_results_json outputs from separate policy runs",
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

    if args.mode == "combine_keyword":
        if not args.combine_inputs:
            parser.error("--mode combine_keyword requires --combine_inputs <keyword_results.json> ...")

        multi_keyword_results = load_multi_policy_keyword_results(args.combine_inputs)
        all_concepts = sorted({c for data in multi_keyword_results.values() for c in data["concepts"]})

        plot_multi_policy_keyword_purity(
            multi_keyword_results, all_concepts, f"{args.combine_output_prefix}_keyword_purity.png"
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

    # Everything downstream only needs bundle's extracted arrays (semantic
    # embeddings, metadata, W_out) -- the raw model (its language_model
    # backbone alone is ~14GB for a 7B checkpoint) is never touched again.
    # Holding onto it through the kNN clustering step, which is itself
    # memory-heavy at OpenVLA's ~350K-vector scale, is what was pushing an
    # already memory-constrained host into OOM.
    del policy, text_model
    import gc

    gc.collect()

    if args.mode in ("keyword", "all") and args.keywords_json:
        with open(args.keywords_json) as f:
            concept_keywords_map = json.load(f)
        keyword_results = run_keyword_based_ranking(bundle["metadata"], tokenizer, concept_keywords_map, top_n=6)
        print_keyword_ranking_summary(keyword_results)

        if args.keyword_results_json:
            export_keyword_results_json(
                keyword_results, args.policy_family, args.policy_path, args.top_k_tokens, args.keyword_results_json
            )

    if args.mode in ("knn", "all"):
        cluster_results = run_concept_clustering(
            bundle, args.concepts, k_values=tuple(args.knn_k_values), partition=args.partition, device=device
        )
        print_cluster_summary(cluster_results, tokenizer)

        if args.purity_keywords_json:
            with open(args.purity_keywords_json) as f:
                purity_keyword_map = json.load(f)
        else:
            purity_keyword_map = {c: [c] for c in args.concepts}

        summary_rows = summarize_concept_results(cluster_results, tokenizer, purity_keyword_map)
        print_concept_summary_table(summary_rows)

        if args.plot_prefix:
            plot_concept_summary(cluster_results, tokenizer, purity_keyword_map, output_prefix=args.plot_prefix)

        if args.results_json:
            export_results_json(
                cluster_results, summary_rows, args.policy_family, args.policy_path, args.results_json
            )

    if args.export_csv or args.mode == "export":
        export_top_tokens_for_classification(
            bundle["metadata"], tokenizer, args.export_csv or "top_tokens_for_labeling.csv"
        )

    if args.mode in ("logit", "all"):
        print_top_neurons_overall_by_logit(bundle["metadata"], tokenizer, top_n=args.top_n_logit)
