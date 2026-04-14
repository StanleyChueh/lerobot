import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def get_multi_concept_clusters(policy_path="ethanCSL/svla_koch_sorting_n_stacking_vla_steering", concept_targets=None):
    if concept_targets is None:
        concept_targets = {}
        
    print(f"[*] Loading SmolVLA Policy: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    vlm_model = policy.model.vlm_with_expert.vlm
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    W_unembed = vlm_model.lm_head.weight.data.to(device)
    num_layers = len(text_model.layers)

    # Initialize storage for all concepts
    full_clusters = {concept: {} for concept in concept_targets.keys()}
    total_neurons = {concept: 0 for concept in concept_targets.keys()}

    print(f"\n[*] Scanning ALL layers for concepts: {list(concept_targets.keys())}...")

    for layer_idx in range(num_layers):
        # 1. Project the layer's neurons into vocabulary space (Heavy lifting done ONCE)
        vlm_W_value = text_model.layers[layer_idx].mlp.down_proj.weight.data.to(device)
        vlm_logits = torch.matmul(W_unembed, vlm_W_value)

        # 2. Extract top 5 tokens for every neuron
        top_scores, top_token_ids = torch.topk(vlm_logits, 5, dim=0)
        num_neurons_in_layer = vlm_logits.shape[1]
        
        # Temporary storage for matches in this specific layer
        layer_matches = {concept: [] for concept in concept_targets.keys()}
        
        # 3. Check each neuron against ALL concept lists
        for neuron_idx in range(num_neurons_in_layer):
            tokens = [tokenizer.decode(t.item()).replace('\n', '').strip().lower() for t in top_token_ids[:, neuron_idx]]
            
            for concept, target_words in concept_targets.items():
                if any(target in tokens for target in target_words):
                    layer_matches[concept].append(neuron_idx)
        
        # 4. Save the matches to the main dictionary
        for concept in concept_targets.keys():
            if layer_matches[concept]:
                full_clusters[concept][layer_idx] = layer_matches[concept]
                total_neurons[concept] += len(layer_matches[concept])

    # Print results cleanly
    print("\n" + "="*60)
    print("✅ Multi-Concept Scan Complete!")
    print("="*60)
    
    for concept in concept_targets.keys():
        print(f"\n🎯 [ {concept.upper()} CLUSTER ]")
        print(f"Total Neurons: {total_neurons[concept]} across {len(full_clusters[concept])} layers.")
        print(f"cluster_{concept} = {full_clusters[concept]}")

if __name__ == "__main__":
    # Define all the semantic concepts you want to isolate in one go
    target_concepts = {
        "slow": ["slow", "safe", "careful", "cautious", "steady", "gradual"],
        "fast": ["fast", "quick", "rapid", "risk"],
        "high": ["high", "higher", "top", "up"],
        "low":  ["low", "lower", "bottom", "down"]
    }
    
    get_multi_concept_clusters(concept_targets=target_concepts)


