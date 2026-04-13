import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def search_vla_concepts(policy_path="ethanCSL/svla_koch_sorting_n_stacking", layer_idx=8, target_words=["slow", "safe", "careful"]):
    print(f"[*] Loading SmolVLA Policy: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    vlm_model = policy.model.vlm_with_expert.vlm
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    # 1. Get weights and calculate logits for ALL neurons in the layer
    W_unembed = vlm_model.lm_head.weight.data.to(device)
    vlm_W_value = text_model.layers[layer_idx].mlp.down_proj.weight.data.to(device)
    
    print(f"[*] Projecting all neurons into Vocabulary Space...")
    vlm_logits = torch.matmul(W_unembed, vlm_W_value)
    
    num_neurons = vlm_logits.shape[1]
    print(f"\n🔍 SCANNING ALL {num_neurons} NEURONS FOR: {target_words} 🔍")
    print("="*60)

    # 2. Fast Batched Top-K for the entire layer
    # top_token_ids shape: [10, num_neurons]
    top_scores, top_token_ids = torch.topk(vlm_logits, 10, dim=0) 
    
    found_clusters = []

    # 3. Search the layer for the target words
    for neuron_idx in range(num_neurons):
        # Decode top 10 tokens for this specific neuron
        tokens = [tokenizer.decode(t.item()).replace('\n', '').strip().lower() for t in top_token_ids[:, neuron_idx]]
        
        # Check if any of our target words are in this neuron's top tokens
        if any(target in tokens for target in target_words):
            print(f"🎯 FOUND MATCH! VLM Neuron {neuron_idx:04d} -> {tokens[:7]}")
            found_clusters.append(neuron_idx)

    print("="*60)
    print(f"[*] Total neurons found matching {target_words}: {len(found_clusters)}")
    if found_clusters:
        print(f"[*] To steer this behavior, use: behavior_cluster = {found_clusters}")

if __name__ == "__main__":
    # Change these words to whatever behavior you want to find!
    # Examples: ["low", "down"], ["fast", "quick"], ["grasp", "close"]
    search_targets = ["slow", "safe", "careful"]
    search_vla_concepts(layer_idx=8, target_words=search_targets)