import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def get_full_model_cluster(policy_path="ethanCSL/svla_koch_sorting_n_stacking", target_words=["slow", "safe", "careful"]):
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

    full_cluster = {}
    total_neurons = 0

    print(f"\n[*] Scanning ALL layers for concepts: {target_words}...")

    for layer_idx in range(num_layers):
        vlm_W_value = text_model.layers[layer_idx].mlp.down_proj.weight.data.to(device)
        vlm_logits = torch.matmul(W_unembed, vlm_W_value)

        top_scores, top_token_ids = torch.topk(vlm_logits, 10, dim=0)
        num_neurons_in_layer = vlm_logits.shape[1]
        
        layer_matches = []
        for neuron_idx in range(num_neurons_in_layer):
            tokens = [tokenizer.decode(t.item()).replace('\n', '').strip().lower() for t in top_token_ids[:, neuron_idx]]
            if any(target in tokens for target in target_words):
                layer_matches.append(neuron_idx)
        
        if layer_matches:
            full_cluster[layer_idx] = layer_matches
            total_neurons += len(layer_matches)

    print("\n" + "="*60)
    print(f"✅ Full-Model Scan Complete! Found {total_neurons} total neurons across {len(full_cluster)} layers.")
    print("🎯 COPY THIS EXACT DICTIONARY INTO YOUR ROBOT SCRIPT:")
    print(f"multi_layer_cluster = {full_cluster}")
    print("="*60)

if __name__ == "__main__":
    # search_targets = ["slow", "safe", "careful"]
    # search_targets = ["fast", "quick", "rapid", "risk"]
    search_targets = ["high", "higher", "top", "up"]
    get_full_model_cluster(target_words=search_targets)