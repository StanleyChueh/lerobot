import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def find_top_neurons_for_concept(concept_word="fast", target_layer=8, top_n=5):
    print(f"Loading SmolVLA to search for '{concept_word}' neurons...")
    
    # 1. Load the policy directly, bypassing the dataset requirement
    policy_path = "ethanCSL/svla_koch_sorting_n_stacking" 
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    
    model = policy.model.vlm_with_expert
    tokenizer = model.processor.tokenizer
    
    # 2. Get the target word's Token ID
    token_ids = tokenizer.encode(concept_word, add_special_tokens=False)
    if not token_ids:
        print(f"Word '{concept_word}' not found in tokenizer.")
        return
    target_token_id = token_ids[0]
    
    # 3. Get the VLM's Unembedding Matrix (The Dictionary)
    if hasattr(model.vlm, "language_model"):
        lm_head_weight = model.vlm.language_model.lm_head.weight.detach()
    else:
        lm_head_weight = model.vlm.lm_head.weight.detach()
        
    # 4. Get the Action Expert's FFN down_proj weights for the target layer
    text_layers = model.get_vlm_model().text_model.layers
    down_proj_weight = mlp_layer.down_proj.weight.detach()
    
    print(f"lm_head shape: {lm_head_weight.shape}")
    print(f"down_proj shape: {down_proj_weight.shape}")
    print("Projecting all FFN Neurons to Vocabulary Space...")
    
    # 5. Project all neurons to the vocabulary
    neuron_vocab_logits = torch.matmul(lm_head_weight, down_proj_weight.to(lm_head_weight.dtype))
    
    # 6. Get scores for our specific word
    word_neuron_scores = neuron_vocab_logits[target_token_id]
    
    # 7. Find the top N neurons
    top_scores, top_neurons = torch.topk(word_neuron_scores, top_n)
    
    print("\n" + "="*50)
    print(f"🏆 TOP NEURONS FOR CONCEPT: '{concept_word}' (Layer {target_layer}) 🏆")
    print("="*50)
    
    for rank, (score, neuron_idx) in enumerate(zip(top_scores, top_neurons)):
        print(f"Rank {rank+1}: Neuron #{neuron_idx.item()} | Logit Score: {score.item():.3f}")
        
    return top_neurons[0].item()

if __name__ == "__main__":
    # You can change this word to 'red', 'slow', 'high', etc.
    best_neuron = find_top_neurons_for_concept(concept_word="slow", target_layer=8)