import torch
import numpy as np
from datasets import load_dataset

def calculate_max_height(dataset_id, z_index=2):
    """Calculates the average maximum height reached across all episodes."""
    dataset = load_dataset(dataset_id, split="train")
    
    # Extract states and episode indices
    states = torch.tensor(dataset["observation.state"])
    episodes = np.array(dataset["episode_index"])
    
    max_heights = []
    for ep_id in np.unique(episodes):
        # Isolate the Z-axis (height) for this specific episode
        ep_heights = states[episodes == ep_id][:, z_index]
        max_heights.append(torch.max(ep_heights).item())
    
    avg_max_height = np.mean(max_heights)
    print(f"-> {dataset_id}: Avg Max Height = {avg_max_height:.4f} rad")
    return avg_max_height

if __name__ == "__main__":
    baseline_h = calculate_max_height("ethanCSL/eval_koch_baseline")
    steered_h = calculate_max_height("ethanCSL/eval_koch_high")
    
    diff = ((steered_h - baseline_h) / baseline_h) * 100
    print(f"=== HEIGHT PROOF: {diff:>+7.2f}% change in max elevation ===")
