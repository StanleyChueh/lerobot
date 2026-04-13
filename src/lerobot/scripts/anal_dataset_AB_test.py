# import numpy as np
# import matplotlib.pyplot as plt
# from datasets import load_dataset

# def get_episode_data(dataset_id, episode_idx=0):
#     """Extracts the state trajectory for a specific episode."""
#     dataset = load_dataset(dataset_id, split="train")
#     ep_indices = np.array(dataset["episode_index"])
#     mask = (ep_indices == episode_idx)
#     states = np.array(dataset["observation.state"])[mask]
#     return states

# def plot_trajectories():
#     print("Loading datasets for plotting...")
    
#     states_baseline = get_episode_data("ethanCSL/eval_koch_baseline", episode_idx=0)
#     states_steered = get_episode_data("ethanCSL/eval_koch_steered_slow", episode_idx=0)
    
#     # --- TIME NORMALIZATION CALCULATION ---
#     # Create an array from 0 to 100 representing the percentage of the episode
#     time_baseline = np.linspace(0, 100, len(states_baseline))
#     time_steered = np.linspace(0, 100, len(states_steered))
    
#     fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
#     # ==========================================
#     # PLOT 1: Cumulative Displacement (Speed)
#     # ==========================================
#     disp_baseline = np.mean(np.abs(states_baseline[1:] - states_baseline[:-1]), axis=1)
#     cum_disp_baseline = np.cumsum(disp_baseline)
    
#     disp_steered = np.mean(np.abs(states_steered[1:] - states_steered[:-1]), axis=1)
#     cum_disp_steered = np.cumsum(disp_steered)
    
#     # Note: Cumulative displacement is still plotted against raw steps to show duration differences
#     axs[0].plot(cum_disp_baseline, label='Baseline (0.0)', color='#1f77b4', linewidth=2.5)
#     axs[0].plot(cum_disp_steered, label='Steered Slow (-30.0)', color='#2ca02c', linestyle='--', linewidth=2.5)
#     axs[0].set_title('Cumulative Joint Displacement', fontsize=14)
#     axs[0].set_xlabel('Action Step (Raw Time)', fontsize=12)
#     axs[0].set_ylabel('Cumulative Movement (rad)', fontsize=12)
#     axs[0].legend(fontsize=12)
#     axs[0].grid(True, alpha=0.4)
    
#     # ==========================================
#     # PLOT 2: Single Joint Trajectory (TIME NORMALIZED)
#     # ==========================================
#     # Here, we plot against the 0-100% time arrays to align the shapes!
#     axs[1].plot(time_baseline, states_baseline[:, 1], label='Baseline (0.0)', color='#1f77b4', linewidth=2.5)
#     axs[1].plot(time_steered, states_steered[:, 1], label='Steered Slow (-30.0)', color='#2ca02c', linestyle='--', linewidth=2.5)
#     axs[1].set_title('Joint 1 Trajectory Arc (Time Normalized)', fontsize=14)
    
#     # Updated X-Axis label to reflect normalization
#     axs[1].set_xlabel('Task Completion Percentage (%)', fontsize=12)
#     axs[1].set_ylabel('Joint Angle (rad)', fontsize=12)
#     axs[1].legend(fontsize=12)
#     axs[1].grid(True, alpha=0.4)
    
#     plt.tight_layout()
#     output_filename = 'trajectory_comparison_normalized.png'
#     plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#     print(f"\n✅ Plot successfully saved to your current directory as: {output_filename}")

# if __name__ == "__main__":
#     plot_trajectories()
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

# ==========================================
# CONFIGURATION: Define your datasets here
# ==========================================
DATASETS = {
    "Baseline (0)": "ethanCSL/eval_koch_baseline",
    "Steered (15)": "ethanCSL/eval_koch_15",
    "Steered (30)": "ethanCSL/eval_koch_30",
    "Steered (-15)": "ethanCSL/eval_koch_-15",
    "Steered (-30)": "ethanCSL/eval_koch_-30"
}

# Define distinct colors and styles for the plot to keep it readable
PLOT_STYLES = {
    "Baseline (0)": {"color": "black", "linestyle": "-", "linewidth": 3.0, "alpha": 1.0},
    "Steered (15)": {"color": "orange", "linestyle": "--", "linewidth": 2.0, "alpha": 0.8},
    "Steered (30)": {"color": "red", "linestyle": ":", "linewidth": 2.0, "alpha": 0.8},
    "Steered (-15)": {"color": "dodgerblue", "linestyle": "--", "linewidth": 2.0, "alpha": 0.8},
    "Steered (-30)": {"color": "blue", "linestyle": ":", "linewidth": 2.0, "alpha": 0.8}
}

# ==========================================
# PART 1: QUANTITATIVE MATH (SPEED DIFFERENCE)
# ==========================================
def calculate_average_speed(dataset_id):
    """Calculates the average joint speed across all valid frames in the dataset."""
    try:
        dataset = load_dataset(dataset_id, split="train")
        states = torch.tensor(dataset["observation.state"])
        displacement = torch.abs(states[1:] - states[:-1])
        valid_displacements = displacement[torch.max(displacement, dim=1).values < 1.0]
        avg_speed = torch.mean(valid_displacements).item()
        return avg_speed
    except Exception as e:
        print(f"  [!] Could not load {dataset_id}. Error: {e}")
        return None

# ==========================================
# PART 2: QUALITATIVE VISUALS (TRAJECTORY PLOT)
# ==========================================
def get_episode_data(dataset_id, episode_idx=0):
    """Extracts the state trajectory for a specific episode."""
    try:
        dataset = load_dataset(dataset_id, split="train")
        ep_indices = np.array(dataset["episode_index"])
        mask = (ep_indices == episode_idx)
        states = np.array(dataset["observation.state"])[mask]
        if len(states) == 0:
            print(f"  [!] No data found for episode {episode_idx} in {dataset_id}")
            return None
        return states
    except Exception:
        return None

def analyze_and_plot():
    # --- 1. RUN THE MATH ---
    print("=== QUANTITATIVE ANALYSIS (ALL EPISODES) ===")
    
    speeds = {}
    for name, repo_id in DATASETS.items():
        print(f"Analyzing {name} ({repo_id})...")
        speed = calculate_average_speed(repo_id)
        if speed is not None:
            speeds[name] = speed

    print("\n--- RESULTS ---")
    baseline_speed = speeds.get("Baseline (0)")
    
    if baseline_speed is None:
        print("❌ Cannot compute differences because Baseline failed to load.")
    else:
        print(f"-> { 'Baseline (0)':<15} | Speed: {baseline_speed:.6f} rad/step | ---")
        for name, speed in speeds.items():
            if name == "Baseline (0)":
                continue
            diff = ((speed - baseline_speed) / baseline_speed) * 100
            trend = "⬇️ SLOWER" if diff < 0 else "⬆️ FASTER"
            print(f"-> {name:<15} | Speed: {speed:.6f} rad/step | Diff: {diff:>+7.2f}% ({trend})")

    # --- 2. RUN THE PLOT ---
    print("\n=== VISUAL ANALYSIS (EPISODE 0) ===")
    print("Loading datasets for plotting...")
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    for name, repo_id in DATASETS.items():
        states = get_episode_data(repo_id, episode_idx=0)
        if states is None:
            continue
            
        style = PLOT_STYLES[name]
        
        # Left Plot: Cumulative Displacement
        disp = np.mean(np.abs(states[1:] - states[:-1]), axis=1)
        cum_disp = np.cumsum(disp)
        axs[0].plot(cum_disp, label=name, **style)
        
        # Right Plot: Time Normalized Trajectory (Joint 1)
        time_norm = np.linspace(0, 100, len(states))
        axs[1].plot(time_norm, states[:, 1], label=name, **style)

    # Format Left Plot
    axs[0].set_title('Cumulative Joint Displacement (Duration & Speed)', fontsize=14)
    axs[0].set_xlabel('Action Step (Raw Time)', fontsize=12)
    axs[0].set_ylabel('Cumulative Movement (rad)', fontsize=12)
    axs[0].legend(fontsize=11)
    axs[0].grid(True, alpha=0.4)
    
    # Format Right Plot
    axs[1].set_title('Joint 1 Trajectory Arc (Time Normalized)', fontsize=14)
    axs[1].set_xlabel('Task Completion Percentage (%)', fontsize=12)
    axs[1].set_ylabel('Joint Angle (rad)', fontsize=12)
    axs[1].legend(fontsize=11)
    axs[1].grid(True, alpha=0.4)
    
    plt.tight_layout()
    output_filename = 'parameter_sweep_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot successfully saved to your current directory as: {output_filename}")

if __name__ == "__main__":
    analyze_and_plot()