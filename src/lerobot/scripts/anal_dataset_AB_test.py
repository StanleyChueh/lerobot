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
#     states_steered = get_episode_data("ethanCSL/eval_koch_high", episode_idx=0)
#     states_baseline[1:] - states_baseline[:-1]
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
#     axs[0].plot(cum_disp_steered, label='Steered fast (-30.0)', color='#2ca02c', linestyle='--', linewidth=2.5)
#     axs[0].set_title('Cumulative Joint Displacement', fontsize=14)
#     axs[0].set_xlabel('Action Stepstates_steered (Raw Time)', fontsize=12)
#     axs[0].set_ylabel('Cumulative Movement (rad)', fontsize=12)
#     axs[0].legend(fontsize=12)
#     axs[0].grid(True, alpha=0.4)
    
#     # ==========================================
#     # PLOT 2: Single Joint Trajectory (TIME NORMALIZED)
#     # ==========================================
#     # Here, we plot against the 0-100% time arrays to align the shapes!
#     axs[1].plot(time_baseline, states_baseline[:, 1], label='Baseline (0.0)', color='#1f77b4', linewidth=2.5)
#     axs[1].plot(time_steered, states_steered[:, 1], label='Steered fast (-30.0)', color='#2ca02c', linestyle='--', linewidth=2.5)
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


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from datasets import load_dataset

# # ==========================================
# # CONFIGURATION: Define your datasets here
# # ==========================================
# DATASETS = {
#     "Baseline (0)": "ethanCSL/eval_koch_baseline",
#     "Steered (15)": "ethanCSL/eval_koch_15",
#     "Steered (30)": "ethanCSL/eval_koch_30",
#     "Steered (-15)": "ethanCSL/eval_koch_-15",
#     "Steered (-30)": "ethanCSL/eval_koch_-30"
# }

# # Define distinct colors and styles for the plot to keep it readable
# PLOT_STYLES = {
#     "Baseline (0)": {"color": "black", "linestyle": "-", "linewidth": 3.0, "alpha": 1.0},
#     "Steered (15)": {"color": "orange", "linestyle": "--", "linewidth": 2.0, "alpha": 0.8},
#     "Steered (30)": {"color": "red", "linestyle": ":", "linewidth": 2.0, "alpha": 0.8},
#     "Steered (-15)": {"color": "dodgerblue", "linestyle": "--", "linewidth": 2.0, "alpha": 0.8},
#     "Steered (-30)": {"color": "blue", "linestyle": ":", "linewidth": 2.0, "alpha": 0.8}
# }

# # ==========================================
# # PART 1: QUANTITATIVE MATH (SPEED DIFFERENCE)
# # ==========================================
# def calculate_average_speed(dataset_id):
#     """Calculates the average joint speed across all valid frames in the dataset."""
#     try:
#         dataset = load_dataset(dataset_id, split="train")
#         states = torch.tensor(dataset["observation.state"])
#         displacement = torch.abs(states[1:] - states[:-1])
#         valid_displacements = displacement[torch.max(displacement, dim=1).values < 1.0]
#         avg_speed = torch.mean(valid_displacements).item()
#         return avg_speed
#     except Exception as e:
#         print(f"  [!] Could not load {dataset_id}. Error: {e}")
#         return None

# # ==========================================
# # PART 2: QUALITATIVE VISUALS (TRAJECTORY PLOT)
# # ==========================================
# def get_episode_data(dataset_id, episode_idx=0):
#     """Extracts the state trajectory for a specific episode."""
#     try:
#         dataset = load_dataset(dataset_id, split="train")
#         ep_indices = np.array(dataset["episode_index"])
#         mask = (ep_indices == episode_idx)
#         states = np.array(dataset["observation.state"])[mask]
#         if len(states) == 0:
#             print(f"  [!] No data found for episode {episode_idx} in {dataset_id}")
#             return None
#         return states
#     except Exception:
#         return None

# def analyze_and_plot():
#     # --- 1. RUN THE MATH ---
#     print("=== QUANTITATIVE ANALYSIS (ALL EPISODES) ===")
    
#     speeds = {}
#     for name, repo_id in DATASETS.items():
#         print(f"Analyzing {name} ({repo_id})...")
#         speed = calculate_average_speed(repo_id)
#         if speed is not None:
#             speeds[name] = speed

#     print("\n--- RESULTS ---")
#     baseline_speed = speeds.get("Baseline (0)")
    
#     if baseline_speed is None:
#         print("❌ Cannot compute differences because Baseline failed to load.")
#     else:
#         print(f"-> { 'Baseline (0)':<15} | Speed: {baseline_speed:.6f} rad/step | ---")
#         for name, speed in speeds.items():
#             if name == "Baseline (0)":
#                 continue
#             diff = ((speed - baseline_speed) / baseline_speed) * 100
#             trend = "⬇️ SLOWER" if diff < 0 else "⬆️ FASTER"
#             print(f"-> {name:<15} | Speed: {speed:.6f} rad/step | Diff: {diff:>+7.2f}% ({trend})")

#     # --- 2. RUN THE PLOT ---
#     print("\n=== VISUAL ANALYSIS (EPISODE 0) ===")
#     print("Loading datasets for plotting...")
    
#     fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
#     for name, repo_id in DATASETS.items():
#         states = get_episode_data(repo_id, episode_idx=0)
#         if states is None:
#             continue
            
#         style = PLOT_STYLES[name]
        
#         # Left Plot: Cumulative Displacement
#         disp = np.mean(np.abs(states[1:] - states[:-1]), axis=1)
#         cum_disp = np.cumsum(disp)
#         axs[0].plot(cum_disp, label=name, **style)
        
#         # Right Plot: Time Normalized Trajectory (Joint 1)
#         time_norm = np.linspace(0, 100, len(states))
#         axs[1].plot(time_norm, states[:, 1], label=name, **style)

#     # Format Left Plot
#     axs[0].set_title('Cumulative Joint Displacement (Duration & Speed)', fontsize=14)
#     axs[0].set_xlabel('Action Step (Raw Time)', fontsize=12)
#     axs[0].set_ylabel('Cumulative Movement (rad)', fontsize=12)
#     axs[0].legend(fontsize=11)
#     axs[0].grid(True, alpha=0.4)
    
#     # Format Right Plot
#     axs[1].set_title('Joint 1 Trajectory Arc (Time Normalized)', fontsize=14)
#     axs[1].set_xlabel('Task Completion Percentage (%)', fontsize=12)
#     axs[1].set_ylabel('Joint Angle (rad)', fontsize=12)
#     axs[1].legend(fontsize=11)
#     axs[1].grid(True, alpha=0.4)
    
#     plt.tight_layout()
#     output_filename = 'parameter_sweep_analysis.png'
#     plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#     print(f"\n✅ Plot successfully saved to your current directory as: {output_filename}")

# if __name__ == "__main__":
#     analyze_and_plot()

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

def extract_episode_states(dataset, episode_index, state_key='observation.state'):
    """
    從資料集中提取特定 episode 的 joint state 序列。
    """
    # 假設資料集支援過濾或類似 Hugging Face datasets 的操作
    # 實作時可依據實際的 Dataset API 調整 (例如 pandas, dict 等)
    episode_data = dataset.filter(lambda x: x['episode_index'] == episode_index)
    
    # 提取狀態並轉換為 NumPy array
    # 預期轉換後的形狀為 (num_frames, num_joints)
    states = np.array(episode_data[state_key])
    
    # 若陣列為多維度 (例如包含了 batch size: (1, num_frames, num_joints))，則壓縮維度
    if states.ndim > 2:
        states = np.squeeze(states)
        
    return states

def plot_joint_states_comparison(ds1, ds2, ds3, episode_index, num_joints=6, state_key='observation.state'):
    """
    比較三個資料集在同一個 episode 中的 joint state，並繪製折線圖。
    
    參數:
        ds1, ds2, ds3: 三個不同的資料集物件
        episode_index (int): 欲比較的 episode 編號
        num_joints (int): 機器人的關節數量
        state_key (str): 資料集中代表 joint state 的欄位名稱
    """
    # 提取三個資料集的序列資料
    states1 = extract_episode_states(ds1, episode_index, state_key)
    states2 = extract_episode_states(ds2, episode_index, state_key)
    states3 = extract_episode_states(ds3, episode_index, state_key)

    # 確保資料長度足夠繪圖，並以最短的 frame 數為基準（若三個資料集長度略有差異）
    min_frames = max(len(states1), len(states2), len(states3))
    states1 = states1[:min_frames]
    states2 = states2[:min_frames]
    states3 = states3[:min_frames]

    # 建立子圖，為每個 Joint 建立一個獨立的圖表以便清晰觀察
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2.5 * num_joints), sharex=True)
    
    if num_joints == 1:
        axes = [axes] # 確保單一子圖時也能迭代

    labels = ['Dataset 1--baseline', 'Dataset 2--high', 'Dataset 3-low']
    line_styles = ['-', '--', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for j in range(num_joints):
        ax = axes[j]
        
        # 繪製原始數值疊加比較
        ax.plot(states1[:, j], label=labels[0], linestyle=line_styles[0], color=colors[0], linewidth=2, alpha=0.8)
        ax.plot(states2[:, j], label=labels[1], linestyle=line_styles[1], color=colors[1], linewidth=2, alpha=0.8)
        ax.plot(states3[:, j], label=labels[2], linestyle=line_styles[2], color=colors[2], linewidth=2, alpha=0.8)
        
        # 設定標題與格式
        ax.set_title(f'Joint {j} State over Time')
        ax.set_ylabel('State Value (rad / pos)')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 只在第一個子圖顯示圖例，保持畫面整潔
        if j == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # X 軸標籤與整體排版
    plt.xlabel('Timestep (Frames)')
    plt.tight_layout()
    plt.show()

    # ==========================================
    # PLOT 2: Per-Joint Cumulative Displacement
    # ==========================================
    # 建立一個新的畫布，避免與之前的 axes 混淆
    plt.figure(figsize=(12, 8))
    
    # 計算每個資料集的每個關節絕對位移
    # (num_steps-1, num_joints)
    disp1 = np.abs(states1[1:] - states1[:-1])
    disp2 = np.abs(states2[1:] - states2[:-1])
    disp3 = np.abs(states3[1:] - states3[:-1])
    
    # 計算累積位移
    cum1 = np.cumsum(disp1, axis=0)
    cum2 = np.cumsum(disp2, axis=0)
    cum3 = np.cumsum(disp3, axis=0)
    
    # 設定顏色
    joint_colors = plt.cm.get_cmap('tab10', num_joints)
    
    for j in range(num_joints):
        color = joint_colors(j)
        # Baseline: 實線 (Solid)
        plt.plot(cum1[:, j], color=color, linestyle='-', alpha=0.3, linewidth=1)
        # High: 虛線 (Dashed)
        plt.plot(cum2[:, j], color=color, linestyle='--', alpha=0.8, linewidth=2, 
                 label=f'Joint {j} (High)' if j == 0 else "") 
        # Low: 點線 (Dotted)
        plt.plot(cum3[:, j], color=color, linestyle=':', alpha=0.8, linewidth=2,
                 label=f'Joint {j} (Low)' if j == 0 else "")
        
        # 標記每組線段屬於哪個關節 (放在線段末端或是用 legend)
        plt.text(len(cum1)-1, cum1[-1, j], f' J{j}', color=color, va='center')

    plt.title('Per-Joint Cumulative Displacement Comparison', fontsize=14)
    plt.xlabel('Timestep (Frames)', fontsize=12)
    plt.ylabel('Cumulative Movement (rad)', fontsize=12)
    
    # 自定義圖例，說明線條樣式
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', linestyle='-', alpha=0.5),
                    Line2D([0], [0], color='gray', linestyle='--'),
                    Line2D([0], [0], color='gray', linestyle=':')]
    
    plt.legend(custom_lines, ['Baseline', 'High (Steered)', 'Low (Steered)'], loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

# ==========================================
# 執行範例 
# ==========================================

if __name__ == "__main__":
    print("正在載入資料集，請稍候...")
    
    # 1. 將名稱加上引號，並透過 load_dataset 載入為資料集物件
    # 通常 LeRobot 的資料集會放在 'train' split 中
    dataset_baseline = load_dataset("ethanCSL/eval_koch_baseline", split="train")
    dataset_high = load_dataset("ethanCSL/eval_koch_high", split="train")
    dataset_low = load_dataset("ethanCSL/eval_koch_low", split="train")

    print("資料集載入完成，開始繪製圖表...")
    
    # 2. 將實際的資料集物件傳入函式中
    plot_joint_states_comparison(
        ds1=dataset_baseline, 
        ds2=dataset_high, 
        ds3=dataset_low, 
        episode_index=1, 
        num_joints=6,           # 視你的 Koch 機器手臂硬體設定調整
        state_key='observation.state' 
    )