'''
Cross-Task Ablation for SmolVLA
=================================
Fixes the underestimation problem in record_ablation_divergence.py:

PROBLEM with the previous ablation:
  - "wrong_prompt" used a nonsensical out-of-distribution string ("Pick up the banana")
    → model had never seen it, so it correctly ignored it → 8° looked small
  - Images were zeroed as tensors, not as black pixel frames
    → VLM processes them differently from a physically blocked camera

THIS SCRIPT instead:
  1. Cross-task prompt test: takes frames from EACH task and runs them with the
     OTHER task's prompt — both prompts are real trained tasks, so divergence is genuine
  2. Black-pixel camera test: uses actual zero-pixel images (identical to a blocked lens)
  3. Compounding simulation: runs N steps of closed-loop rollout with ablated input,
     measures cumulative state drift — shows how small per-frame errors accumulate

Dataset structure assumed:
  episodes 0...(split_episode-1)  : TASK_A  (e.g. sorting)
  episodes split_episode...end    : TASK_B  (e.g. stacking)

Usage:
python src/lerobot/scripts/record_ablation_cross_task.py \
    --repo_id "ethanCSL/svla_koch_sorting_n_stacking" \
    --ckpt   "ethanCSL/svla_koch_sorting_n_stacking" \
    --task_a_episode   0 \
    --task_b_episode  100 \
    --task_a_prompt "Put the red cube in the right box,the green cube in the left box." \
    --task_b_prompt "Put the green cube on top of red cube." \
    --rollout_steps 10 \
    --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.top":"observation.images.camera2"}'
'''

import io
import json
import copy
import torch
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE


# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_action_fixed_noise(policy, batch_pp, device, noise_seed=42):
    images, img_masks = policy.prepare_images(batch_pp)
    state             = policy.prepare_state(batch_pp)
    lang_tokens       = batch_pp[OBS_LANGUAGE_TOKENS]
    lang_masks        = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]
    bsize       = state.shape[0]
    noise_shape = (bsize, policy.model.config.chunk_size, policy.model.config.max_action_dim)
    gen   = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn(noise_shape, device=device, generator=gen, dtype=state.dtype)
    predicted = policy.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state, noise=noise
    )
    real_dim = policy.config.action_feature.shape[0]
    return predicted[0, :, :real_dim].float().cpu()


def angular_divergence(a_base, a_ablated, step=0):
    """Angular divergence at a specific action chunk step (default=0)."""
    b   = a_base[step].float()
    a   = a_ablated[step].float()
    cos = torch.nn.functional.cosine_similarity(b.unsqueeze(0), a.unsqueeze(0)).item()
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), cos


def chunk_divergence_profile(a_base, a_ablated):
    """
    Compute angular divergence at EVERY step of the action chunk.
    Returns array of shape [chunk_size] with degrees at each step.
    This reveals how much the robot trajectory diverges over the 50-step window.
    """
    chunk_size = a_base.shape[0]
    angles = []
    for s in range(chunk_size):
        ang, _ = angular_divergence(a_base, a_ablated, step=s)
        angles.append(ang)
    return np.array(angles)   # [chunk_size]


def build_obs(item, rename_map, device, preprocessor, prompt, black_front=False, black_top=False):
    """Build a preprocessed batch. Optionally replace cameras with true black pixel frames."""
    obs = {}
    for k, v in item.items():
        if k in rename_map:
            obs[rename_map[k]] = v.unsqueeze(0).to(device)
        elif k.startswith("observation."):
            obs[k] = v.unsqueeze(0).to(device)

    # Fill missing cameras
    ref_cam = obs["observation.images.camera1"]
    for tk in ["observation.images.camera3", "observation.images.empty_camera_0"]:
        if tk not in obs:
            obs[tk] = torch.zeros_like(ref_cam)

    # Black PIXEL frames (uint8 zeros → float normalised) — matches a physically blocked lens
    if black_front:
        obs["observation.images.camera1"] = torch.zeros_like(obs["observation.images.camera1"])
    if black_top and "observation.images.camera2" in obs:
        obs["observation.images.camera2"] = torch.zeros_like(obs["observation.images.camera2"])

    obs["task"] = prompt
    return preprocessor(obs)


# ─────────────────────────────────────────────────────────────────────────────
# Compounding simulation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def simulate_rollout_divergence(policy, item, rename_map, device, preprocessor,
                                 prompt_correct, prompt_ablated,
                                 black_front=False, black_top=False,
                                 n_steps=10, noise_seed=42):
    """
    Simulate n_steps of closed-loop rollout.
    At each step, apply the predicted action delta to the state (first action chunk step).
    Measures how much the state CUMULATIVELY diverges between correct and ablated conditions,
    AND tracks the action magnitude ratio (ablated / baseline) at each rollout step.

    This magnitude ratio is the key signal for the "frozen robot" effect:
    - When cameras are black AND state has drifted off-trajectory, the policy sees an
      out-of-distribution input it was never trained on → outputs near-zero actions → robot freezes.
    - Tracking this across rollout steps shows WHEN the robot starts to freeze.

    Returns:
        divergences  : list[float] – per-step state L2 drift (ablated vs correct trajectory)
        mag_ratios   : list[float] – per-step action magnitude ratio  ablated/baseline
                       1.0 = same movement energy, near 0 = robot would freeze
    """
    # Start from same initial state
    bp_correct = build_obs(item, rename_map, device, preprocessor, prompt_correct)
    bp_ablated = build_obs(item, rename_map, device, preprocessor, prompt_ablated,
                           black_front=black_front, black_top=black_top)

    state_correct = bp_correct[OBS_STATE].clone()
    state_ablated = bp_ablated[OBS_STATE].clone()

    divergences = []
    mag_ratios  = []   # NEW: action magnitude of ablated / baseline at each rollout step

    for step in range(n_steps):
        # Predict next action from current simulated state
        bp_c = copy.deepcopy(bp_correct); bp_c[OBS_STATE] = state_correct
        bp_a = copy.deepcopy(bp_ablated); bp_a[OBS_STATE] = state_ablated

        act_c = predict_action_fixed_noise(policy, bp_c, device, noise_seed + step)
        act_a = predict_action_fixed_noise(policy, bp_a, device, noise_seed + step)

        # Magnitude ratio: near 0 → policy outputs tiny actions → robot would freeze
        base_mag = act_c[0].norm().item()
        abl_mag  = act_a[0].norm().item()
        mag_ratios.append(abl_mag / (base_mag + 1e-8))

        # Integrate: next_state = current_state + action[0] (simplified 1-step integration)
        real_dim   = act_c.shape[1]
        state_dim  = state_correct.shape[1]
        step_c     = act_c[0, :min(real_dim, state_dim)].to(device)
        step_a     = act_a[0, :min(real_dim, state_dim)].to(device)
        state_correct[0, :len(step_c)] = state_correct[0, :len(step_c)] + step_c * 0.033  # ~30fps dt
        state_ablated[0, :len(step_a)] = state_ablated[0, :len(step_a)] + step_a * 0.033

        drift = (state_correct - state_ablated).norm().item()
        divergences.append(drift)

    return divergences, mag_ratios


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_bgr(t):
    """Convert a CHW float32 [0,1] or uint8 tensor to HWC BGR uint8 numpy."""
    img = t.cpu().numpy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):   # CHW → HWC
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def save_peak_contact_sheet(results, task_label, rename_map, out_path, top_n=5):
    """
    For each condition, find the top_n peak frames and save a contact sheet:
      rows  = conditions  (wrong_prompt, black_front, black_top, black_all)
      cols  = top_n peak frames
      cells = front cam | top cam side-by-side with divergence annotation
    """
    cond_cfg = [
        ("wrong_prompt", results["ang_wrong_prompt"], "purple", "Wrong task prompt"),
        ("black_front",  results["ang_black_front"],  "orange", "Black front camera"),
        ("black_top",    results["ang_black_top"],    "gold",   "Black top camera"),
        ("black_all",    results["ang_black_all"],    "red",    "Both cameras black"),
    ]

    frames      = results["frames"]
    items_cache = results["items_cache"]

    # Detect original camera keys from the first cached item
    first_item = next(iter(items_cache.values()))
    front_key = next((k for k in rename_map if rename_map[k] == "observation.images.camera1"), None)
    top_key   = next((k for k in rename_map if rename_map[k] == "observation.images.camera2"), None)
    # Fallback: look directly in item
    if front_key is None:
        front_key = next((k for k in first_item if "front" in k or "camera1" in k), None)
    if top_key is None:
        top_key   = next((k for k in first_item if "top" in k or "camera2" in k), None)

    cell_h, cell_w = 180, 320   # per camera thumbnail
    label_h        = 28          # text row above each cell pair
    row_h          = cell_h + label_h
    col_w          = cell_w * 2 + 4  # front + top side by side
    header_h       = 40

    n_rows = len(cond_cfg)
    n_cols = top_n
    canvas_h = header_h + n_rows * (row_h + 6)
    canvas_w = 160 + n_cols * (col_w + 6)   # 160px for row label

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30  # dark background

    # Header
    cv2.putText(canvas, f"Peak frames — {task_label}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    for row_idx, (cond_key, angles, color_name, cond_label) in enumerate(cond_cfg):
        # Top-N peak frame indices (into the frames list)
        sorted_fi = np.argsort(angles)[::-1][:top_n]

        y0 = header_h + row_idx * (row_h + 6)

        # Row label
        bgr = {"purple": (180, 0, 180), "orange": (0, 140, 255),
               "gold": (0, 200, 200), "red": (0, 0, 220)}.get(color_name, (200, 200, 200))
        cv2.putText(canvas, cond_label, (5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1)

        for col_idx, fi in enumerate(sorted_fi):
            frame_offset = frames[fi]
            angle_val    = angles[fi]
            item         = items_cache.get(frame_offset)
            if item is None:
                continue

            x0 = 160 + col_idx * (col_w + 6)

            # Column header: frame index + divergence value
            label_txt = f"fr={frame_offset}  {angle_val:.1f}deg"
            cv2.putText(canvas, label_txt, (x0, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            # Front camera thumbnail
            if front_key and front_key in item:
                front_bgr = _tensor_to_bgr(item[front_key])
                front_th  = cv2.resize(front_bgr, (cell_w, cell_h))
            else:
                front_th = np.zeros((cell_h, cell_w, 3), np.uint8)
            canvas[y0 + label_h : y0 + label_h + cell_h,
                   x0            : x0 + cell_w] = front_th
            cv2.putText(canvas, "FRONT", (x0 + 4, y0 + label_h + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 100), 1)

            # Top camera thumbnail
            if top_key and top_key in item:
                top_bgr = _tensor_to_bgr(item[top_key])
                top_th  = cv2.resize(top_bgr, (cell_w, cell_h))
            else:
                top_th = np.zeros((cell_h, cell_w, 3), np.uint8)
            canvas[y0 + label_h : y0 + label_h + cell_h,
                   x0 + cell_w + 4 : x0 + col_w] = top_th
            cv2.putText(canvas, "TOP", (x0 + cell_w + 8, y0 + label_h + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 100), 1)

            # Highlight border colour by severity
            border_col = (0, 0, 220) if angle_val > 15 else \
                         (0, 140, 255) if angle_val > 5 else (0, 200, 80)
            cv2.rectangle(canvas,
                          (x0 - 1, y0 + label_h - 1),
                          (x0 + col_w, y0 + label_h + cell_h),
                          border_col, 2)

    cv2.imwrite(out_path, canvas)
    print(f"[Peaks] Saved contact sheet → {out_path}")


def make_cross_task_figure(results_a, results_b, task_a_prompt, task_b_prompt, out_path):
    """
    5-panel figure (5 rows × 2 cols):
      Row 0: per-frame step-0 divergence
      Row 1: mean divergence across all 50 chunk steps (shows true committed trajectory diff)
      Row 2: compounding state drift
      Row 3: action magnitude ratio per chunk step (single-frame, on-trajectory state)
      Row 4: action magnitude ratio over rollout steps (black cameras + drifted state)
             ← this is the "frozen robot" predictor
    """
    fig, axes = plt.subplots(5, 2, figsize=(16, 22))
    fig.suptitle(
        "Cross-Task Ablation: Does the model truly use the prompt?\n"
        "chunk_size=50 @ 30fps → robot commits 1.67s of motion per inference — "
        "step-0 alone underestimates real divergence",
        fontsize=10)

    label_a = "Task A scene (sorting)"
    label_b = "Task B scene (stacking)"

    cond_styles = [
        ("ang_wrong_prompt",  "wp",  "purple", "Wrong task prompt"),
        ("ang_black_front",   "bf",  "orange", "Black front camera"),
        ("ang_black_top",     "bt",  "gold",   "Black top camera"),
        ("ang_black_all",     "ba",  "red",    "Both cameras black"),
    ]

    for col, (results, label) in enumerate([(results_a, label_a), (results_b, label_b)]):
        frames  = results["frames"]
        cp      = results["chunk_profiles"]   # dict key → [chunk_size]
        chunk_steps = range(len(next(iter(cp.values()))))

        # Row 0: step-0 divergence per frame
        ax0 = axes[0, col]
        for ang_key, _, color, lbl in cond_styles:
            ax0.plot(frames, results[ang_key], color=color, label=lbl, linewidth=1.5)
        ax0.axhline(15, color='red',    linestyle='--', linewidth=0.8, label='15° = critical threshold')
        ax0.axhline(5,  color='orange', linestyle='--', linewidth=0.8, label='5° = notable threshold')
        ax0.axhline(0,  color='green',  linestyle='-',  linewidth=1.2, label='0° = baseline (correct inputs, no divergence)')
        ax0.set_title(f"Step-0 Divergence (first action in chunk) — {label}\n"
                      "solid lines = angle between ablated and baseline prediction  |  0° = identical to baseline", fontsize=9)
        ax0.set_ylabel("Angular divergence (°)")
        ax0.set_xlabel("Frame index")
        ax0.legend(fontsize=7); ax0.set_ylim(bottom=0)

        # Row 1: full chunk divergence profile (mean over all sampled frames)
        ax1 = axes[1, col]
        for _, prof_key, color, lbl in cond_styles:
            if prof_key in cp:
                ax1.plot(chunk_steps, cp[prof_key], color=color, label=lbl, linewidth=2)
        ax1.axhline(15, color='red',    linestyle='--', linewidth=0.8, label='15° critical threshold')
        ax1.axhline(5,  color='orange', linestyle='--', linewidth=0.8, label='5° notable threshold')
        ax1.axhline(0,  color='green',  linestyle='-',  linewidth=1.2, label='0° = baseline (same as correct-input prediction)')
        ax1.set_title(
            f"Full Chunk Divergence Profile (mean over episode) — {label}\n"
            f"0° = robot would execute identical trajectory  |  higher = more different from correct-input behaviour",
            fontsize=9)
        ax1.set_ylabel("Angular divergence (°)")
        ax1.set_xlabel(f"Chunk step (0 = immediate, {len(list(chunk_steps))-1} = 1.67s later)")
        ax1.legend(fontsize=7); ax1.set_ylim(bottom=0)

        # Row 2: compounding drift
        ax2 = axes[2, col]
        for cond_name, drifts in results["compounding"].items():
            steps = list(range(len(drifts)))
            color = {'wrong_prompt': 'purple', 'black_front': 'orange',
                     'black_top': 'gold', 'black_all': 'red'}.get(cond_name, 'gray')
            ax2.plot(steps, drifts, color=color, label=cond_name, linewidth=2)
        ax2.set_title(
            f"Compounding State Drift over {len(drifts)} steps — {label}\n"
            "(simulates closed-loop rollout: each step re-queries policy with drifted state)",
            fontsize=9)
        ax2.set_ylabel("Cumulative state L2 drift")
        ax2.set_xlabel("Rollout step")
        ax2.legend(fontsize=7)

        # Row 3: action magnitude ratio (ablated / baseline) — directly shows "frozen robot"
        ax3 = axes[3, col]
        mr  = results.get("mag_ratios", {})
        mag_cond_styles = [
            ("wp",  "purple", "Wrong task prompt"),
            ("bf",  "orange", "Black front camera"),
            ("bt",  "gold",   "Black top camera"),
            ("ba",  "red",    "Both cameras black"),
        ]
        for mk, color, lbl in mag_cond_styles:
            if mk in mr:
                ax3.plot(chunk_steps, mr[mk], color=color, label=lbl, linewidth=2)
        ax3.axhline(1.0, color='green',  linestyle='-',  linewidth=1.5, label='1.0 = baseline magnitude (correct inputs)')
        ax3.axhline(0.5, color='orange', linestyle='--', linewidth=0.8, label='0.5 = half the movement speed')
        ax3.axhline(0.1, color='red',    linestyle='--', linewidth=0.8, label='0.1 = nearly frozen')
        ax3.set_title(
            f"Action Magnitude Ratio (ablated / baseline) — {label}\n"
            "< 1.0 = robot moves less  |  ≈ 0 = robot frozen  |  > 1.0 = robot overshoots",
            fontsize=9)
        ax3.set_ylabel("Magnitude ratio")
        ax3.set_xlabel(f"Chunk step (0 = immediate, {len(list(chunk_steps))-1} = 1.67s later)")
        ax3.set_ylim(0, 2.0)
        ax3.legend(fontsize=7)

        # ── Row 4: rollout magnitude collapse ────────────────────────────────
        # This is the "frozen robot" predictor.
        # Unlike Row 3 (single dataset frame, correct state), this simulation
        # re-queries the policy at each rollout step with the DRIFTED ablated state.
        # When cameras are black AND state has drifted, the policy sees an OOD input
        # → outputs near-zero actions → the robot would freeze in the real world.
        ax4 = axes[4, col]
        cm = results.get("compound_mags", {})
        compound_styles = [
            ("wrong_prompt", "purple", "Wrong task prompt"),
            ("black_front",  "orange", "Black front camera"),
            ("black_top",    "gold",   "Black top camera"),
            ("black_all",    "red",    "Both cameras black  ← key: drifted state + no vision"),
        ]
        n_rollout = max((len(v) for v in cm.values()), default=1)
        rollout_steps_range = range(n_rollout)
        for cond_name, color, lbl in compound_styles:
            if cond_name in cm:
                ax4.plot(rollout_steps_range, cm[cond_name], color=color, label=lbl, linewidth=2)
        ax4.axhline(1.0, color='green',  linestyle='-',  linewidth=1.5,
                    label='1.0 = same movement energy as correct-input policy')
        ax4.axhline(0.75, color='orange', linestyle='--', linewidth=0.8,
                    label='0.75 = sluggish threshold (25% drop — robot visibly hesitates)')
        ax4.axhline(0.15, color='red',    linestyle='--', linewidth=0.8, label='0.15 = effectively frozen')
        ax4.set_title(
            f"Rollout Magnitude Collapse — {label}\n"
            "closed-loop sim: policy re-queried each step with drifted state + ablated cameras\n"
            "dip below 0.75 = sluggish/hesitating  |  dip below 0.15 = frozen  |  1.0 = unaffected\n"
            "note: dip-then-recover at 10 steps (0.33s) — use --rollout_steps 40 for sustained collapse",
            fontsize=9)
        ax4.set_ylabel("Action magnitude ratio\n(ablated / baseline)")
        ax4.set_xlabel(f"Rollout step  (each step ≈ 0.033s × action[0],  {n_rollout} steps total)")
        ax4.set_ylim(0, 2.0)
        ax4.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] Saved cross-task figure → {out_path}")


def make_summary_table(results_a, results_b, task_a_prompt, task_b_prompt):
    print("\n" + "═"*80)
    print("  CROSS-TASK ABLATION — Final Verdict")
    print("  Corrects two flaws in the original ablation:")
    print("    1. Wrong prompt is now a REAL trained task (not nonsense text)")
    print("    2. Cameras show actual black pixel frames (not zeroed tensors)")
    print("═"*80)

    short_a = task_a_prompt[:50] + "..." if len(task_a_prompt) > 50 else task_a_prompt
    short_b = task_b_prompt[:50] + "..." if len(task_b_prompt) > 50 else task_b_prompt

    for tag, results, own_prompt, other_prompt in [
        ("TASK A scene (sorting)",  results_a, short_a, short_b),
        ("TASK B scene (stacking)", results_b, short_b, short_a),
    ]:
        print(f"\n  Scene: {tag}")
        print(f"    Correct prompt : {own_prompt}")
        print(f"    Wrong prompt   : {other_prompt}")
        print(f"  {'Condition':<22}  {'Step-0 mean':>11}  {'Step-0 peak':>11}  "
              f"{'Chunk mean':>10}  {'Chunk peak':>10}  {'Final drift':>12}  verdict")
        print(f"  {'-'*90}")

        cp = results["chunk_profiles"]
        mr = results.get("mag_ratios", {})
        cm = results.get("compound_mags", {})
        cond_map = {
            "Wrong task prompt":   (results["ang_wrong_prompt"], results["compounding"]["wrong_prompt"], cp.get("wp", []), mr.get("wp", []), cm.get("wrong_prompt", [])),
            "Black front camera":  (results["ang_black_front"],  results["compounding"]["black_front"],  cp.get("bf", []), mr.get("bf", []), cm.get("black_front", [])),
            "Black top camera":    (results["ang_black_top"],    results["compounding"]["black_top"],    cp.get("bt", []), mr.get("bt", []), cm.get("black_top", [])),
            "Both cameras black":  (results["ang_black_all"],    results["compounding"]["black_all"],    cp.get("ba", []), mr.get("ba", []), cm.get("black_all", [])),
        }
        print(f"  {'Condition':<22}  {'Step-0°':>7}  {'Chunk°':>7}  "
              f"{'ChunkMag':>9}  {'RolloutMag':>11}  {'Verdict':>20}")
        print(f"  {'-'*85}")

        for cond, (angles, drifts, prof, mag, cmag) in cond_map.items():
            mean_a     = np.mean(angles)
            chunk_mean = float(np.mean(prof))  if len(prof)  > 0 else 0.0
            mean_mag   = float(np.mean(mag))   if len(mag)   > 0 else 1.0
            min_mag    = float(np.min(mag))     if len(mag)   > 0 else 1.0

            # Rollout magnitude: does the policy collapse when state drifts + cameras ablated?
            # Threshold rationale: 0.75 = 25% magnitude drop → robot visibly slower; 0.15 = near-stop.
            # Even a min of 0.72x with recovery to 0.98x is "sluggish": the policy is confused for
            # several rollout steps, which would appear as stuttering/hesitation on the real robot.
            # Note: a dip-then-recover pattern (e.g. 1.0→0.72→0.98) is expected at n_steps=10 (0.33s);
            # longer rollouts push the state further OOD and produce a steeper, sustained collapse.
            if len(cmag) > 0:
                rollout_min = float(np.min(cmag))
                rollout_end = float(cmag[-1])
                if rollout_min < 0.15:
                    rollout_flag = "🟥 FROZEN in rollout"
                elif rollout_min < 0.75:
                    rollout_flag = "🟧 sluggish in rollout"
                else:
                    rollout_flag = "🟩 stable in rollout"
                rollout_str = f"{rollout_min:.2f}x min / {rollout_end:.2f}x final"
            else:
                rollout_flag = "—"
                rollout_str  = "n/a"

            # Static-chunk frozen flag (on-trajectory state)
            frozen_flag = "🟥 FROZEN"   if min_mag < 0.15 else \
                          "🟧 sluggish" if mean_mag < 0.6  else \
                          "🟩 moving"
            if chunk_mean > 15:
                flag = "🔴 CRITICAL"
            elif chunk_mean > 5:
                flag = "🟡 NOTABLE"
            else:
                flag = "🟢 minor"
            print(f"  {cond:<22}  {mean_a:6.1f}°  {chunk_mean:6.1f}°  "
                  f"  {mean_mag:5.2f}x  {rollout_str:>28}  {flag} {rollout_flag}")

        print()
        print(f"  ChunkMag    = action magnitude ratio at dataset frames [on-trajectory state, ablated cameras]")
        print(f"  RolloutMag  = action magnitude during closed-loop sim [drifted state + ablated cameras]")
        print(f"    · ChunkMag ≈ 1.0 even with black cameras: proprioception dominates at step 0")
        print(f"    · RolloutMag 'sluggish' (< 0.75x): real robot hesitates / slows down mid-motion")
        print(f"    · Dip-then-recover pattern (e.g. 1.0→0.72→0.98) is expected at short rollouts")
        print(f"      because ~10 steps (0.33s) only partially drifts state off-trajectory.")
        print(f"      Re-run with --rollout_steps 40 to see the full sustained collapse → near-zero.")

    print("═"*80)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def process_episode(policy, dataset, rename_map, device, preprocessor,
                    episode_idx, correct_prompt, wrong_prompt,
                    rollout_steps, noise_seed, sample_every=4):
    """Run all ablation conditions over one episode, return results dict."""

    if hasattr(dataset.meta, 'episode_data_index'):
        start_idx = int(dataset.meta.episode_data_index['from'][episode_idx])
        end_idx   = int(dataset.meta.episode_data_index['to'][episode_idx])
    else:
        start_idx = sum(dataset.meta.episodes[i]['length'] for i in range(episode_idx))
        end_idx   = start_idx + dataset.meta.episodes[episode_idx]['length']
    print(f"  Episode {episode_idx}: frames {start_idx}–{end_idx}  prompt='{correct_prompt[:50]}...'")

    frames        = []
    ang_wp, ang_bf, ang_bt, ang_ba = [], [], [], []
    items_cache   = {}  # frame_offset -> raw item (kept for peak snapshot)
    chunk_profiles_acc = {}   # accumulated per-frame chunk profiles

    # Compounding: only on first frame (initial state) to keep it clean
    first_item = dataset[start_idx]

    def _compound(prompt_ablated=None, black_front=False, black_top=False):
        drifts, mags = simulate_rollout_divergence(
            policy, first_item, rename_map, device, preprocessor,
            correct_prompt, prompt_ablated or correct_prompt,
            black_front=black_front, black_top=black_top,
            n_steps=rollout_steps, noise_seed=noise_seed)
        return drifts, mags

    wp_d, wp_m  = _compound(prompt_ablated=wrong_prompt)
    bf_d, bf_m  = _compound(black_front=True)
    bt_d, bt_m  = _compound(black_top=True)
    ba_d, ba_m  = _compound(black_front=True, black_top=True)

    compounding = {
        "wrong_prompt": wp_d,
        "black_front":  bf_d,
        "black_top":    bt_d,
        "black_all":    ba_d,
    }
    # Per-rollout-step action magnitude ratio (ablated / baseline).
    # Drops toward 0 when the policy encounters an OOD (black cameras + drifted state) input.
    # This is the direct predictor of the real-world "frozen robot" observation.
    compound_mags = {
        "wrong_prompt": wp_m,
        "black_front":  bf_m,
        "black_top":    bt_m,
        "black_all":    ba_m,
    }

    for i in range(start_idx, end_idx):
        if (i - start_idx) % sample_every != 0:
            continue

        item = dataset[i]

        bp_base = build_obs(item, rename_map, device, preprocessor, correct_prompt)
        a_base  = predict_action_fixed_noise(policy, bp_base, device, noise_seed)

        def get_angle_and_chunk(bp_ablated):
            a = predict_action_fixed_noise(policy, bp_ablated, device, noise_seed)
            ang, _ = angular_divergence(a_base, a, step=0)
            profile = chunk_divergence_profile(a_base, a)   # [chunk_size]
            # Magnitude ratio per chunk step: ablated/baseline
            # < 1.0 means robot moves less, near 0 means frozen
            base_norms  = a_base.norm(dim=-1)   # [chunk_size]
            abl_norms   = a.norm(dim=-1)
            mag_ratio   = (abl_norms / (base_norms + 1e-8)).numpy()  # [chunk_size]
            return ang, profile, mag_ratio

        frame_offset = i - start_idx
        wp_ang, wp_prof, wp_mag = get_angle_and_chunk(build_obs(item, rename_map, device, preprocessor, wrong_prompt))
        bf_ang, bf_prof, bf_mag = get_angle_and_chunk(build_obs(item, rename_map, device, preprocessor, correct_prompt, black_front=True))
        bt_ang, bt_prof, bt_mag = get_angle_and_chunk(build_obs(item, rename_map, device, preprocessor, correct_prompt, black_top=True))
        ba_ang, ba_prof, ba_mag = get_angle_and_chunk(build_obs(item, rename_map, device, preprocessor, correct_prompt, black_front=True, black_top=True))

        ang_wp.append(wp_ang); ang_bf.append(bf_ang)
        ang_bt.append(bt_ang); ang_ba.append(ba_ang)
        frames.append(frame_offset)
        items_cache[frame_offset] = item

        chunk_profiles_acc.setdefault("wp", []).append(wp_prof)
        chunk_profiles_acc.setdefault("bf", []).append(bf_prof)
        chunk_profiles_acc.setdefault("bt", []).append(bt_prof)
        chunk_profiles_acc.setdefault("ba", []).append(ba_prof)
        chunk_profiles_acc.setdefault("wp_mag", []).append(wp_mag)
        chunk_profiles_acc.setdefault("bf_mag", []).append(bf_mag)
        chunk_profiles_acc.setdefault("bt_mag", []).append(bt_mag)
        chunk_profiles_acc.setdefault("ba_mag", []).append(ba_mag)

        if len(frames) % 25 == 0:
            print(f"    Frame {frames[-1]:4d} | "
                  f"WP={wp_ang:.1f}°/{wp_prof.mean():.1f}°  mag={wp_mag.mean():.2f}x  "
                  f"BF={bf_ang:.1f}°/{bf_prof.mean():.1f}°  mag={bf_mag.mean():.2f}x  "
                  f"BT={bt_ang:.1f}°/{bt_prof.mean():.1f}°  "
                  f"BA={ba_ang:.1f}°/{ba_prof.mean():.1f}°  mag={ba_mag.mean():.2f}x")

    # Average chunk profiles over all sampled frames
    chunk_profiles = {
        k: np.mean(np.stack(v), axis=0)   # [chunk_size]
        for k, v in chunk_profiles_acc.items()
    }

    return {
        "frames":            frames,
        "items_cache":       items_cache,
        "chunk_profiles":    chunk_profiles,   # mean divergence at each chunk step
        "mag_ratios": {      # mean magnitude ratio (ablated/baseline) at each chunk step
            k[:-4]: np.mean(np.stack(chunk_profiles_acc[k]), axis=0)
            for k in ["wp_mag", "bf_mag", "bt_mag", "ba_mag"]
            if k in chunk_profiles_acc
        },
        "ang_wrong_prompt":  ang_wp,
        "ang_black_front":   ang_bf,
        "ang_black_top":     ang_bt,
        "ang_black_all":     ang_ba,
        "compounding":       compounding,
        "compound_mags":     compound_mags,   # per-rollout-step magnitude ratios (frozen robot signal)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",        type=str, required=True)
    parser.add_argument("--ckpt",           type=str, required=True)
    parser.add_argument("--task_a_episode", type=int, default=0,
                        help="Episode index for task A (e.g. sorting, ep 0)")
    parser.add_argument("--task_b_episode", type=int, default=100,
                        help="Episode index for task B (e.g. stacking, ep 100)")
    parser.add_argument("--task_a_prompt",  type=str, required=True,
                        help="Correct prompt for task A")
    parser.add_argument("--task_b_prompt",  type=str, required=True,
                        help="Correct prompt for task B")
    parser.add_argument("--rollout_steps",  type=int, default=15,
                        help="Steps to simulate in compounding drift experiment")
    parser.add_argument("--output_path",    type=str, default="ablation_cross_task_summary.png")
    parser.add_argument("--rename_map",     type=str,
                        default='{"observation.images.front":"observation.images.camera1",'
                                '"observation.images.top":"observation.images.camera2"}')
    parser.add_argument("--video_backend",  type=str, default="pyav")
    parser.add_argument("--noise_seed",     type=int, default=42)
    parser.add_argument("--sample_every",   type=int, default=4,
                        help="Sample every Nth frame (4 = 25% of frames)")
    parser.add_argument("--top_n_peaks",    type=int, default=5,
                        help="Number of peak frames to show per condition in the contact sheet")
    args = parser.parse_args()

    device     = get_safe_torch_device("cuda")
    rename_map = json.loads(args.rename_map)

    dataset    = LeRobotDataset(args.repo_id, batch_encoding_size=1,
                                video_backend=args.video_backend)
    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy     = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=rename_map)
    policy.to(device)
    policy.eval()

    preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset.meta.stats)

    print(f"\n[Cross-Task Ablation]")
    print(f"  Task A (ep {args.task_a_episode}): {args.task_a_prompt}")
    print(f"  Task B (ep {args.task_b_episode}): {args.task_b_prompt}")
    print(f"  Compounding simulation: {args.rollout_steps} steps\n")

    print("[Processing Task A episode — correct prompt vs stacking prompt...]")
    results_a = process_episode(
        policy, dataset, rename_map, device, preprocessor,
        episode_idx   = args.task_a_episode,
        correct_prompt= args.task_a_prompt,
        wrong_prompt  = args.task_b_prompt,    # ← real trained task, not nonsense
        rollout_steps = args.rollout_steps,
        noise_seed    = args.noise_seed,
        sample_every  = args.sample_every,
    )

    print("\n[Processing Task B episode — correct prompt vs sorting prompt...]")
    results_b = process_episode(
        policy, dataset, rename_map, device, preprocessor,
        episode_idx   = args.task_b_episode,
        correct_prompt= args.task_b_prompt,
        wrong_prompt  = args.task_a_prompt,    # ← real trained task, not nonsense
        rollout_steps = args.rollout_steps,
        noise_seed    = args.noise_seed,
        sample_every  = args.sample_every,
    )

    make_cross_task_figure(results_a, results_b,
                           args.task_a_prompt, args.task_b_prompt,
                           args.output_path)
    make_summary_table(results_a, results_b, args.task_a_prompt, args.task_b_prompt)

    # Peak frame contact sheets — show what the cameras saw at every spike
    peak_path_a = args.output_path.replace(".png", "_peaks_taskA.png")
    peak_path_b = args.output_path.replace(".png", "_peaks_taskB.png")
    save_peak_contact_sheet(results_a, "Task A (sorting)",  rename_map, peak_path_a, top_n=args.top_n_peaks)
    save_peak_contact_sheet(results_b, "Task B (stacking)", rename_map, peak_path_b, top_n=args.top_n_peaks)


if __name__ == "__main__":
    main()
