#!/usr/bin/env python3
"""
analyze_sequence_replay_rest_to_grasp_predict_lift.py

Sequence-aware offline replay test.

Goal:
  Test whether the same saved rest->reach->grasp observation history produces
  the same lift-stage predicted action when the policy is reset once at the
  beginning of each replay, then fed the observation sequence without resetting
  between intermediate observations.

This is stronger than a single-snapshot determinism test because it includes
policy/preprocessor/action-queue state evolution across the saved sequence.

Typical use:
  python src/lerobot/scripts/analyze_sequence_replay_rest_to_grasp_predict_lift.py \
    --high debug_runs/20260529_112158_high \
    --policy-path ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
    --dataset-repo-id ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
    --xml src/lerobot/scripts/follower.xml \
    --out analysis_sequence_replay_high_lift \
    --intervention-name high_transport \
    --alpha 6.0 \
    --task "Put the red cube in the box." \
    --sequence-chunks 0,1,2 \
    --decision-index -1 \
    --repeat-exact-sequence 100 \
    --repeat-per-episode-sequence 10 \
    --focus-dims 1,2,5 \
    --rename-map-json '{"observation.images.front":"observation.images.camera1","observation.images.top":"observation.images.camera2","observation.images.wrist":"observation.images.camera3"}'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# This script should live in the same directory as:
# analyze_high_low_baseline_replay_reach_to_predict_lift.py
# It reuses the validated policy-loading, steering, observation loading, and plotting helpers.
try:
    from analyze_high_low_baseline_replay_reach_to_predict_lift import (
        apply_activation_steering,
        collect_displayable_observation_images,
        compute_eef_height_from_state,
        extract_observation_frame,
        find_chunk_file,
        flatten_numeric,
        get_state_vec_from_obs_or_data,
        get_task,
        list_episode_indices,
        load_debug_pt,
        load_fk_model,
        load_policy_context,
        natural_sort_key,
        observation_tensors_to_numpy,
        parse_episode_list,
        reset_policy_processors,
        safe_mkdir,
        sanitize_filename,
        set_global_seed,
    )
except Exception as exc:
    raise ImportError(
        "Could not import helper functions from "
        "analyze_high_low_baseline_replay_reach_to_predict_lift.py. "
        "Place this script in src/lerobot/scripts/ next to that file, and make sure "
        "your fixed version of that helper script is present."
    ) from exc

from lerobot.utils.control_utils import predict_action


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError(f"Empty integer list: {text!r}")
    return out


def resolve_decision_index(decision_index: int, n: int) -> int:
    idx = decision_index
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        raise IndexError(f"decision_index={decision_index} resolved to {idx}, but sequence length is {n}")
    return idx


def load_episode_sequence(
    high_root: Path,
    episode_idx: int,
    sequence_chunks: list[int],
    fallback_task: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """
    Load one episode's saved observation_frame.pt files for the requested chunk sequence.

    Returns:
      observations: list of extracted observation frames
      metadata_rows: list of per-step metadata dicts
      task: task string from the first available chunk, fallback otherwise
    """
    observations: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    task = fallback_task

    for seq_pos, chunk_idx in enumerate(sequence_chunks):
        p = find_chunk_file(high_root, episode_idx, chunk_idx)
        if p is None:
            raise FileNotFoundError(
                f"Missing chunk {chunk_idx} for episode {episode_idx}: "
                f"{high_root / f'episode_{episode_idx:06d}'}"
            )

        data = load_debug_pt(p)
        obs = extract_observation_frame(data)
        if seq_pos == 0:
            task = get_task(data, fallback_task)

        rows.append(
            {
                "episode_idx": int(episode_idx),
                "seq_pos": int(seq_pos),
                "chunk_idx": int(chunk_idx),
                "source_path": str(p),
            }
        )
        observations.append(obs)

    return observations, rows, task


def predict_sequence_once(
    ctx,
    observations: list[dict[str, Any]],
    task: str,
    trial_idx: int,
) -> list[dict[str, float]]:
    """
    Reset policy/preprocessors once at sequence start, then feed all observations
    without resetting between intermediate observations.

    This is the key difference from single-snapshot replay.
    """
    if ctx.reset_seed_each_trial:
        set_global_seed(ctx.seed + trial_idx)

    reset_policy_processors(ctx.policy, ctx.preprocessor, ctx.postprocessor)

    apply_activation_steering(
        policy=ctx.policy,
        intervention_name=ctx.intervention_name,
        alpha=ctx.alpha,
        enable_steering=ctx.enable_steering,
    )

    step_actions: list[dict[str, float]] = []

    for obs in observations:
        obs_np = observation_tensors_to_numpy(obs)

        with torch.no_grad():
            action_values = predict_action(
                observation=obs_np,
                policy=ctx.policy,
                device=ctx.device,
                preprocessor=ctx.preprocessor,
                postprocessor=ctx.postprocessor,
                use_amp=ctx.use_amp,
                task=task,
                robot_type=ctx.robot_type,
            )

        step_actions.append(flatten_numeric(action_values, "action_values"))

    return step_actions


def replay_exact_sequence_repeated(
    ctx,
    observations: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    task: str,
    repeat: int,
    decision_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeat the exact same observation sequence many times.
    Saves:
      all_steps_df: all actions at every sequence step for every repeat
      final_df: the action at the decision step for every repeat
    """
    decision_pos = resolve_decision_index(decision_index, len(observations))

    all_rows = []
    final_rows = []

    for trial in range(repeat):
        actions_by_step = predict_sequence_once(ctx, observations, task, trial_idx=trial)

        for seq_pos, action_dict in enumerate(actions_by_step):
            row = {
                "trial": int(trial),
                "is_decision_step": bool(seq_pos == decision_pos),
            }
            row.update(sequence_rows[seq_pos])
            row.update(action_dict)
            all_rows.append(row)

            if seq_pos == decision_pos:
                final_rows.append(row.copy())

    return pd.DataFrame(all_rows), pd.DataFrame(final_rows)


def summarize_repeat_trials(df: pd.DataFrame, action_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in action_cols:
        vals = df[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "action_key": col,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "range": float(np.max(vals) - np.min(vals)),
                "max_abs_deviation_from_mean": float(np.max(np.abs(vals - np.mean(vals)))),
            }
        )
    return pd.DataFrame(rows).sort_values(["std", "range"], ascending=False)



def get_action_vector_from_row(row: pd.Series, max_dims: int = 6) -> np.ndarray | None:
    vals = []
    for i in range(max_dims):
        key = f"action_values.{i}"
        if key not in row.index:
            return None
        v = row[key]
        if pd.isna(v):
            return None
        vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def add_predicted_action_fk_eef_z(
    df: pd.DataFrame,
    mj_model,
    mj_data,
    action_dim_count: int = 6,
    out_col: str = "predicted_action_eef_z",
) -> pd.DataFrame:
    """
    Convert predicted joint-position actions to commanded EEF z using FK.

    Important:
      This estimates the EEF height implied by the predicted joint target.
      It is not the measured real robot EEF height after physics/contact/motor execution.
    """
    if df.empty:
        return df

    needed = [f"action_values.{i}" for i in range(action_dim_count)]
    if any(c not in df.columns for c in needed):
        warnings.warn(
            f"Cannot compute {out_col}: missing one or more columns {needed}. "
            "This script expects action_values.0..5 to be joint-position targets."
        )
        df[out_col] = np.nan
        return df

    vals = []
    for _, row in df.iterrows():
        q_like = get_action_vector_from_row(row, max_dims=action_dim_count)
        if q_like is None:
            vals.append(float("nan"))
            continue
        try:
            vals.append(float(compute_eef_height_from_state(q_like, mj_model, mj_data)))
        except Exception:
            vals.append(float("nan"))
    df[out_col] = vals
    return df


def summarize_scalar_repeat(df: pd.DataFrame, col: str) -> dict[str, float]:
    if df.empty or col not in df.columns:
        return {
            f"{col}_mean": float("nan"),
            f"{col}_std": float("nan"),
            f"{col}_min": float("nan"),
            f"{col}_max": float("nan"),
            f"{col}_range": float("nan"),
        }
    vals = df[col].to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            f"{col}_mean": float("nan"),
            f"{col}_std": float("nan"),
            f"{col}_min": float("nan"),
            f"{col}_max": float("nan"),
            f"{col}_range": float("nan"),
        }
    return {
        f"{col}_mean": float(np.mean(vals)),
        f"{col}_std": float(np.std(vals, ddof=0)),
        f"{col}_min": float(np.min(vals)),
        f"{col}_max": float(np.max(vals)),
        f"{col}_range": float(np.max(vals) - np.min(vals)),
    }


def load_lift_eef_for_episode(
    high_root: Path,
    episode_idx: int,
    lift_chunk: int,
    mj_model,
    mj_data,
) -> float:
    p = find_chunk_file(high_root, episode_idx, lift_chunk)
    if p is None:
        return float("nan")
    data = load_debug_pt(p)
    obs = extract_observation_frame(data)
    try:
        state = get_state_vec_from_obs_or_data(obs, data)
        return float(compute_eef_height_from_state(state, mj_model, mj_data))
    except Exception:
        return float("nan")


def replay_all_episode_sequences(
    ctx,
    high_root: Path,
    sequence_chunks: list[int],
    decision_index: int,
    repeat_per_episode: int,
    fallback_task: str,
    lift_chunk: int,
    mj_model,
    mj_data,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Repeat each episode's rest->grasp sequence and summarize the decision-step action.
    Useful for the later diverse-sequence experiment.
    """
    trial_dfs = []
    mean_rows = []
    missing_rows = []

    decision_pos = resolve_decision_index(decision_index, len(sequence_chunks))

    for ep_idx in list_episode_indices(high_root):
        try:
            observations, seq_rows, task = load_episode_sequence(
                high_root=high_root,
                episode_idx=ep_idx,
                sequence_chunks=sequence_chunks,
                fallback_task=fallback_task,
            )
        except Exception as exc:
            missing_rows.append({"episode_idx": ep_idx, "error": str(exc)})
            continue

        all_steps, final_trials = replay_exact_sequence_repeated(
            ctx=ctx,
            observations=observations,
            sequence_rows=seq_rows,
            task=task,
            repeat=repeat_per_episode,
            decision_index=decision_index,
        )
        final_trials["task"] = task
        final_trials["lift_eef_z"] = load_lift_eef_for_episode(
            high_root=high_root,
            episode_idx=ep_idx,
            lift_chunk=lift_chunk,
            mj_model=mj_model,
            mj_data=mj_data,
        )
        final_trials = add_predicted_action_fk_eef_z(final_trials, mj_model, mj_data)
        trial_dfs.append(final_trials)

        action_cols = sorted([c for c in final_trials.columns if c.startswith("action_values.")], key=natural_sort_key)
        mean_row: dict[str, Any] = {
            "episode_idx": int(ep_idx),
            "decision_seq_pos": int(decision_pos),
            "decision_chunk_idx": int(sequence_chunks[decision_pos]),
            "num_repeats": int(len(final_trials)),
            "lift_eef_z": float(final_trials["lift_eef_z"].iloc[0]),
            "task": task,
        }
        for c in action_cols:
            vals = final_trials[c].to_numpy(dtype=float)
            mean_row[c] = float(np.mean(vals))
            mean_row[f"{c}_repeat_std"] = float(np.std(vals, ddof=0))
            mean_row[f"{c}_repeat_range"] = float(np.max(vals) - np.min(vals))

        if "predicted_action_eef_z" in final_trials.columns:
            zvals = final_trials["predicted_action_eef_z"].to_numpy(dtype=float)
            zvals = zvals[np.isfinite(zvals)]
            if zvals.size:
                mean_row["predicted_action_eef_z"] = float(np.mean(zvals))
                mean_row["predicted_action_eef_z_repeat_std"] = float(np.std(zvals, ddof=0))
                mean_row["predicted_action_eef_z_repeat_range"] = float(np.max(zvals) - np.min(zvals))
            else:
                mean_row["predicted_action_eef_z"] = float("nan")
                mean_row["predicted_action_eef_z_repeat_std"] = float("nan")
                mean_row["predicted_action_eef_z_repeat_range"] = float("nan")

        mean_rows.append(mean_row)

    all_trials = pd.concat(trial_dfs, ignore_index=True) if trial_dfs else pd.DataFrame()
    mean_df = pd.DataFrame(mean_rows).sort_values("episode_idx") if mean_rows else pd.DataFrame()
    missing_df = pd.DataFrame(missing_rows)
    return all_trials, mean_df, missing_df


def save_sequence_observation_montage(
    plot_dir: Path,
    observations: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
) -> None:
    """
    Save a grid montage: rows=sequence positions/chunks, columns=cameras.
    """
    safe_mkdir(plot_dir)

    per_step = []
    camera_keys = []

    for obs in observations:
        imgs = collect_displayable_observation_images(obs)
        per_step.append(imgs)
        for k, _ in imgs:
            if k not in camera_keys:
                camera_keys.append(k)

    if not camera_keys:
        warnings.warn("No displayable images found for sequence montage.")
        return

    n_rows = len(observations)
    n_cols = len(camera_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.asarray([[axes]])
    elif n_rows == 1:
        axes = np.asarray([axes])
    elif n_cols == 1:
        axes = np.asarray([[ax] for ax in axes])

    for r in range(n_rows):
        img_by_key = {k: img for k, img in per_step[r]}
        for c, key in enumerate(camera_keys):
            ax = axes[r, c]
            if key in img_by_key:
                ax.imshow(img_by_key[key])
            ax.axis("off")
            if r == 0:
                ax.set_title(key, fontsize=9)
            if c == 0:
                ax.set_ylabel(
                    f"seq={sequence_rows[r]['seq_pos']} | chunk={sequence_rows[r]['chunk_idx']}",
                    fontsize=9,
                )

    fig.suptitle("Exact rest-to-grasp observation sequence used for repeated replay")
    fig.tight_layout()
    fig.savefig(plot_dir / "sequence_exact_observation_montage.png", dpi=160)
    plt.close(fig)

    # Save individual images too.
    for r, imgs in enumerate(per_step):
        chunk_idx = sequence_rows[r]["chunk_idx"]
        for key, img in imgs:
            out = plot_dir / f"sequence_obs_seq{r:02d}_chunk{chunk_idx:03d}_{sanitize_filename(key)}.png"
            plt.imsave(out, img)


def save_final_action_repeat_plot(
    plot_dir: Path,
    final_trials: pd.DataFrame,
    action_cols: list[str],
    focus_dims: list[int],
) -> None:
    safe_mkdir(plot_dir)
    cols = list(action_cols)
    if focus_dims:
        focus_cols = [f"action_values.{d}" for d in focus_dims if f"action_values.{d}" in cols]
        if focus_cols:
            cols = focus_cols

    if not cols:
        return

    plt.figure(figsize=(10, 5))
    x = final_trials["trial"].to_numpy(dtype=int)
    for c in cols:
        plt.plot(x, final_trials[c].to_numpy(dtype=float), marker="o", markersize=3, linewidth=1, label=c)
    plt.xlabel("repeat trial index")
    plt.ylabel("decision-step predicted action value")
    plt.title("Same observation sequence repeated: decision-step action")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "sequence_exact_decision_action_repeat_plot.png", dpi=160)
    plt.close()



ACTION_DIM_LABELS = {
    0: "shoulder_pan.pos",
    1: "shoulder_lift.pos",
    2: "elbow_flex.pos",
    3: "wrist_flex.pos",
    4: "wrist_roll.pos",
    5: "gripper.pos",
}


def action_col_label(col: str) -> str:
    try:
        dim = int(str(col).split(".")[-1])
        return f"{col} ({ACTION_DIM_LABELS.get(dim, f'dim_{dim}')})"
    except Exception:
        return str(col)


def save_all_action_repeat_plot(
    plot_dir: Path,
    final_trials: pd.DataFrame,
    action_cols: list[str],
) -> None:
    """
    Save one plot containing all predicted actuator/joint target dimensions
    across repeated exact-sequence replays.

    This is the most direct visual check that all 6 predicted command dimensions
    are unchanged across the 100 repeated trials.
    """
    if final_trials.empty or not action_cols:
        return
    safe_mkdir(plot_dir)

    plt.figure(figsize=(12, 6))
    x = final_trials["trial"].to_numpy(dtype=int)
    for c in action_cols:
        plt.plot(
            x,
            final_trials[c].to_numpy(dtype=float),
            marker="o",
            markersize=2.5,
            linewidth=1,
            label=action_col_label(c),
        )

    plt.xlabel("repeat trial index")
    plt.ylabel("predicted actuator/joint target value")
    plt.title("Same observation sequence repeated: all predicted action dimensions")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "sequence_exact_decision_all_action_dims_repeat_plot.png", dpi=160)
    plt.close()

def save_sequence_step_plot(
    plot_dir: Path,
    exact_all_steps: pd.DataFrame,
    focus_dims: list[int],
) -> None:
    """
    Plot the predicted action across sequence steps for trial 0.
    This shows how the action changes as the saved rest->grasp history is fed.
    """
    safe_mkdir(plot_dir)
    trial0 = exact_all_steps[exact_all_steps["trial"] == 0].sort_values("seq_pos")
    if trial0.empty:
        return

    action_cols = sorted([c for c in trial0.columns if c.startswith("action_values.")], key=natural_sort_key)
    if focus_dims:
        focus_cols = [f"action_values.{d}" for d in focus_dims if f"action_values.{d}" in action_cols]
        if focus_cols:
            action_cols = focus_cols

    if not action_cols:
        return

    plt.figure(figsize=(8, 5))
    x = trial0["seq_pos"].to_numpy(dtype=int)
    labels = [f"{int(r.seq_pos)}\nchunk {int(r.chunk_idx)}" for r in trial0.itertuples(index=False)]
    for c in action_cols:
        plt.plot(x, trial0[c].to_numpy(dtype=float), marker="o", linewidth=1.5, label=c)
    plt.xticks(x, labels)
    plt.xlabel("sequence position")
    plt.ylabel("predicted action value")
    plt.title("Action values while feeding rest-to-grasp sequence")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "sequence_step_action_values_trial0.png", dpi=160)
    plt.close()



def save_predicted_fk_eef_repeat_plot(
    plot_dir: Path,
    final_trials: pd.DataFrame,
) -> None:
    if final_trials.empty or "predicted_action_eef_z" not in final_trials.columns:
        return
    safe_mkdir(plot_dir)

    plt.figure(figsize=(10, 5))
    plt.plot(
        final_trials["trial"].to_numpy(dtype=int),
        final_trials["predicted_action_eef_z"].to_numpy(dtype=float),
        marker="o",
        markersize=3,
        linewidth=1,
        label="FK(predicted action) EEF z",
    )
    plt.xlabel("repeat trial index")
    plt.ylabel("commanded EEF z from FK")
    plt.title("Same observation sequence repeated: FK EEF z from predicted action")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "sequence_exact_decision_predicted_fk_eef_z_repeat_plot.png", dpi=160)
    plt.close()


def save_sequence_step_predicted_fk_eef_plot(
    plot_dir: Path,
    exact_all_steps: pd.DataFrame,
) -> None:
    if exact_all_steps.empty or "predicted_action_eef_z" not in exact_all_steps.columns:
        return
    safe_mkdir(plot_dir)

    trial0 = exact_all_steps[exact_all_steps["trial"] == 0].sort_values("seq_pos")
    if trial0.empty:
        return

    x = trial0["seq_pos"].to_numpy(dtype=int)
    labels = [f"{int(r.seq_pos)}\nchunk {int(r.chunk_idx)}" for r in trial0.itertuples(index=False)]

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        trial0["predicted_action_eef_z"].to_numpy(dtype=float),
        marker="o",
        linewidth=1.5,
        label="FK(predicted action) EEF z",
    )
    plt.xticks(x, labels)
    plt.xlabel("sequence position")
    plt.ylabel("commanded EEF z from FK")
    plt.title("FK EEF z implied by predicted action while feeding sequence")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "sequence_step_predicted_fk_eef_z_trial0.png", dpi=160)
    plt.close()


def save_all_episode_sequence_plot(
    plot_dir: Path,
    mean_df: pd.DataFrame,
    focus_dims: list[int],
) -> None:
    if mean_df.empty:
        return
    safe_mkdir(plot_dir)

    for dim in focus_dims:
        col = f"action_values.{dim}"
        if col not in mean_df.columns or "lift_eef_z" not in mean_df.columns:
            continue

        plt.figure(figsize=(6, 5))
        plt.scatter(mean_df[col], mean_df["lift_eef_z"])
        for _, r in mean_df.iterrows():
            plt.text(r[col], r["lift_eef_z"], str(int(r["episode_idx"])), fontsize=8)
        plt.xlabel(f"Mean sequence decision action {col}")
        plt.ylabel("Actual lift EEF z")
        plt.title(f"Sequence decision {col} vs actual lift height")
        plt.tight_layout()
        plt.savefig(plot_dir / f"all_episode_sequence_{col.replace('.', '_')}_vs_lift_eef.png", dpi=160)
        plt.close()

    if "predicted_action_eef_z" in mean_df.columns and "lift_eef_z" in mean_df.columns:
        plt.figure(figsize=(6, 5))
        plt.scatter(mean_df["predicted_action_eef_z"], mean_df["lift_eef_z"])
        for _, r in mean_df.iterrows():
            plt.text(
                r["predicted_action_eef_z"],
                r["lift_eef_z"],
                str(int(r["episode_idx"])),
                fontsize=8,
            )
        plt.xlabel("Mean FK EEF z from sequence decision action")
        plt.ylabel("Actual lift EEF z")
        plt.title("Predicted-action FK EEF z vs actual lift height")
        plt.tight_layout()
        plt.savefig(plot_dir / "all_episode_sequence_predicted_fk_eef_z_vs_lift_eef.png", dpi=160)
        plt.close()



def drain_predicted_action_chunk_after_sequence(
    ctx,
    observations: list[dict[str, Any]],
    task: str,
    trial_idx: int,
    decision_index: int,
    chunk_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reset policy once, feed observations up to the decision observation without
    resetting between observations, then drain the policy action queue for
    chunk_steps actions using the same decision observation.

    This approximates the full predicted action chunk generated at the pre-lift
    decision point. It is most faithful when the policy uses an action queue and
    predict_action() pops from that queue until it is exhausted.

    Returns:
      sequence_df: action returned while feeding the sequence up to decision
      chunk_df: predicted/queued actions indexed by chunk_action_step
    """
    if chunk_steps <= 0:
        raise ValueError(f"chunk_steps must be positive, got {chunk_steps}")

    decision_pos = resolve_decision_index(decision_index, len(observations))

    if ctx.reset_seed_each_trial:
        set_global_seed(ctx.seed + trial_idx)

    reset_policy_processors(ctx.policy, ctx.preprocessor, ctx.postprocessor)

    apply_activation_steering(
        policy=ctx.policy,
        intervention_name=ctx.intervention_name,
        alpha=ctx.alpha,
        enable_steering=ctx.enable_steering,
    )

    sequence_rows = []
    decision_obs = observations[decision_pos]

    # Feed only up to the decision point. Do not feed later observations before
    # extracting the chunk, otherwise the tested decision is no longer pre-lift.
    for seq_pos in range(decision_pos + 1):
        obs_np = observation_tensors_to_numpy(observations[seq_pos])
        with torch.no_grad():
            action_values = predict_action(
                observation=obs_np,
                policy=ctx.policy,
                device=ctx.device,
                preprocessor=ctx.preprocessor,
                postprocessor=ctx.postprocessor,
                use_amp=ctx.use_amp,
                task=task,
                robot_type=ctx.robot_type,
            )

        row = {
            "trial": int(trial_idx),
            "seq_pos": int(seq_pos),
            "is_decision_step": bool(seq_pos == decision_pos),
        }
        row.update(flatten_numeric(action_values, "action_values"))
        sequence_rows.append(row)

    chunk_rows = []
    # chunk step 0 should be the action returned at the decision observation.
    decision_action = {
        k: v
        for k, v in sequence_rows[-1].items()
        if str(k).startswith("action_values.")
    }
    chunk_row0 = {
        "trial": int(trial_idx),
        "chunk_action_step": 0,
    }
    chunk_row0.update(decision_action)
    chunk_rows.append(chunk_row0)

    # Drain the remaining queued actions. We reuse the decision observation only
    # to satisfy predict_action()'s API. For action-queue policies, these calls
    # should mostly pop queued actions rather than regenerate from the observation.
    for chunk_step in range(1, chunk_steps):
        obs_np = observation_tensors_to_numpy(decision_obs)
        with torch.no_grad():
            action_values = predict_action(
                observation=obs_np,
                policy=ctx.policy,
                device=ctx.device,
                preprocessor=ctx.preprocessor,
                postprocessor=ctx.postprocessor,
                use_amp=ctx.use_amp,
                task=task,
                robot_type=ctx.robot_type,
            )

        row = {
            "trial": int(trial_idx),
            "chunk_action_step": int(chunk_step),
        }
        row.update(flatten_numeric(action_values, "action_values"))
        chunk_rows.append(row)

    return pd.DataFrame(sequence_rows), pd.DataFrame(chunk_rows)


def repeat_exact_sequence_predicted_chunk(
    ctx,
    observations: list[dict[str, Any]],
    task: str,
    repeat: int,
    decision_index: int,
    chunk_steps: int,
    mj_model,
    mj_data,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Repeat the exact same observation sequence and extract the predicted
    action chunk each time.

    Outputs:
      sequence_feed_trials: returned actions while feeding sequence to decision
      chunk_trials: 50-step predicted action chunk rows with FK EEF z
      chunk_summary: per-trial summary, including max FK EEF z over the chunk
    """
    seq_dfs = []
    chunk_dfs = []
    summary_rows = []

    for trial in range(repeat):
        seq_df, chunk_df = drain_predicted_action_chunk_after_sequence(
            ctx=ctx,
            observations=observations,
            task=task,
            trial_idx=trial,
            decision_index=decision_index,
            chunk_steps=chunk_steps,
        )
        seq_df = add_predicted_action_fk_eef_z(seq_df, mj_model, mj_data)
        chunk_df = add_predicted_action_fk_eef_z(chunk_df, mj_model, mj_data)

        seq_dfs.append(seq_df)
        chunk_dfs.append(chunk_df)

        row = {
            "trial": int(trial),
            "chunk_steps": int(len(chunk_df)),
        }

        z = chunk_df["predicted_action_eef_z"].to_numpy(dtype=float)
        finite_z = z[np.isfinite(z)]
        if finite_z.size:
            row["predicted_chunk_fk_eef_z_start"] = float(finite_z[0])
            row["predicted_chunk_fk_eef_z_end"] = float(finite_z[-1])
            row["predicted_chunk_fk_eef_z_max"] = float(np.max(finite_z))
            row["predicted_chunk_fk_eef_z_min"] = float(np.min(finite_z))
            row["predicted_chunk_fk_eef_z_range"] = float(np.max(finite_z) - np.min(finite_z))
            row["predicted_chunk_fk_eef_z_argmax_step"] = int(np.nanargmax(z))
        else:
            row["predicted_chunk_fk_eef_z_start"] = float("nan")
            row["predicted_chunk_fk_eef_z_end"] = float("nan")
            row["predicted_chunk_fk_eef_z_max"] = float("nan")
            row["predicted_chunk_fk_eef_z_min"] = float("nan")
            row["predicted_chunk_fk_eef_z_range"] = float("nan")
            row["predicted_chunk_fk_eef_z_argmax_step"] = -1

        action_cols = sorted([c for c in chunk_df.columns if c.startswith("action_values.")], key=natural_sort_key)
        for col in action_cols:
            vals = chunk_df[col].to_numpy(dtype=float)
            row[f"{col}_chunk_start"] = float(vals[0])
            row[f"{col}_chunk_end"] = float(vals[-1])
            row[f"{col}_chunk_min"] = float(np.min(vals))
            row[f"{col}_chunk_max"] = float(np.max(vals))
            row[f"{col}_chunk_range"] = float(np.max(vals) - np.min(vals))

        summary_rows.append(row)

    sequence_feed_trials = pd.concat(seq_dfs, ignore_index=True) if seq_dfs else pd.DataFrame()
    chunk_trials = pd.concat(chunk_dfs, ignore_index=True) if chunk_dfs else pd.DataFrame()
    chunk_summary = pd.DataFrame(summary_rows)

    return sequence_feed_trials, chunk_trials, chunk_summary


def summarize_chunk_repeat_consistency(chunk_trials: pd.DataFrame) -> pd.DataFrame:
    """
    For each action dimension and each chunk action step, summarize variation
    across repeated trials. If the same sequence always produces the same chunk,
    all ranges should be near zero.
    """
    if chunk_trials.empty:
        return pd.DataFrame()

    rows = []
    action_cols = sorted([c for c in chunk_trials.columns if c.startswith("action_values.")], key=natural_sort_key)

    for step, g in chunk_trials.groupby("chunk_action_step"):
        row = {"chunk_action_step": int(step)}
        for col in action_cols:
            vals = g[col].to_numpy(dtype=float)
            row[f"{col}_std"] = float(np.std(vals, ddof=0))
            row[f"{col}_range"] = float(np.max(vals) - np.min(vals))
        if "predicted_action_eef_z" in g.columns:
            z = g["predicted_action_eef_z"].to_numpy(dtype=float)
            z = z[np.isfinite(z)]
            if z.size:
                row["predicted_action_eef_z_std"] = float(np.std(z, ddof=0))
                row["predicted_action_eef_z_range"] = float(np.max(z) - np.min(z))
            else:
                row["predicted_action_eef_z_std"] = float("nan")
                row["predicted_action_eef_z_range"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_chunk_max_eef_repeat(chunk_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize repeat variation of the max FK EEF z over the predicted action chunk.
    """
    if chunk_summary.empty or "predicted_chunk_fk_eef_z_max" not in chunk_summary.columns:
        return pd.DataFrame()

    z = chunk_summary["predicted_chunk_fk_eef_z_max"].to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "predicted_chunk_fk_eef_z_max_mean": float(np.mean(z)),
                "predicted_chunk_fk_eef_z_max_std": float(np.std(z, ddof=0)),
                "predicted_chunk_fk_eef_z_max_min": float(np.min(z)),
                "predicted_chunk_fk_eef_z_max_max": float(np.max(z)),
                "predicted_chunk_fk_eef_z_max_range": float(np.max(z) - np.min(z)),
            }
        ]
    )


def save_predicted_chunk_plots(
    plot_dir: Path,
    chunk_trials: pd.DataFrame,
    chunk_summary: pd.DataFrame,
    focus_dims: list[int],
) -> None:
    if chunk_trials.empty:
        return
    safe_mkdir(plot_dir)

    action_cols = sorted([c for c in chunk_trials.columns if c.startswith("action_values.")], key=natural_sort_key)
    focus_cols = [f"action_values.{d}" for d in focus_dims if f"action_values.{d}" in action_cols]
    if not focus_cols:
        focus_cols = action_cols[: min(6, len(action_cols))]

    # Trial-0 action chunk trajectory for focus dims.
    trial0 = chunk_trials[chunk_trials["trial"] == 0].sort_values("chunk_action_step")
    if not trial0.empty:
        plt.figure(figsize=(12, 6))
        x = trial0["chunk_action_step"].to_numpy(dtype=int)
        for col in focus_cols:
            plt.plot(x, trial0[col].to_numpy(dtype=float), linewidth=1.5, label=action_col_label(col))
        plt.xlabel("predicted chunk action step")
        plt.ylabel("predicted actuator/joint target")
        plt.title("Predicted action chunk trajectory from one rest-to-grasp sequence")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / "sequence_exact_predicted_action_chunk_focus_dims_trial0.png", dpi=160)
        plt.close()

        if "predicted_action_eef_z" in trial0.columns:
            plt.figure(figsize=(12, 5))
            plt.plot(
                x,
                trial0["predicted_action_eef_z"].to_numpy(dtype=float),
                marker="o",
                markersize=3,
                linewidth=1.5,
                label="FK(predicted chunk action) EEF z",
            )
            plt.xlabel("predicted chunk action step")
            plt.ylabel("commanded EEF z from FK")
            plt.title("FK EEF z over predicted action chunk")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plot_dir / "sequence_exact_predicted_action_chunk_fk_eef_z_trial0.png", dpi=160)
            plt.close()

    # Repeat plot for max EEF z over predicted chunk.
    if not chunk_summary.empty and "predicted_chunk_fk_eef_z_max" in chunk_summary.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(
            chunk_summary["trial"].to_numpy(dtype=int),
            chunk_summary["predicted_chunk_fk_eef_z_max"].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            linewidth=1,
            label="max FK EEF z over predicted action chunk",
        )
        plt.xlabel("repeat trial index")
        plt.ylabel("max commanded EEF z from FK")
        plt.title("Same observation sequence repeated: max FK EEF z over predicted chunk")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / "sequence_exact_predicted_action_chunk_max_fk_eef_z_repeat_plot.png", dpi=160)
        plt.close()

    # Heatmap of action variation across trials for each chunk step and action dim.
    consistency = summarize_chunk_repeat_consistency(chunk_trials)
    if not consistency.empty:
        range_cols = [c for c in consistency.columns if c.endswith("_range") and c.startswith("action_values.")]
        if range_cols:
            mat = consistency[range_cols].to_numpy(dtype=float)
            plt.figure(figsize=(10, 6))
            plt.imshow(mat, aspect="auto")
            plt.colorbar(label="range across repeated trials")
            plt.yticks(
                np.arange(len(consistency)),
                consistency["chunk_action_step"].astype(int).tolist(),
                fontsize=7,
            )
            labels = [c.replace("_range", "") for c in range_cols]
            plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
            plt.xlabel("action dimension")
            plt.ylabel("predicted chunk action step")
            plt.title("Repeat variation of predicted action chunk")
            plt.tight_layout()
            plt.savefig(plot_dir / "sequence_exact_predicted_action_chunk_repeat_range_heatmap.png", dpi=160)
            plt.close()


def write_report(
    out_dir: Path,
    args,
    exact_summary: pd.DataFrame,
    exact_fk_summary: pd.DataFrame,
    exact_sequence_rows: list[dict[str, Any]],
    predicted_chunk_max_eef_summary: pd.DataFrame,
    all_episode_mean: pd.DataFrame,
    all_episode_missing: pd.DataFrame,
) -> None:
    max_range = float(exact_summary["range"].max()) if len(exact_summary) else float("nan")
    max_std = float(exact_summary["std"].max()) if len(exact_summary) else float("nan")
    fk_range = (
        float(exact_fk_summary["predicted_action_eef_z_range"].iloc[0])
        if len(exact_fk_summary) and "predicted_action_eef_z_range" in exact_fk_summary.columns
        else float("nan")
    )
    fk_std = (
        float(exact_fk_summary["predicted_action_eef_z_std"].iloc[0])
        if len(exact_fk_summary) and "predicted_action_eef_z_std" in exact_fk_summary.columns
        else float("nan")
    )
    chunk_max_fk_range = (
        float(predicted_chunk_max_eef_summary["predicted_chunk_fk_eef_z_max_range"].iloc[0])
        if len(predicted_chunk_max_eef_summary)
        and "predicted_chunk_fk_eef_z_max_range" in predicted_chunk_max_eef_summary.columns
        else float("nan")
    )
    chunk_max_fk_mean = (
        float(predicted_chunk_max_eef_summary["predicted_chunk_fk_eef_z_max_mean"].iloc[0])
        if len(predicted_chunk_max_eef_summary)
        and "predicted_chunk_fk_eef_z_max_mean" in predicted_chunk_max_eef_summary.columns
        else float("nan")
    )

    lines = []
    lines.append("# Sequence-aware rest-to-grasp replay diagnosis")
    lines.append("")
    lines.append("## What this test does")
    lines.append("")
    lines.append("This experiment resets the policy once at the start of each replay, feeds a saved rest-to-grasp observation sequence, does not reset between intermediate observations, and records the predicted action at the configured decision step.")
    lines.append("")
    lines.append("This is stronger than a single-snapshot determinism test because it includes policy/preprocessor/action-queue state evolution across the observation history.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```")
    lines.append(f"high = {args.high}")
    lines.append(f"policy_path = {args.policy_path}")
    lines.append(f"dataset_repo_id = {args.dataset_repo_id}")
    lines.append(f"intervention_name = {args.intervention_name}")
    lines.append(f"alpha = {args.alpha}")
    lines.append(f"sequence_chunks = {args.sequence_chunks}")
    lines.append(f"decision_index = {args.decision_index}")
    lines.append(f"repeat_exact_sequence = {args.repeat_exact_sequence}")
    lines.append(f"repeat_per_episode_sequence = {args.repeat_per_episode_sequence}")
    lines.append("```")
    lines.append("")
    lines.append("Exact sequence:")
    lines.append("```")
    lines.append(pd.DataFrame(exact_sequence_rows).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Exact same sequence repeated")
    lines.append("")
    lines.append("Decision-step action repeat summary:")
    lines.append("```")
    lines.append(exact_summary.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append(f"- Max decision-step action std: `{max_std:.8f}`")
    lines.append(f"- Max decision-step action range: `{max_range:.8f}`")
    lines.append(f"- FK(predicted action) EEF-z std: `{fk_std:.8f}`")
    lines.append(f"- FK(predicted action) EEF-z range: `{fk_range:.8f}`")
    if np.isfinite(max_range) and max_range < 1e-5 and np.isfinite(fk_range) and fk_range < 1e-5:
        lines.append("- Interpretation: same saved observation sequence produced numerically identical decision-step joint actions and identical FK-implied commanded EEF height.")
    elif np.isfinite(max_range) and max_range < 1e-5:
        lines.append("- Interpretation: same saved observation sequence produced identical joint actions, but FK EEF-z summary should be checked.")
    else:
        lines.append("- Interpretation: same saved observation sequence produced non-identical decision-step actions; inspect stateful queue, stochasticity, seed, dropout/train mode, or steering hooks.")
    lines.append("")
    lines.append("FK note:")
    lines.append("- `predicted_action_eef_z` is the commanded EEF height implied by the predicted joint target using forward kinematics.")
    lines.append("- It is not the measured physical robot EEF height after motor dynamics, grasp contact, or object interaction.")
    lines.append("")
    lines.append("Saved plots:")
    lines.append("- `plots/sequence_exact_observation_montage.png`")
    lines.append("- `plots/sequence_exact_decision_action_repeat_plot.png`")
    lines.append("- `plots/sequence_exact_decision_all_action_dims_repeat_plot.png`")
    lines.append("- `plots/sequence_exact_decision_predicted_fk_eef_z_repeat_plot.png`")
    lines.append("- `plots/sequence_step_action_values_trial0.png`")
    lines.append("- `plots/sequence_step_predicted_fk_eef_z_trial0.png`")
    lines.append("")
    lines.append("## Predicted action chunk repeatability")
    lines.append("")
    lines.append(f"- Drained predicted/queued action chunk steps: `{args.chunk_steps}`")
    lines.append(f"- Mean max FK EEF-z over predicted action chunk: `{chunk_max_fk_mean:.8f}`")
    lines.append(f"- Range of max FK EEF-z over predicted action chunk across repeats: `{chunk_max_fk_range:.8f}`")
    if np.isfinite(chunk_max_fk_range) and chunk_max_fk_range < 1e-5:
        lines.append("- Interpretation: the max FK-implied EEF height inside the predicted action chunk is identical across repeated replays of the same observation sequence.")
    else:
        lines.append("- Interpretation: max FK-implied EEF height inside the predicted action chunk changed across repeated replays; inspect `sequence_exact_predicted_action_chunk_repeat_consistency.csv`.")
    lines.append("")
    lines.append("Saved predicted-chunk files:")
    lines.append("- `sequence_exact_predicted_action_chunk_trials.csv`")
    lines.append("- `sequence_exact_predicted_action_chunk_per_trial_summary.csv`")
    lines.append("- `sequence_exact_predicted_action_chunk_repeat_consistency.csv`")
    lines.append("- `sequence_exact_predicted_action_chunk_max_fk_eef_z_summary.csv`")
    lines.append("")
    lines.append("Saved predicted-chunk plots:")
    lines.append("- `plots/sequence_exact_predicted_action_chunk_focus_dims_trial0.png`")
    lines.append("- `plots/sequence_exact_predicted_action_chunk_fk_eef_z_trial0.png`")
    lines.append("- `plots/sequence_exact_predicted_action_chunk_max_fk_eef_z_repeat_plot.png`")
    lines.append("- `plots/sequence_exact_predicted_action_chunk_repeat_range_heatmap.png`")
    lines.append("")
    lines.append("Important limitation:")
    lines.append("- The chunk is obtained by feeding the sequence up to the decision observation, then repeatedly calling `predict_action()` with the decision observation to drain the policy/action queue.")
    lines.append("- This is a good offline proxy for an action-chunking policy, but the most definitive physical evidence still requires real-robot fixed-action replay.")
    lines.append("")

    lines.append("## Diverse sequence replay across episodes")
    lines.append("")
    if all_episode_mean.empty:
        lines.append("No all-episode sequence results were produced.")
    else:
        std_cols = [c for c in all_episode_mean.columns if c.endswith("_repeat_std")]
        display_cols = [
            "episode_idx",
            "decision_chunk_idx",
            "num_repeats",
            "predicted_action_eef_z",
            "predicted_action_eef_z_repeat_std",
            "predicted_action_eef_z_repeat_range",
            "lift_eef_z",
        ] + std_cols[:12]
        display_cols = [c for c in display_cols if c in all_episode_mean.columns]
        lines.append("Per-episode repeated-sequence stability:")
        lines.append("```")
        lines.append(all_episode_mean[display_cols].to_string(index=False))
        lines.append("```")
        if std_cols:
            lines.append("")
            lines.append(f"- Max within-episode sequence repeat std: `{float(all_episode_mean[std_cols].max().max()):.8f}`")
    lines.append("")
    if not all_episode_missing.empty:
        lines.append("Missing / skipped episodes:")
        lines.append("```")
        lines.append(all_episode_missing.to_string(index=False))
        lines.append("```")
        lines.append("")
    lines.append("## Professor-facing interpretation")
    lines.append("")
    lines.append("If the exact-sequence max range is near zero, the supported claim is:")
    lines.append("")
    lines.append("> For the same saved rest-to-grasp observation history and the same steering configuration, the policy produced the same lift-decision joint action and the same FK-implied commanded EEF height under this offline replay setup.")
    lines.append("")
    lines.append("If normal real-world rollouts still branch into different lift heights, the likely cause is not random inference from the same history, but differences in the actual pre-lift observation/state/contact/grasp history or dataset-induced multimodality.")
    lines.append("")

    (out_dir / "diagnosis_sequence_replay.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", required=True, help="Path to high intervention debug folder.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--intervention-name", default="high_transport")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--disable-steering", action="store_true")
    parser.add_argument("--task", default="Put the red cube in the box.")
    parser.add_argument("--robot-type", default="koch_follower")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rename-map-json", default=None)

    parser.add_argument("--sequence-chunks", default="0,1,2", help="Chunk observation sequence to feed, e.g. 0,1,2.")
    parser.add_argument("--decision-index", type=int, default=-1, help="Which sequence position to treat as the lift-decision action. -1 means final observation.")
    parser.add_argument("--exact-episode", type=int, default=None, help="Episode used for exact sequence repeat. Default: first available episode.")
    parser.add_argument("--lift-chunk", type=int, default=2, help="Used only for all-episode correlation with actual lift EEF height.")

    parser.add_argument("--repeat-exact-sequence", type=int, default=100)
    parser.add_argument("--repeat-per-episode-sequence", type=int, default=10)
    parser.add_argument("--chunk-steps", type=int, default=50, help="Number of queued/predicted actions to drain after the decision observation.")
    parser.add_argument("--skip-all-episodes", action="store_true", help="Only run the exact sequence repeat test.")
    parser.add_argument("--focus-dims", default="1,2,5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-seed-each-trial", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out)
    plot_dir = out_dir / "plots"
    safe_mkdir(out_dir)
    safe_mkdir(plot_dir)

    set_global_seed(args.seed)

    high_root = Path(args.high)
    sequence_chunks = parse_int_list(args.sequence_chunks)
    decision_pos = resolve_decision_index(args.decision_index, len(sequence_chunks))
    focus_dims = parse_episode_list(args.focus_dims) or []

    xml_path = Path(args.xml)
    if not xml_path.exists():
        xml_path = Path.cwd() / args.xml
    mj_model, mj_data = load_fk_model(xml_path)

    if args.exact_episode is None:
        episodes = list_episode_indices(high_root)
        if not episodes:
            raise RuntimeError(f"No episode directories found under {high_root}")
        exact_episode = int(episodes[0])
    else:
        exact_episode = int(args.exact_episode)

    print("[*] Loading policy context...")
    ctx = load_policy_context(args)

    print(f"[*] Loading exact sequence: episode={exact_episode}, chunks={sequence_chunks}")
    observations, seq_rows, task = load_episode_sequence(
        high_root=high_root,
        episode_idx=exact_episode,
        sequence_chunks=sequence_chunks,
        fallback_task=args.task,
    )

    # Save the visual input sequence used for exact repeated replay.
    save_sequence_observation_montage(
        plot_dir=plot_dir,
        observations=observations,
        sequence_rows=seq_rows,
    )

    print("[*] Replaying exact same sequence repeatedly...")
    exact_all_steps, exact_final = replay_exact_sequence_repeated(
        ctx=ctx,
        observations=observations,
        sequence_rows=seq_rows,
        task=task,
        repeat=args.repeat_exact_sequence,
        decision_index=args.decision_index,
    )

    exact_all_steps = add_predicted_action_fk_eef_z(exact_all_steps, mj_model, mj_data)
    exact_final = add_predicted_action_fk_eef_z(exact_final, mj_model, mj_data)

    exact_all_steps.to_csv(out_dir / "sequence_exact_all_step_actions.csv", index=False)
    exact_final.to_csv(out_dir / "sequence_exact_decision_action_trials.csv", index=False)

    action_cols = sorted([c for c in exact_final.columns if c.startswith("action_values.")], key=natural_sort_key)
    exact_summary = summarize_repeat_trials(exact_final, action_cols)
    exact_summary.to_csv(out_dir / "sequence_exact_decision_action_summary.csv", index=False)

    exact_fk_summary = pd.DataFrame([summarize_scalar_repeat(exact_final, "predicted_action_eef_z")])
    exact_fk_summary.to_csv(out_dir / "sequence_exact_decision_predicted_fk_eef_z_summary.csv", index=False)

    save_final_action_repeat_plot(plot_dir, exact_final, action_cols, focus_dims)
    save_all_action_repeat_plot(plot_dir, exact_final, action_cols)
    save_sequence_step_plot(plot_dir, exact_all_steps, focus_dims)
    save_predicted_fk_eef_repeat_plot(plot_dir, exact_final)
    save_sequence_step_predicted_fk_eef_plot(plot_dir, exact_all_steps)

    print("[*] Replaying exact sequence and draining predicted action chunk...")
    chunk_sequence_feed_trials, predicted_chunk_trials, predicted_chunk_summary = repeat_exact_sequence_predicted_chunk(
        ctx=ctx,
        observations=observations,
        task=task,
        repeat=args.repeat_exact_sequence,
        decision_index=args.decision_index,
        chunk_steps=args.chunk_steps,
        mj_model=mj_model,
        mj_data=mj_data,
    )
    predicted_chunk_consistency = summarize_chunk_repeat_consistency(predicted_chunk_trials)
    predicted_chunk_max_eef_summary = summarize_chunk_max_eef_repeat(predicted_chunk_summary)

    chunk_sequence_feed_trials.to_csv(out_dir / "sequence_exact_chunk_sequence_feed_trials.csv", index=False)
    predicted_chunk_trials.to_csv(out_dir / "sequence_exact_predicted_action_chunk_trials.csv", index=False)
    predicted_chunk_summary.to_csv(out_dir / "sequence_exact_predicted_action_chunk_per_trial_summary.csv", index=False)
    predicted_chunk_consistency.to_csv(out_dir / "sequence_exact_predicted_action_chunk_repeat_consistency.csv", index=False)
    predicted_chunk_max_eef_summary.to_csv(out_dir / "sequence_exact_predicted_action_chunk_max_fk_eef_z_summary.csv", index=False)
    save_predicted_chunk_plots(plot_dir, predicted_chunk_trials, predicted_chunk_summary, focus_dims)

    if args.skip_all_episodes:
        all_episode_trials = pd.DataFrame()
        all_episode_mean = pd.DataFrame()
        all_episode_missing = pd.DataFrame()
    else:
        print("[*] Replaying all episode sequences for diverse-sequence comparison...")
        all_episode_trials, all_episode_mean, all_episode_missing = replay_all_episode_sequences(
            ctx=ctx,
            high_root=high_root,
            sequence_chunks=sequence_chunks,
            decision_index=args.decision_index,
            repeat_per_episode=args.repeat_per_episode_sequence,
            fallback_task=args.task,
            lift_chunk=args.lift_chunk,
            mj_model=mj_model,
            mj_data=mj_data,
        )
        all_episode_trials.to_csv(out_dir / "sequence_all_episode_decision_action_trials.csv", index=False)
        all_episode_mean.to_csv(out_dir / "sequence_all_episode_decision_action_mean.csv", index=False)
        all_episode_missing.to_csv(out_dir / "sequence_all_episode_missing.csv", index=False)
        save_all_episode_sequence_plot(plot_dir, all_episode_mean, focus_dims)

    write_report(
        out_dir=out_dir,
        args=args,
        exact_summary=exact_summary,
        exact_fk_summary=exact_fk_summary,
        exact_sequence_rows=seq_rows,
        predicted_chunk_max_eef_summary=predicted_chunk_max_eef_summary,
        all_episode_mean=all_episode_mean,
        all_episode_missing=all_episode_missing,
    )

    print("[DONE] Sequence-aware replay analysis complete.")
    print(f"[DONE] Output directory: {out_dir.resolve()}")
    print()
    print("Exact sequence:")
    print(pd.DataFrame(seq_rows).to_string(index=False))
    print()
    print("Exact same sequence decision-action repeat summary:")
    print(exact_summary.head(20).to_string(index=False))
    print()
    print("Key outputs:")
    print(f"  - {out_dir / 'diagnosis_sequence_replay.md'}")
    print(f"  - {out_dir / 'sequence_exact_decision_action_summary.csv'}")
    print(f"  - {out_dir / 'sequence_exact_decision_predicted_fk_eef_z_summary.csv'}")
    print(f"  - {out_dir / 'sequence_exact_predicted_action_chunk_trials.csv'}")
    print(f"  - {out_dir / 'sequence_exact_predicted_action_chunk_per_trial_summary.csv'}")
    print(f"  - {out_dir / 'sequence_exact_predicted_action_chunk_repeat_consistency.csv'}")
    print(f"  - {out_dir / 'sequence_exact_predicted_action_chunk_max_fk_eef_z_summary.csv'}")
    print(f"  - {out_dir / 'sequence_exact_decision_action_trials.csv'}")
    print(f"  - {out_dir / 'sequence_exact_all_step_actions.csv'}")
    print(f"  - {out_dir / 'sequence_all_episode_decision_action_mean.csv'}")
    print(f"  - {plot_dir / 'sequence_exact_observation_montage.png'}")
    print(f"  - {plot_dir / 'sequence_exact_decision_action_repeat_plot.png'}")
    print(f"  - {plot_dir / 'sequence_exact_decision_all_action_dims_repeat_plot.png'}")
    print(f"  - {plot_dir / 'sequence_exact_decision_predicted_fk_eef_z_repeat_plot.png'}")
    print(f"  - {plot_dir / 'sequence_step_action_values_trial0.png'}")
    print(f"  - {plot_dir / 'sequence_step_predicted_fk_eef_z_trial0.png'}")
    print(f"  - {plot_dir / 'sequence_exact_predicted_action_chunk_fk_eef_z_trial0.png'}")
    print(f"  - {plot_dir / 'sequence_exact_predicted_action_chunk_max_fk_eef_z_repeat_plot.png'}")
    print(f"  - {plot_dir / 'sequence_exact_predicted_action_chunk_repeat_range_heatmap.png'}")


if __name__ == "__main__":
    main()