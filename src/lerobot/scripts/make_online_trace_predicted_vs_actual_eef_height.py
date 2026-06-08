#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import torch


ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

MUJOCO_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

CALIB_REST_STATE_DEG = np.asarray(
    [-11.20879121, 97.00520833, 17.25596857, 100.0, -9.01098901, 47.38863287],
    dtype=np.float64,
)

CALIB_TARGET_REST_DEG = np.asarray(
    [91.7, 15.0, 40.0, 65.0, 0.0, -30.0],
    dtype=np.float64,
)

CALIB_ORDER = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
CALIB_SCALE = np.asarray([1.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
CALIB_SIGN = np.asarray([1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float64)


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def vec_from_any(x: Any) -> np.ndarray | None:
    if x is None:
        return None

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    if isinstance(x, dict):
        if all(k in x for k in ACTION_KEYS):
            vals = []
            for k in ACTION_KEYS:
                v = x[k]
                if torch.is_tensor(v):
                    v = v.detach().cpu().numpy()
                vals.append(float(np.asarray(v).reshape(-1)[0]))
            return np.asarray(vals, dtype=np.float64)

        # fallback for flattened action_values.0 ... action_values.5
        flat_keys = [f"action_values.{i}" for i in range(6)]
        if all(k in x for k in flat_keys):
            return np.asarray([float(np.asarray(x[k]).reshape(-1)[0]) for k in flat_keys], dtype=np.float64)

        return None

    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        return None
    return arr[:6].copy()


def state_to_q_rad(state_vec: np.ndarray) -> np.ndarray:
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)
    if state_vec.size < 6:
        raise ValueError(f"Expected state dim >= 6, got {state_vec.shape}")

    raw_deg = state_vec[:6].copy()
    raw_delta = raw_deg[CALIB_ORDER] - CALIB_REST_STATE_DEG[CALIB_ORDER]
    q_deg = CALIB_TARGET_REST_DEG + CALIB_SIGN * CALIB_SCALE * raw_delta
    return np.deg2rad(q_deg)


def eef_z_from_state(state_vec: np.ndarray, model, data) -> float:
    q = state_to_q_rad(state_vec)

    mujoco.mj_resetData(model, data)

    for i, joint_name in enumerate(MUJOCO_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint not found in XML: {joint_name}")
        data.qpos[model.jnt_qposadr[jid]] = float(q[i])

    mujoco.mj_forward(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
    if sid < 0:
        raise ValueError("Site not found in XML: end_effector_site")

    return float(data.site_xpos[sid][2])


def load_trace(trace_path: Path) -> list[dict[str, Any]]:
    x = torch.load(trace_path, map_location="cpu", weights_only=False)
    if "trace" not in x:
        raise KeyError(f"No 'trace' in {trace_path}")
    return x["trace"]


def fit_action_to_obs_mapping(run: Path, pred_source: str, delay: int) -> pd.DataFrame:
    xs = []
    ys = []

    for trace_path in sorted(run.glob("episode_*/policy_action_trace.pt")):
        trace = load_trace(trace_path)

        for i, step in enumerate(trace):
            j = i + delay
            if j >= len(trace):
                continue

            action = vec_from_any(step.get(pred_source))
            obs = vec_from_any(trace[j].get("observation_state"))

            if action is None or obs is None:
                continue

            xs.append(action)
            ys.append(obs)

    if len(xs) < 5:
        raise RuntimeError("Not enough action/observation pairs to fit mapping.")

    X = np.asarray(xs, dtype=np.float64)
    Y = np.asarray(ys, dtype=np.float64)

    rows = []
    for j, name in enumerate(JOINT_NAMES):
        x = X[:, j]
        y = Y[:, j]

        obs_std = float(np.std(y))
        if obs_std < 1e-9:
            a = 0.0
            b = float(np.mean(y))
            y_hat = np.full_like(y, b)
            corr = np.nan
            note = "obs_constant"
        else:
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            y_hat = a * x + b
            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 else np.nan
            note = "fit"

        rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))

        rows.append({
            "joint_idx": j,
            "joint_name": name,
            "a": float(a),
            "b": float(b),
            "rmse": rmse,
            "corr": corr,
            "obs_std": obs_std,
            "note": note,
        })

    return pd.DataFrame(rows)


def map_action_to_obs(action: np.ndarray, mapping: pd.DataFrame) -> np.ndarray:
    out = np.asarray(action, dtype=np.float64).reshape(-1)[:6].copy()
    for _, row in mapping.iterrows():
        j = int(row["joint_idx"])
        out[j] = float(row["a"]) * out[j] + float(row["b"])
    return out


def robust_metrics(z: np.ndarray) -> dict[str, float]:
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return {"max": np.nan, "p95": np.nan, "top5_mean": np.nan}
    top_k = min(5, z.size)
    return {
        "max": float(np.max(z)),
        "p95": float(np.percentile(z, 95)),
        "top5_mean": float(np.mean(np.sort(z)[-top_k:])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="debug_runs/<run_id>")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-source", default="sent_action", choices=[
        "policy_action_values",
        "act_processed_policy",
        "robot_action_to_send",
        "sent_action",
    ])
    ap.add_argument("--actual-source", default="observation_state")
    ap.add_argument("--chunk-index", type=int, default=2)
    ap.add_argument("--delay", type=int, default=4)
    ap.add_argument("--actual-window", default="chunk", choices=["chunk", "full"])
    args = ap.parse_args()

    run = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)

    print("[*] Fitting action -> observation_state mapping...")
    mapping = fit_action_to_obs_mapping(run, pred_source=args.pred_source, delay=args.delay)
    mapping.to_csv(out_dir / "action_to_observation_mapping.csv", index=False)
    print(mapping.to_string(index=False))

    summary_rows = []
    step_rows = []

    for trace_path in sorted(run.glob("episode_*/policy_action_trace.pt")):
        ep = int(trace_path.parent.name.split("_")[-1])
        trace = load_trace(trace_path)

        pred_z = []
        actual_z = []
        sent_minus_actual_z = []

        selected_indices = [
            i for i, step in enumerate(trace)
            if step.get("chunk_index") == args.chunk_index
        ]

        # Predicted side: selected chunk action sequence.
        for local_step, i in enumerate(selected_indices):
            action = vec_from_any(trace[i].get(args.pred_source))
            if action is None:
                continue

            mapped = map_action_to_obs(action, mapping)
            z_pred = eef_z_from_state(mapped, model, data)
            pred_z.append(z_pred)

            row = {
                "episode_idx": ep,
                "chunk_step": local_step,
                "trace_step": i,
                "pred_fk_z": z_pred,
            }

            for j, name in enumerate(JOINT_NAMES):
                row[f"pred_mapped_{name}"] = mapped[j]

            # Actual side for chunk-aligned mode.
            if args.actual_window == "chunk":
                obs_i = i + args.delay
                if obs_i < len(trace):
                    obs = vec_from_any(trace[obs_i].get(args.actual_source))
                    if obs is not None:
                        z_actual = eef_z_from_state(obs, model, data)
                        actual_z.append(z_actual)
                        row["actual_trace_step"] = obs_i
                        row["actual_fk_z"] = z_actual
                        row["pred_minus_actual_fk_z"] = z_pred - z_actual

                        for j, name in enumerate(JOINT_NAMES):
                            row[f"actual_{name}"] = obs[j]
                            row[f"pred_minus_actual_{name}"] = mapped[j] - obs[j]

            step_rows.append(row)

        # Actual side for full-trajectory mode.
        if args.actual_window == "full":
            for i, step in enumerate(trace):
                obs = vec_from_any(step.get(args.actual_source))
                if obs is None:
                    continue
                actual_z.append(eef_z_from_state(obs, model, data))

        pred_z = np.asarray(pred_z, dtype=np.float64)
        actual_z = np.asarray(actual_z, dtype=np.float64)

        pm = robust_metrics(pred_z)
        am = robust_metrics(actual_z)

        row = {
            "episode_idx": ep,
            "num_pred_points": int(pred_z.size),
            "num_actual_points": int(actual_z.size),
            "pred_source": args.pred_source,
            "actual_source": args.actual_source,
            "chunk_index": args.chunk_index,
            "delay": args.delay,
            "actual_window": args.actual_window,
            "pred_max_fk_z_m": pm["max"],
            "actual_max_fk_z_m": am["max"],
            "diff_max_m": pm["max"] - am["max"],
            "diff_max_cm": (pm["max"] - am["max"]) * 100.0,
            "pred_p95_fk_z_m": pm["p95"],
            "actual_p95_fk_z_m": am["p95"],
            "diff_p95_cm": (pm["p95"] - am["p95"]) * 100.0,
            "pred_top5_mean_fk_z_m": pm["top5_mean"],
            "actual_top5_mean_fk_z_m": am["top5_mean"],
            "diff_top5_mean_cm": (pm["top5_mean"] - am["top5_mean"]) * 100.0,
        }
        summary_rows.append(row)

        # Per-episode trajectory plot for chunk-aligned mode.
        if args.actual_window == "chunk" and pred_z.size > 0 and actual_z.size > 0:
            n = min(pred_z.size, actual_z.size)
            plt.figure(figsize=(8, 4.5))
            plt.plot(np.arange(n), pred_z[:n] * 100.0, marker="o", label="Predicted EEF height")
            plt.plot(np.arange(n), actual_z[:n] * 100.0, marker="o", label=f"Actual EEF height t+{args.delay}")
            plt.xlabel(f"Chunk {args.chunk_index} step")
            plt.ylabel("EEF height (cm)")
            plt.title(f"Episode {ep:06d}: predicted vs actual EEF height")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"episode_{ep:06d}_predicted_vs_actual_by_step.png", dpi=300, bbox_inches="tight")
            plt.savefig(out_dir / f"episode_{ep:06d}_predicted_vs_actual_by_step.pdf", bbox_inches="tight")
            plt.close()

    summary = pd.DataFrame(summary_rows).sort_values("episode_idx")
    steps = pd.DataFrame(step_rows).sort_values(["episode_idx", "chunk_step"])

    summary.to_csv(out_dir / "online_trace_predicted_vs_actual_eef_summary.csv", index=False)
    steps.to_csv(out_dir / "online_trace_predicted_vs_actual_eef_by_step.csv", index=False)

    # Main paper line plot.
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(summary["episode_idx"], summary["pred_max_fk_z_m"] * 100.0, marker="o", label="Predicted max EEF height")
    plt.plot(summary["episode_idx"], summary["actual_max_fk_z_m"] * 100.0, marker="o", label="Actual max EEF height")
    plt.xlabel("Episode index")
    plt.ylabel("Max EEF height (cm)")
    plt.title("Predicted vs actual EEF height")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "paper_predicted_vs_actual_max_eef_height.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "paper_predicted_vs_actual_max_eef_height.pdf", bbox_inches="tight")
    plt.close()

    # Residual plot.
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(summary["episode_idx"], summary["diff_max_cm"], marker="o", label="Predicted - actual")
    plt.axhline(0.0, linestyle="--")
    plt.axhspan(-0.5, 0.5, alpha=0.2, label="±0.5 cm validation band")
    plt.xlabel("Episode index")
    plt.ylabel("Max height error (cm)")
    plt.title("Predicted minus actual EEF height")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "paper_predicted_minus_actual_max_eef_height.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "paper_predicted_minus_actual_max_eef_height.pdf", bbox_inches="tight")
    plt.close()

    # Parity plot.
    lo = min(summary["actual_max_fk_z_m"].min(), summary["pred_max_fk_z_m"].min()) * 100.0 - 0.5
    hi = max(summary["actual_max_fk_z_m"].max(), summary["pred_max_fk_z_m"].max()) * 100.0 + 0.5

    plt.figure(figsize=(5, 5))
    plt.scatter(summary["actual_max_fk_z_m"] * 100.0, summary["pred_max_fk_z_m"] * 100.0)
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="y = x")
    for _, row in summary.iterrows():
        plt.text(row["actual_max_fk_z_m"] * 100.0, row["pred_max_fk_z_m"] * 100.0, str(int(row["episode_idx"])), fontsize=8)
    plt.xlabel("Actual max EEF height (cm)")
    plt.ylabel("Predicted max EEF height (cm)")
    plt.title("Predicted vs actual EEF height parity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "paper_predicted_vs_actual_max_eef_height_parity.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "paper_predicted_vs_actual_max_eef_height_parity.pdf", bbox_inches="tight")
    plt.close()

    print("\nSaved outputs:")
    print(" ", out_dir / "online_trace_predicted_vs_actual_eef_summary.csv")
    print(" ", out_dir / "online_trace_predicted_vs_actual_eef_by_step.csv")
    print(" ", out_dir / "paper_predicted_vs_actual_max_eef_height.png")
    print(" ", out_dir / "paper_predicted_minus_actual_max_eef_height.png")
    print(" ", out_dir / "paper_predicted_vs_actual_max_eef_height_parity.png")

    print("\n=== Summary ===")
    cols = [
        "episode_idx",
        "num_pred_points",
        "num_actual_points",
        "pred_max_fk_z_m",
        "actual_max_fk_z_m",
        "diff_max_cm",
        "diff_p95_cm",
        "diff_top5_mean_cm",
    ]
    print(summary[cols].to_string(index=False))

    print("\n=== Metrics ===")
    print("N =", len(summary))
    print("mean max diff cm =", summary["diff_max_cm"].mean())
    print("std max diff cm =", summary["diff_max_cm"].std())
    print("max abs max diff cm =", summary["diff_max_cm"].abs().max())
    print("mean p95 diff cm =", summary["diff_p95_cm"].mean())
    print("mean top5 diff cm =", summary["diff_top5_mean_cm"].mean())


if __name__ == "__main__":
    main()
