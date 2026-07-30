#!/usr/bin/env python3
"""
LIBERO CLOSED-LOOP Activation-Steering Evaluation for SmolVLA
─────────────────────────────────────────────────────────────
Unlike libero_eval_steering.py (offline, replayed images + open-loop state),
this runs the ACTUAL LIBERO simulator in the loop:

  reset → get real camera image + joint state
        → SmolVLA predicts 8D action  [target_joint_1..7, gripper_cmd]
        → command robot (JOINT_POSITION control, delta = target − current)
        → env renders NEW image reflecting the robot's new pose
        → repeat

Because each steered action produces a new observation, the steering effect
COMPOUNDS over the trajectory — this is the only faithful way to reproduce the
paper's (arXiv:2509.00328) closed-loop steering result for high/low EEF carry.

Runs in the `lerobot` conda env (which can import both SmolVLA and LIBERO).

Conditions (same steering mechanism as libero_eval_steering.py):
  none          baseline, no steering
  keyword_high  SET high-concept neurons = alpha  (paper keyword method)
  keyword_low   SET low-concept  neurons = alpha
  caa_high / caa_low   dense CAA vector ± alpha  (if --caa-path given)

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_closedloop_eval.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero \\
    --neurons-json outputs/libero_height_neurons_ref.json \\
    --conditions none keyword_high keyword_low \\
    --keyword-alpha 6.0 --steering-mode set --keyword-top-n 10 \\
    --task-idx 0 --n-rollouts 20 --max-steps 220 \\
    --save-video --out-dir outputs/libero_closedloop_keyword
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Reuse steering + metrics + plotting from the offline eval script
from lerobot.scripts.libero_eval_steering import (
    setup_keyword_neurons,
    setup_caa,
    clear_steering,
    rollout_metrics,
    aggregate,
    cohen_d,
    save_csv,
    plot_comparison,
)

BDDL_ROOT = Path(
    "/home/bruce/anaconda3/envs/libero_sim/lib/python3.10/"
    "site-packages/libero/libero/bddl_files"
)
DEMO_DIR = Path("/home/bruce/datasets/libero_demos")

_EMPTY_256 = np.zeros((256, 256, 3), dtype=np.uint8)
_EMPTY_480 = np.zeros((480, 640, 3), dtype=np.uint8)


# ── Observation builder (matches training input format) ───────────────────────

def build_obs(img_hwc: np.ndarray, state8: np.ndarray) -> dict:
    return {
        "observation.images.agentview":      img_hwc,
        "observation.images.camera2":        _EMPTY_256,
        "observation.images.camera3":        _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.images.empty_camera_1": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


# ── LIBERO env with absolute-joint-position control ───────────────────────────

def patch_controller_identity(env, lim: float = 3.15):
    """
    Patch the JointPositionController so a commanded joint DELTA (radians)
    passes through unscaled: env action = [Δq_1..7, gripper].
    Default LIBERO scaling is input±1 → output±0.05 rad, too small/nonlinear.
    """
    ctrl = env.env.robots[0].controller
    jd = len(ctrl.qpos_index)
    ctrl.input_max  = np.ones(jd) * lim
    ctrl.input_min  = -np.ones(jd) * lim
    ctrl.output_max = np.ones(jd) * lim
    ctrl.output_min = -np.ones(jd) * lim
    ctrl.action_scale = (ctrl.output_max - ctrl.output_min) / (ctrl.input_max - ctrl.input_min)
    ctrl.action_input_transform  = (ctrl.input_max + ctrl.input_min) / 2.0
    ctrl.action_output_transform = (ctrl.output_max + ctrl.output_min) / 2.0


def make_env(bddl_path: Path):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=["agentview"],
        camera_heights=256, camera_widths=256,
        controller="JOINT_POSITION",
    )
    return env


def find_demo_file(suite: str, task_fname: str) -> Path | None:
    stem = task_fname.replace(".bddl", "")
    for c in [DEMO_DIR / suite / f"{stem}_demo.hdf5", DEMO_DIR / suite / f"{stem}.hdf5"]:
        if c.exists():
            return c
    d = DEMO_DIR / suite
    if d.exists():
        for p in d.glob("*.hdf5"):
            if stem[:30] in p.stem:
                return p
    return None


def load_demo_init_states(demo_file: Path) -> list[np.ndarray]:
    """Return the initial sim state of each human demo (for env.set_init_state)."""
    states = []
    with h5py.File(demo_file, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            states.append(f[f"data/{k}/states"][()][0])
    return states


# ── One closed-loop rollout ───────────────────────────────────────────────────

def run_closedloop_rollout(env, init_state, policy, preprocessor, postprocessor,
                           device, task, max_steps, arc_type):
    from lerobot.utils.control_utils import predict_action

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    env.reset()
    obs = env.set_init_state(init_state)
    # settle a few steps (zero delta, gripper open)
    for _ in range(5):
        obs, _, _, _ = env.step(np.concatenate([np.zeros(7), [-1.0]]))

    gripper_cmd = -1.0  # open
    eef_z, images, joints = [], [], []

    for t in range(max_steps):
        img = obs["agentview_image"]
        cur_joints = obs["robot0_joint_pos"].astype(np.float64)
        state8 = np.concatenate([cur_joints, [gripper_cmd]]).astype(np.float32)

        with torch.no_grad():
            action = predict_action(
                observation=build_obs(img, state8),
                policy=policy, device=device,
                preprocessor=preprocessor, postprocessor=postprocessor,
                use_amp=False, task=task,
            )
        act = (action.detach().cpu().numpy() if torch.is_tensor(action)
               else np.asarray(action)).reshape(-1)

        target_joints = act[:7].astype(np.float64)
        gripper_cmd   = float(np.clip(act[7], -1.0, 1.0))
        delta = target_joints - cur_joints
        env_action = np.concatenate([delta, [gripper_cmd]])
        obs, _, done, _ = env.step(env_action)

        eef_z.append(float(obs["robot0_eef_pos"][2]))
        images.append(img)
        joints.append(cur_joints)
        if done:
            break

    return {
        "eef_heights": np.array(eef_z, dtype=np.float64),          # world-frame z (m)
        "ref_eef_z":   np.array(eef_z, dtype=np.float64),           # no separate demo ref here
        "pred_joints": np.array(joints, dtype=np.float64),
        "imgs":        np.array(images, dtype=np.uint8),
        "T": len(eef_z),
        "arc_type": arc_type,
    }


# ── Video ─────────────────────────────────────────────────────────────────────

def save_rollout_video(result, condition, out_path: Path, fps=20.0):
    imgs = result["imgs"]
    eef  = result["eef_heights"] * 100
    T, H, W = len(imgs), 256, 256
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    peak = float(np.max(eef)) if len(eef) else 0.0
    for t in range(T):
        frame = cv2.cvtColor(imgs[t], cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"z={eef[t]:.1f}cm t={t}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{condition} peak={peak:.1f}", (6, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LIBERO closed-loop steering eval for SmolVLA.")
    ap.add_argument("--policy-path", default="ethanCSL/svla_franka_pick_n_place_vla_steering_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--task", default="Pick up the black bowl and place it on the plate.")
    ap.add_argument("--conditions", nargs="+", default=["none", "keyword_high", "keyword_low"])
    ap.add_argument("--n-rollouts", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--neurons-json", default=None)
    ap.add_argument("--keyword-alpha", type=float, default=6.0)
    ap.add_argument("--keyword-top-n", type=int, default=10)
    ap.add_argument("--steering-mode", choices=["add", "set"], default="set")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--caa-path", default=None)
    ap.add_argument("--caa-alpha", type=float, default=2.0)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--video-rollouts", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--out-dir", default="outputs/libero_closedloop")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device
    from libero.libero import benchmark

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ── Resolve task + demo init states ───────────────────────────────────────
    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    fname = suite_obj.get_task_bddl_files()[args.task_idx]
    bddl_path = BDDL_ROOT / args.suite / fname
    print(f"Task {args.task_idx}: {fname}")

    demo_file = find_demo_file(args.suite, fname)
    if demo_file is None:
        raise FileNotFoundError(f"No human demo HDF5 for task under {DEMO_DIR/args.suite}")
    init_states = load_demo_init_states(demo_file)
    print(f"Loaded {len(init_states)} demo init states from {demo_file.name}")

    # ── Policy ────────────────────────────────────────────────────────────────
    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy_cfg = policy.config
    policy_cfg.device = str(device)
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=args.policy_path,
    )

    # ── Env ───────────────────────────────────────────────────────────────────
    print("Creating LIBERO env (JOINT_POSITION control) ...")
    env = make_env(bddl_path)
    env.reset()
    patch_controller_identity(env)

    # ── Condition loop ────────────────────────────────────────────────────────
    all_traces:  dict[str, list] = {}
    all_metrics: dict[str, list] = {}

    for cond in args.conditions:
        print(f"\n{'='*60}\nCondition: {cond}\n{'='*60}")
        if cond == "none":
            clear_steering(policy)
        elif cond == "keyword_high":
            setup_keyword_neurons(policy, args.neurons_json, "high",
                                  alpha=args.keyword_alpha, top_n=args.keyword_top_n,
                                  mode=args.steering_mode, bidirectional=args.bidirectional)
        elif cond == "keyword_low":
            setup_keyword_neurons(policy, args.neurons_json, "low",
                                  alpha=args.keyword_alpha, top_n=args.keyword_top_n,
                                  mode=args.steering_mode, bidirectional=args.bidirectional)
        elif cond == "caa_high":
            setup_caa(policy, args.caa_path, alpha=+args.caa_alpha)
        elif cond == "caa_low":
            setup_caa(policy, args.caa_path, alpha=-args.caa_alpha)
        else:
            print(f"  [skip] unknown condition '{cond}'"); continue

        cond_traces, cond_metrics = [], []
        for ri in tqdm(range(args.n_rollouts), desc=f"  {cond}"):
            init = init_states[ri % len(init_states)]
            result = run_closedloop_rollout(
                env, init, policy, preprocessor, postprocessor,
                device, args.task, args.max_steps, arc_type=cond,
            )
            cond_traces.append(result["eef_heights"])
            cond_metrics.append(rollout_metrics(ri, result, cond))
            if args.save_video and ri in args.video_rollouts:
                save_rollout_video(result, cond,
                                   out_dir / "videos" / cond / f"rollout_{ri:03d}.mp4",
                                   fps=args.fps)

        all_traces[cond] = cond_traces
        all_metrics[cond] = cond_metrics
        save_csv(out_dir / f"rollout_metrics_{cond}.csv", cond_metrics)

        agg = aggregate(cond_metrics, "carry_peak_cm")
        if agg:
            print(f"\n  carry_peak: mean={agg['mean']:.2f} ± {agg['std']:.2f} cm  "
                  f"CI95=[{agg['ci95_lo']:.2f},{agg['ci95_hi']:.2f}]  n={agg['n']}")

        # raw traces
        max_T = max(len(t) for t in cond_traces)
        arr = np.full((len(cond_traces), max_T), np.nan)
        for i, t in enumerate(cond_traces):
            arr[i, :len(t)] = t
        np.savez_compressed(out_dir / f"eef_traces_{cond}.npz", traces_m=arr, traces_cm=arr*100)

    env.close()

    # ── Figure ────────────────────────────────────────────────────────────────
    if all_traces:
        trace_arrays = {}
        for cond, traces in all_traces.items():
            max_T = max(len(t) for t in traces)
            arr = np.full((len(traces), max_T), np.nan)
            for i, t in enumerate(traces):
                arr[i, :len(t)] = t
            trace_arrays[cond] = arr
        plot_comparison(trace_arrays, out_dir / "eef_trajectory_comparison.png",
                        ref_traces=None, metrics_by_cond=all_metrics)

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(all_metrics) > 1:
        print("\n" + "="*72)
        print("Cross-condition Summary (carry-phase peak EEF height, cm)")
        print("="*72)
        print(f"{'Condition':<16}{'Mean':>8}{'±Std':>7}{'CI95_lo':>9}{'CI95_hi':>9}{'Cohen-d':>9}")
        print("-"*72)
        base = None
        for cond, metrics in all_metrics.items():
            peaks = np.array([r["carry_peak_cm"] for r in metrics])
            agg = aggregate(metrics, "carry_peak_cm")
            d = ""
            if cond == "none":
                base = peaks
            elif base is not None:
                d = f"{cohen_d(base, peaks):+.3f}"
            print(f"  {cond:<14}{agg.get('mean',0):8.2f}{agg.get('std',0):7.2f}"
                  f"{agg.get('ci95_lo',0):9.2f}{agg.get('ci95_hi',0):9.2f}{d:>9}")

    print(f"\n[✓] All outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
