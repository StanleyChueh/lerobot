#!/usr/bin/env python3
"""
LIBERO CLOSED-LOOP eval for a SmolVLA trained in the OFFICIAL OSC format.
──────────────────────────────────────────────────────────────────────────
Companion to libero_closedloop_eval.py (which is for the joint-position model).
This one is for the retrained OSC model (dataset: image/image2, 8D eef state,
7D OSC-pose-delta action). Evaluation is MUCH simpler than the joint model:

  reset → real agentview + wrist image + 8D eef state
        → SmolVLA predicts 7D OSC delta [dx,dy,dz,drx,dry,drz, gripper]
        → env.step(action)   (default OSC_POSE controller — NO patching)
        → env renders new obs → repeat

Reports grasp success (bowl lifted) and task success (bowl on plate).

The training rename_map maps observation.images.image→camera1 and
observation.images.image2→camera2 (empty_cameras=1), so the model's image
inputs are camera1 (agentview), camera2 (wrist), camera3+empty_camera_0 blank.

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_osc_eval.py \\
    --policy-path outputs/train/<run>/checkpoints/last/pretrained_model \\
    --task-idx 0 --n-rollouts 20 --max-steps 400 --n-action-steps 10 \\
    --save-video --out-dir outputs/libero_osc_baseline
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

BDDL_ROOT = Path(
    "/home/bruce/anaconda3/envs/libero_sim/lib/python3.10/"
    "site-packages/libero/libero/bddl_files"
)
DEMO_DIR = Path("/home/bruce/datasets/libero_demos")

_EMPTY_256 = np.zeros((256, 256, 3), dtype=np.uint8)
_EMPTY_480 = np.zeros((480, 640, 3), dtype=np.uint8)


def eef_state(obs) -> np.ndarray:
    """8D LIBERO eef state: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2). Matches collector."""
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([
        obs["robot0_eef_pos"].astype(np.float32),
        quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        obs["robot0_gripper_qpos"].astype(np.float32),
    ])


def build_obs(agent_hwc: np.ndarray, wrist_hwc: np.ndarray, state8: np.ndarray) -> dict:
    """camera1=agentview, camera2=wrist, camera3+empty_camera_0 blank (rename_map + empty_cameras=1)."""
    return {
        "observation.images.camera1":        agent_hwc,
        "observation.images.camera2":        wrist_hwc,
        "observation.images.camera3":        _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


def make_env(bddl_path: Path):
    """Default LIBERO controller is OSC_POSE — exactly what the OSC dataset was collected under."""
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256, camera_widths=256,
    )


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
    with h5py.File(demo_file, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        return [f[f"data/{k}/states"][()][0] for k in keys]


def _bowl_keys(obs):
    return [k for k in obs if "bowl" in k.lower() and k.endswith("_pos") and "to_robot" not in k]


def run_rollout(env, init_state, policy, preprocessor, postprocessor,
                device, task, max_steps, plate_key=None):
    from lerobot.utils.control_utils import predict_action

    policy.reset(); preprocessor.reset(); postprocessor.reset()
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):  # settle (zero OSC delta, gripper open)
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))

    bowls = _bowl_keys(obs)
    b0 = {b: float(obs[b][2]) for b in bowls}
    bowl_peak = dict(b0)
    eef_z, images = [], []

    for t in range(max_steps):
        agent = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        with torch.no_grad():
            action = predict_action(
                observation=build_obs(agent, wrist, eef_state(obs)),
                policy=policy, device=device,
                preprocessor=preprocessor, postprocessor=postprocessor,
                use_amp=False, task=task,
            )
        act = (action.detach().cpu().numpy() if torch.is_tensor(action)
               else np.asarray(action)).reshape(-1)[:7].astype(np.float32)
        obs, _, done, _ = env.step(act)
        eef_z.append(float(obs["robot0_eef_pos"][2]))
        for b in bowls:
            bowl_peak[b] = max(bowl_peak[b], float(obs[b][2]))
        images.append(agent)
        if done:
            break

    lifted = max(bowls, key=lambda b: bowl_peak[b] - b0[b]) if bowls else None
    grasped = bool(lifted is not None and bowl_peak[lifted] - b0[lifted] > 0.03)
    on_plate = False
    if lifted is not None and plate_key is not None and plate_key in obs:
        fb, tp = obs[lifted], obs[plate_key]
        dxy = float(np.linalg.norm(fb[:2] - tp[:2])); dz = float(fb[2] - tp[2])
        on_plate = bool(dxy < 0.06 and -0.02 < dz < 0.08 and fb[2] > 0.88)

    return {
        "eef_heights": np.array(eef_z, dtype=np.float64),
        "imgs": np.array(images, dtype=np.uint8),
        "grasped": grasped, "on_plate": on_plate, "T": len(eef_z),
    }


def save_rollout_video(result, tag, out_path: Path, fps=20.0):
    imgs = result["imgs"]; eef = result["eef_heights"] * 100
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 256))
    for t in range(len(imgs)):
        f = cv2.cvtColor(imgs[t], cv2.COLOR_RGB2BGR)
        z = f"z={eef[t]:.1f}cm" if t < len(eef) else ""
        cv2.putText(f, f"{tag}  {z}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)
        w.write(f)
    w.release()


def main():
    ap = argparse.ArgumentParser(description="LIBERO OSC-format closed-loop eval for SmolVLA.")
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--task", default="pick the akita black bowl between the plate "
                    "and the ramekin and place it on the plate")
    ap.add_argument("--n-rollouts", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--out-dir", default="outputs/libero_osc_baseline")
    ap.add_argument("--device", default=None)
    # ── steering ──
    ap.add_argument("--conditions", nargs="+", default=["none"],
                    help="none, caa_high, caa_low, physical_high, physical_low")
    ap.add_argument("--caa-path", default=None, help="expert-space CAA vectors (setup_caa)")
    ap.add_argument("--caa-alpha", type=float, default=6.0)
    ap.add_argument("--physical-vectors", default=None, help="VLM-space contrast vectors (setup_physical_caa)")
    ap.add_argument("--physical-alpha", type=float, default=6.0)
    ap.add_argument("--physical-top-k", type=int, default=None)
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device
    from libero.libero import benchmark
    from lerobot.scripts.libero_eval_steering import (
        setup_caa, setup_physical_caa, clear_steering)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    fname = suite_obj.get_task_bddl_files()[args.task_idx]
    bddl_path = BDDL_ROOT / args.suite / fname
    print(f"Task {args.task_idx}: {fname}")

    demo_file = find_demo_file(args.suite, fname)
    if demo_file is None:
        raise FileNotFoundError(f"No human demo HDF5 under {DEMO_DIR/args.suite}")
    init_states = load_demo_init_states(demo_file)
    print(f"Loaded {len(init_states)} demo init states")

    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    if args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy_path)

    env = make_env(bddl_path)
    obs0 = env.reset()
    plate_key = next((k for k in obs0 if "plate" in k.lower()
                      and k.endswith("_pos") and "to_robot" not in k), None)
    print(f"plate_key={plate_key}")

    def carry_peak_cm(eef_m):
        e = np.asarray(eef_m) * 100
        if len(e) > 4:
            lo, hi = int(0.20 * len(e)), int(0.80 * len(e))
            return float(np.max(e[lo:hi]))
        return float(np.max(e)) if len(e) else 0.0

    def apply_condition(cond):
        if cond == "none":
            clear_steering(policy)
        elif cond == "caa_high":
            setup_caa(policy, args.caa_path, alpha=+args.caa_alpha)
        elif cond == "caa_low":
            setup_caa(policy, args.caa_path, alpha=-args.caa_alpha)
        elif cond == "physical_high":
            setup_physical_caa(policy, args.physical_vectors, "high",
                               alpha=args.physical_alpha, top_k=args.physical_top_k)
        elif cond == "physical_low":
            setup_physical_caa(policy, args.physical_vectors, "low",
                               alpha=args.physical_alpha, top_k=args.physical_top_k)
        else:
            raise ValueError(f"unknown condition {cond}")

    n = args.n_rollouts
    peaks_by_cond, peaks_grasped_by_cond, summary = {}, {}, {}
    for cond in args.conditions:
        print(f"\n{'='*56}\nCondition: {cond}\n{'='*56}")
        apply_condition(cond)
        n_grasp = n_plate = 0
        cond_peaks, cond_peaks_grasped = [], []
        for ri in tqdm(range(n), desc=f"  {cond}"):
            init = init_states[ri % len(init_states)]
            r = run_rollout(env, init, policy, preprocessor, postprocessor,
                            device, args.task, args.max_steps, plate_key)
            n_grasp += int(r["grasped"]); n_plate += int(r["on_plate"])
            pk = carry_peak_cm(r["eef_heights"])
            cond_peaks.append(pk)
            if r["grasped"]:                       # only a HELD bowl → real controlled carry
                cond_peaks_grasped.append(pk)
            if args.save_video:
                ok = "OK" if r["on_plate"] else ("GRASP" if r["grasped"] else "FAIL")
                save_rollout_video(r, f"{cond} r{ri:03d} {ok}",
                                   out_dir / "videos" / cond / f"rollout_{ri:03d}_{ok}.mp4")
        peaks_by_cond[cond] = cond_peaks
        peaks_grasped_by_cond[cond] = cond_peaks_grasped
        gmu = float(np.mean(cond_peaks_grasped)) if cond_peaks_grasped else float("nan")
        summary[cond] = (n_grasp, n_plate, float(np.mean(cond_peaks)), float(np.std(cond_peaks)), gmu)
        print(f"  grasp={n_grasp}/{n}  on_plate={n_plate}/{n}  "
              f"carry-peak(all)={np.mean(cond_peaks):.1f}cm  "
              f"carry-peak(grasped n={len(cond_peaks_grasped)})={gmu:.1f}cm")
    clear_steering(policy)
    env.close()

    # ── summary + separation ──
    # carry-peak(grasped) is the HONEST steering metric: it only counts rollouts
    # where the bowl was actually held, so a broken direction (e.g. caa_high that
    # fails to grasp) can't fake a "high carry" from flailing on failed rollouts.
    print("\n" + "=" * 78)
    print(f"OSC steering eval  ({args.policy_path})")
    print(f"{'condition':<15}{'grasp':>8}{'on_plate':>10}{'peak all':>11}{'peak grasped(n)':>18}")
    base = summary.get("none", (0, 0, 0, 0, float('nan')))[4]
    for cond, (g, p, mu, sd, gmu) in summary.items():
        ng = len(peaks_grasped_by_cond[cond])
        d = f"  Δ{gmu-base:+.1f}" if (cond != "none" and base == base and gmu == gmu) else ""
        print(f"{cond:<15}{g:>6}/{n}{p:>8}/{n}{mu:>11.1f}{gmu:>13.1f}(n={ng}){d}")
    print("=" * 78)

    # boxplot of carry-peak by condition — GRASPED rollouts only (honest metric)
    if len(peaks_grasped_by_cond) > 1:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = {"none": "#555", "caa_high": "#C0392B", "caa_low": "#E8A87C",
                "physical_high": "#1ABC9C", "physical_low": "#8E44AD"}
        labels = list(peaks_grasped_by_cond.keys())
        data = [peaks_grasped_by_cond[c] if peaks_grasped_by_cond[c] else [np.nan] for c in labels]
        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "white", "lw": 2})
        for patch, c in zip(bp["boxes"], labels):
            patch.set_facecolor(cols.get(c, "#888")); patch.set_alpha(0.85)
        # annotate each box with its sample count (grasped rollouts)
        ymax = np.nanmax([np.nanmax(d) for d in data])
        for i, c in enumerate(labels):
            ax.text(i + 1, ymax + 1, f"n={len(peaks_grasped_by_cond[c])}",
                    ha="center", fontsize=9, color="black")
        ax.set_xticklabels(labels, rotation=15, fontsize=10)
        ax.set_ylabel("Carry-phase peak EEF z (cm)  [grasped rollouts only]")
        ax.set_title(f"Height steering separation (task-valid)\n{Path(args.policy_path).name}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "steering_separation_boxplot.png", dpi=200)
        print(f"[✓] boxplot (grasped-only) → {out_dir/'steering_separation_boxplot.png'}")
    print(f"[✓] outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
