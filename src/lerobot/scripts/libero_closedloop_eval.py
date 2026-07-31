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
    setup_physical_caa,
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
    patch_controller_identity(env)   # env.reset() resets the controller — re-apply patch
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


def _bowl_keys(obs):
    return [k for k in obs if "bowl" in k.lower() and k.endswith("_pos") and "to_robot" not in k]


def run_hybrid_rollout(env, init_state, ref_jp, ref_grip, policy, preprocessor, postprocessor,
                       device, task, max_steps, arc_type, grasp_lift=0.05,
                       scripted_place=False, place_frac=0.80, plate_key=None, max_dq=0.12,
                       place_pool=None):
    """
    Hybrid rollout (matches how demos were collected): replay the recorded joint
    trajectory through the GRASP (guaranteed successful pick), then hand off to
    the POLICY + steering for the carry, then a SMOOTH descent to place the bowl.

    All commands are velocity-limited (max_dq rad/step) so the grasp→carry and
    carry→place transitions are smooth (no single-step joint jumps / jerkiness).
    The place is a joint-space interpolation from the ACTUAL carry height down to
    the plate — it never jumps up to the reference carry height, so the video
    always matches the measured carry-peak distribution.
    """
    from lerobot.utils.control_utils import predict_action

    def vstep(target7, grip):
        """Velocity-limited joint-position step."""
        dq = np.clip(np.asarray(target7) - obs["robot0_joint_pos"], -max_dq, max_dq)
        return env.step(np.concatenate([dq, [grip]]))

    policy.reset(); preprocessor.reset(); postprocessor.reset()
    env.reset()
    patch_controller_identity(env)   # env.reset() resets the controller — re-apply patch
    obs = env.set_init_state(init_state)
    bowls = _bowl_keys(obs)
    b0 = {b: float(obs[b][2]) for b in bowls}

    # ── Phase A: replay grasp EXACTLY until a bowl lifts grasp_lift ──
    # (The grasp reference moves faster than max_dq; it must be replayed exactly
    #  or the pick fails. This phase is smooth anyway — consecutive ref joints are
    #  ~0.003 rad apart. Smoothing is only needed at the handoff into the policy.)
    grasp_imgs = [obs["agentview_image"]]
    ho = None
    for t in range(len(ref_jp)):
        delta = ref_jp[t] - obs["robot0_joint_pos"]
        obs, _, done, _ = env.step(np.concatenate([delta, [ref_grip[t]]]))
        grasp_imgs.append(obs["agentview_image"])
        if any(obs[b][2] - b0[b] > grasp_lift for b in bowls):
            ho = t; break
    if ho is None:
        ho = int(0.6 * len(ref_jp))
    grasped_bowl = max(bowls, key=lambda b: obs[b][2] - b0[b])
    gripper_cmd = float(ref_grip[ho])

    # ── Phase B: policy carry with steering (velocity-limited for smoothness) ──
    eef_z, images, joints, bowl_z = [], [], [], []
    for t in range(max_steps):
        img = obs["agentview_image"]
        cur = obs["robot0_joint_pos"].astype(np.float64)
        state8 = np.concatenate([cur, [gripper_cmd]]).astype(np.float32)
        with torch.no_grad():
            action = predict_action(
                observation=build_obs(img, state8), policy=policy, device=device,
                preprocessor=preprocessor, postprocessor=postprocessor, use_amp=False, task=task,
            )
        act = (action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)).reshape(-1)
        gripper_cmd = float(np.clip(act[7], -1.0, 1.0))
        obs, _, done, _ = vstep(act[:7], gripper_cmd)
        eef_z.append(float(obs["robot0_eef_pos"][2]))
        bowl_z.append(float(obs[grasped_bowl][2]))
        images.append(img); joints.append(cur)
        if done:
            break

    eef = np.array(eef_z); bz = np.array(bowl_z)
    # bowl held during carry = it tracks the eef (grip offset stays small) and stayed lifted
    held = bool(len(bz) and np.median(eef - bz) < 0.20 and bz.max() - b0[grasped_bowl] > 0.03)

    # ── Phase C: place by replaying a demo descent tail. Pick the reference whose
    # carry-peak height is closest to the current carry height (low steering → low
    # demo's place path, which descends over the plate AT LOW height), and enter at
    # the matched-height point. This gives reliable placement (follows a real demo
    # place) while the video stays at the steered height (no jump up).
    place_imgs, place_eef = [], []
    if scripted_place and place_pool:
        cur_eef = float(obs["robot0_eef_pos"][2])
        # choose ref with carry-peak nearest the current carry height
        pj, pg, pe, ppk = min(place_pool, key=lambda r: abs(r[3] - cur_eef))
        peak_idx = int(np.argmax(pe))
        descent = pe[peak_idx:]
        entry = peak_idx + int(np.argmin(np.abs(descent - cur_eef)))
        for t in range(entry, len(pj)):
            obs, _, done, _ = vstep(pj[t], pg[t])
            place_imgs.append(obs["agentview_image"]); place_eef.append(float(obs["robot0_eef_pos"][2]))
            if done:
                break

    # Task success: bowl resting on the plate
    on_plate = False
    if plate_key is not None and plate_key in obs:
        fb = obs[grasped_bowl]; tp = obs[plate_key]
        dxy = float(np.linalg.norm(fb[:2] - tp[:2])); dz = float(fb[2] - tp[2])
        on_plate = bool(dxy < 0.06 and -0.02 < dz < 0.08)

    # Full task video: initial pose → grasp (Phase A) → steered carry (B) → place (C)
    all_imgs = grasp_imgs + images + place_imgs
    return {
        "eef_heights": eef,                 # policy carry phase (metric measured here)
        "ref_eef_z":   eef,
        "place_eef":   np.array(place_eef, dtype=np.float64),
        "pred_joints": np.array(joints, dtype=np.float64),
        "bowl_z":      bz,
        "bowl_held":   held,
        "on_plate":    on_plate,
        "handoff":     ho,
        "imgs":        np.array(all_imgs, dtype=np.uint8),
        "n_grasp":     len(grasp_imgs),     # phase boundaries for the video overlay
        "n_carry":     len(images),
        "T": len(eef_z),
        "arc_type": arc_type,
    }


def load_reference_grasps(ref_hdf5: str):
    """Load per-episode (joint_pos, gripper_cmd, eef_z) reference trajectories."""
    refs = []
    with h5py.File(ref_hdf5, "r") as f:
        for k in sorted(x for x in f.keys() if x.startswith("ep_")):
            refs.append((f[k]["joint_pos"][()], f[k]["actions"][:, 6],
                         f[k]["eef_pos"][:, 2]))
    return refs


# ── Video ─────────────────────────────────────────────────────────────────────

def save_rollout_video(result, condition, out_path: Path, fps=20.0):
    imgs = result["imgs"]
    eef  = result["eef_heights"] * 100
    T, H, W = len(imgs), 256, 256
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    # carry-peak over the measured (20-80%) window — matches the boxplot metric
    if len(eef) > 4:
        lo, hi = int(0.20 * len(eef)), int(0.80 * len(eef))
        peak = float(np.max(eef[lo:hi]))
    else:
        peak = float(np.max(eef)) if len(eef) else 0.0
    place_eef = result.get("place_eef", np.array([])) * 100
    # Phase boundaries: [0,n_grasp)=grasp(initial pose→pick), then carry, then place
    n_grasp = int(result.get("n_grasp", 0))
    n_carry = int(result.get("n_carry", len(eef)))
    for t in range(T):
        frame = cv2.cvtColor(imgs[t], cv2.COLOR_RGB2BGR)
        if t < n_grasp:
            label, color, ztxt = "GRASP", (0, 180, 255), ""
        elif t < n_grasp + n_carry:
            ci = t - n_grasp
            label, color = "CARRY (steered)", (0, 255, 0)
            ztxt = f"  z={eef[ci]:.1f}cm" if ci < len(eef) else ""
        else:
            pi = t - n_grasp - n_carry
            label, color = "PLACE", (0, 200, 255)
            ztxt = f"  z={place_eef[pi]:.1f}cm" if pi < len(place_eef) else ""
        cv2.putText(frame, f"[{label}]{ztxt}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{condition} carry-peak={peak:.1f}cm", (6, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def plot_conditions_boxplot(traces_by_cond, out_path):
    """Standalone carry-peak boxplot for the steering conditions ONLY (no dataset ref)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"none": "#555555", "caa_high": "#C0392B", "caa_low": "#E8A87C",
              "keyword_high": "#2B8BE0", "keyword_low": "#E0B82B",
              "physical_high": "#1ABC9C", "physical_low": "#8E44AD"}
    LABELS = {"none": "Baseline", "caa_high": "Steer-high", "caa_low": "Steer-low",
              "keyword_high": "Steer-high", "keyword_low": "Steer-low",
              "physical_high": "Steer-high", "physical_low": "Steer-low"}

    box_data, box_labels, box_colors = [], [], []
    for cond, traces in traces_by_cond.items():
        peaks = []
        for tr in traces:
            valid = np.asarray(tr)[np.isfinite(tr)] * 100
            if len(valid) > 4:
                lo, hi = int(0.20 * len(valid)), int(0.80 * len(valid))
                peaks.append(float(np.max(valid[lo:hi])))
        if peaks:
            box_data.append(peaks)
            box_labels.append(LABELS.get(cond, cond))
            box_colors.append(COLORS.get(cond, "black"))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True,
                        medianprops={"color": "white", "lw": 2},
                        whiskerprops={"lw": 1.2}, capprops={"lw": 1.2})
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.8)
        ax.set_xticklabels(box_labels, fontsize=11)
    ax.set_ylabel("Carry-phase peak EEF z (cm)", fontsize=12)
    ax.set_title("Carry-Peak Height by Steering Condition", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Conditions-only boxplot: {out_path}")


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
    ap.add_argument("--physical-vectors", default=None,
                    help="Path to *_vectors.pt from libero_find_physical_neurons.py (sparse-CAA)")
    ap.add_argument("--physical-alpha", type=float, default=6.0)
    ap.add_argument("--physical-top-k", type=int, default=None,
                    help="Keep top-k neurons per layer by |contrast| (None = dense VLM-space CAA)")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--video-rollouts", type=int, nargs="+", default=[],
                    help="Which rollout indices to save as video (default: empty = ALL rollouts)")
    ap.add_argument("--max-dq", type=float, default=0.12,
                    help="Velocity limit (rad/step) on joint commands; smooths stage transitions")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--out-dir", default="outputs/libero_closedloop")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset-high-hdf5", default=None,
                    help="HDF5 of high-arc demos; shown as dataset-HIGH reference boxplot")
    ap.add_argument("--dataset-low-hdf5", default=None,
                    help="HDF5 of low-arc demos; shown as dataset-LOW reference boxplot")
    ap.add_argument("--hybrid", action="store_true",
                    help="Replay recorded grasp (guaranteed pick), then policy+steering carry. "
                         "Isolates steering effect on carry height with a real bowl in gripper.")
    ap.add_argument("--hybrid-ref-hdf5",
                    default="/home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5",
                    help="HDF5 providing per-episode grasp joint trajectories for hybrid replay")
    ap.add_argument("--place-ref-low-hdf5",
                    default="/home/bruce/datasets/libero_height_demos/libero_spatial/low/task_00.hdf5",
                    help="Low-demo HDF5 added to the place-ref pool so low-steered carries "
                         "descend via a low place path (matched by carry height)")
    ap.add_argument("--grasp-lift", type=float, default=0.05,
                    help="Bowl lift (m) that triggers handoff from grasp-replay to policy")
    ap.add_argument("--scripted-place", action="store_true",
                    help="After policy carry, replay the dataset place tail so the TASK COMPLETES "
                         "(bowl onto plate). Only the carry height stays policy+steering controlled.")
    ap.add_argument("--place-frac", type=float, default=0.80,
                    help="Fraction into the reference trajectory where the scripted place tail begins")
    ap.add_argument("--carry-steps", type=int, default=None,
                    help="Number of policy-controlled carry steps before the scripted place")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="Override policy action-chunk execution length (default from config, 50)")
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

    # Hybrid mode: reference grasp trajectories (replayed through the pick)
    ref_grasps = None
    place_ref_pool = None   # pool of place-descent refs, selected per rollout by height
    if args.hybrid:
        ref_grasps = load_reference_grasps(args.hybrid_ref_hdf5)
        print(f"HYBRID mode: loaded {len(ref_grasps)} grasp references from "
              f"{Path(args.hybrid_ref_hdf5).name} (replay grasp → policy carry)")
        # Place references: high + (optional) low demos, so the place descent can
        # match the steered carry height (low steering → low place path).
        pool = list(ref_grasps)
        place_lo = args.place_ref_low_hdf5 or args.dataset_low_hdf5
        if place_lo and Path(place_lo).exists():
            pool += load_reference_grasps(place_lo)
            print(f"  place-ref pool += low demos from {Path(place_lo).name}")
        # Precompute each ref's carry-peak height for matching
        place_ref_pool = [(jp, grip, np.asarray(eef), float(np.max(eef)))
                          for (jp, grip, eef) in pool]

    # ── Policy ────────────────────────────────────────────────────────────────
    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy_cfg = policy.config
    policy_cfg.device = str(device)
    if args.n_action_steps is not None:
        policy_cfg.n_action_steps = args.n_action_steps
        print(f"n_action_steps set to {args.n_action_steps}")
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=args.policy_path,
    )

    # ── Env ───────────────────────────────────────────────────────────────────
    print("Creating LIBERO env (JOINT_POSITION control) ...")
    env = make_env(bddl_path)
    obs0 = env.reset()
    patch_controller_identity(env)
    plate_key = next((k for k in obs0 if "plate" in k.lower()
                      and k.endswith("_pos") and "to_robot" not in k), None)
    if args.scripted_place:
        print(f"SCRIPTED-PLACE mode: place_frac={args.place_frac}, plate_key={plate_key} "
              f"(task completes; success = bowl on plate)")

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
        elif cond == "physical_high":
            setup_physical_caa(policy, args.physical_vectors, "high",
                               alpha=args.physical_alpha, top_k=args.physical_top_k)
        elif cond == "physical_low":
            setup_physical_caa(policy, args.physical_vectors, "low",
                               alpha=args.physical_alpha, top_k=args.physical_top_k)
        else:
            print(f"  [skip] unknown condition '{cond}'"); continue

        cond_traces, cond_metrics = [], []
        n_held = 0
        for ri in tqdm(range(args.n_rollouts), desc=f"  {cond}"):
            init = init_states[ri % len(init_states)]
            if args.hybrid:
                ref_jp, ref_grip, _ = ref_grasps[ri % len(ref_grasps)]
                result = run_hybrid_rollout(
                    env, init, ref_jp, ref_grip, policy, preprocessor, postprocessor,
                    device, args.task, args.carry_steps or args.max_steps, arc_type=cond,
                    grasp_lift=args.grasp_lift,
                    scripted_place=args.scripted_place, place_frac=args.place_frac,
                    plate_key=plate_key, max_dq=args.max_dq, place_pool=place_ref_pool,
                )
            else:
                result = run_closedloop_rollout(
                    env, init, policy, preprocessor, postprocessor,
                    device, args.task, args.max_steps, arc_type=cond,
                )
            m = rollout_metrics(ri, result, cond)
            m["bowl_held"] = int(result.get("bowl_held", 0))
            m["on_plate"]  = int(result.get("on_plate", 0))
            n_held += m["on_plate"] if args.scripted_place else m["bowl_held"]
            cond_traces.append(result["eef_heights"])
            cond_metrics.append(m)
            # Save video for the requested rollouts (empty list / [-1] = ALL rollouts).
            save_all = args.save_video and (not args.video_rollouts or -1 in args.video_rollouts)
            if args.save_video and (save_all or ri in args.video_rollouts):
                peak_cm = int(round(m.get("carry_peak_cm", 0)))
                ok = "OK" if m["on_plate"] else "FAIL"
                save_rollout_video(
                    result, cond,
                    out_dir / "videos" / cond / f"rollout_{ri:03d}_peak{peak_cm}cm_{ok}.mp4",
                    fps=args.fps)

        all_traces[cond] = cond_traces
        all_metrics[cond] = cond_metrics
        save_csv(out_dir / f"rollout_metrics_{cond}.csv", cond_metrics)

        agg = aggregate(cond_metrics, "carry_peak_cm")
        if agg:
            label = "on_plate" if args.scripted_place else "bowl_held"
            hb = f"  {label}={n_held}/{len(cond_metrics)}" if args.hybrid else ""
            print(f"\n  carry_peak: mean={agg['mean']:.2f} ± {agg['std']:.2f} cm  "
                  f"CI95=[{agg['ci95_lo']:.2f},{agg['ci95_hi']:.2f}]  n={agg['n']}{hb}")

        # raw traces
        max_T = max(len(t) for t in cond_traces)
        arr = np.full((len(cond_traces), max_T), np.nan)
        for i, t in enumerate(cond_traces):
            arr[i, :len(t)] = t
        np.savez_compressed(out_dir / f"eef_traces_{cond}.npz", traces_m=arr, traces_cm=arr*100)

    env.close()

    # ── Dataset reference traces (ground-truth demo high/low EEF-z) ───────────
    def _load_dataset_eef(path):
        traces = []
        with h5py.File(path, "r") as f:
            for k in sorted(x for x in f.keys() if x.startswith("ep_")):
                traces.append(f[k]["eef_pos"][:, 2].astype(np.float64))  # metres
        max_T = max(len(t) for t in traces)
        arr = np.full((len(traces), max_T), np.nan)
        for i, t in enumerate(traces):
            arr[i, :len(t)] = t
        return arr

    ref_by_arc = None
    if args.dataset_high_hdf5 or args.dataset_low_hdf5:
        ref_by_arc = {}
        if args.dataset_high_hdf5:
            ref_by_arc["high"] = _load_dataset_eef(args.dataset_high_hdf5)
        if args.dataset_low_hdf5:
            ref_by_arc["low"] = _load_dataset_eef(args.dataset_low_hdf5)

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
                        ref_traces=ref_by_arc, metrics_by_cond=all_metrics)
        # Second figure: carry-peak boxplot with ONLY the conditions (no dataset ref)
        plot_conditions_boxplot(trace_arrays, out_dir / "carry_peak_boxplot_conditions.png")

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
