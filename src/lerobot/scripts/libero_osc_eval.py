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
import os
import sys
import time
from pathlib import Path

# Must be set before the CUDA context / first cuBLAS call for deterministic matmuls.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


# Obs key layout: "custom" = the retrained OSC model (camera1/2/3 + empty_camera_0,
# empty_cameras=1); "standard" = the paper checkpoint HuggingFaceVLA/smolvla_libero
# (observation.images.image=agentview, image2=wrist, state8; empty_cameras=0).
INPUT_FORMAT = "custom"


def build_obs(agent_hwc: np.ndarray, wrist_hwc: np.ndarray, state8: np.ndarray,
              front_hwc: np.ndarray | None = None) -> dict:
    if INPUT_FORMAT == "standard":
        return {
            "observation.images.image":  agent_hwc,
            "observation.images.image2": wrist_hwc,
            "observation.state":         state8.astype(np.float32),
        }
    return {
        "observation.images.camera1":        agent_hwc,
        "observation.images.camera2":        wrist_hwc,
        "observation.images.camera3":        front_hwc if front_hwc is not None else _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


def make_env(bddl_path: Path, third_camera: str | None = None):
    """Default LIBERO controller is OSC_POSE — exactly what the OSC dataset was collected under."""
    from libero.libero.envs import OffScreenRenderEnv
    cams = ["agentview", "robot0_eye_in_hand"]
    if third_camera:
        cams.append(third_camera)
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=cams,
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


def _target_keys(obs, keywords):
    """Generic target-object position key finder. keywords is a list of strings
    that must ALL appear in the key name (e.g. ['alphabet','soup'] or ['bowl']).
    Skips robot/to_robot keys."""
    return [k for k in obs
            if all(w in k.lower() for w in keywords)
            and k.endswith("_pos") and "to_robot" not in k
            and "robot0" not in k]


def run_rollout(env, init_state, policy, preprocessor, postprocessor,
                device, task, max_steps, plate_key=None, phase_gate=False,
                third_camera: str | None = None,
                target_keywords: list | None = None,
                container_mode: str = "plate"):
    from lerobot.utils.control_utils import predict_action

    policy.reset(); preprocessor.reset(); postprocessor.reset()
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):  # settle (zero OSC delta, gripper open)
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))

    kws   = target_keywords if target_keywords else ["bowl"]
    bowls = _target_keys(obs, kws) or _bowl_keys(obs)   # fallback to bowl for libero_spatial
    b0 = {b: float(obs[b][2]) for b in bowls}
    bowl_peak = dict(b0)
    eef_z, eef_xyz, images = [], [], []
    # Paper-faithful steering metric = the feature computed from the model's OWN
    # predicted action each step (DiMaS steers this, not realized EEF motion).
    act_speed, act_dz = [], []

    for t in range(max_steps):
        # Flip upright to match the collector (LIBERO renders vertically flipped).
        agent = np.ascontiguousarray(obs["agentview_image"][::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
        front = (np.ascontiguousarray(obs[f"{third_camera}_image"][::-1])
                 if third_camera else None)
        # Phase-gated steering: only steer during CARRY (bowl lifted, not yet over
        # the plate) so grasp + place run unperturbed. When off, hooks pass through.
        if phase_gate:
            lifted_now = any((float(obs[b][2]) - b0[b]) > 0.03 for b in bowls)
            over_plate = False
            if plate_key is not None and plate_key in obs and bowls:
                dmin = min(float(np.linalg.norm(
                    np.asarray(obs[b][:2], dtype=np.float64)
                    - np.asarray(obs[plate_key][:2], dtype=np.float64))) for b in bowls)
                over_plate = dmin < 0.08
            policy._caa_gate = bool(lifted_now and not over_plate)
        else:
            policy._caa_gate = True
        with torch.no_grad():
            action = predict_action(
                observation=build_obs(agent, wrist, eef_state(obs), front),
                policy=policy, device=device,
                preprocessor=preprocessor, postprocessor=postprocessor,
                use_amp=False, task=task,
            )
        act = (action.detach().cpu().numpy() if torch.is_tensor(action)
               else np.asarray(action)).reshape(-1)[:7].astype(np.float32)
        act_speed.append(float(np.linalg.norm(act[:3])))   # DiMaS 'speed' feature
        act_dz.append(float(act[2]))                        # DiMaS 'height' feature
        obs, _, done, _ = env.step(act)
        eef_z.append(float(obs["robot0_eef_pos"][2]))
        eef_xyz.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy())
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
        if container_mode == "basket":
            # Object is inside the basket: within 15cm XY radius, 0–20cm above basket centre
            on_plate = bool(dxy < 0.15 and -0.05 < dz < 0.20)
        else:
            on_plate = bool(dxy < 0.06 and -0.02 < dz < 0.08 and fb[2] > 0.88)

    return {
        "eef_heights": np.array(eef_z, dtype=np.float64),
        "eef_xyz": np.array(eef_xyz, dtype=np.float64),
        "imgs": np.array(images, dtype=np.uint8),
        "grasped": grasped, "on_plate": on_plate, "T": len(eef_z),
        "act_speed": np.array(act_speed, dtype=np.float64),
        "act_dz": np.array(act_dz, dtype=np.float64),
    }


def save_rollout_video(result, tag, out_path: Path, fps=20.0, feature="height"):
    """Overlay the steered feature per frame: height run -> EEF z (cm); speed run
    -> instantaneous EEF displacement per step (cm), the paper's speed metric."""
    imgs = result["imgs"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if feature == "speed":
        p = result["eef_xyz"] * 100
        step = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.array([])
        overlay = [f"speed={step[t]:.2f}cm/step" if t < len(step) else "" for t in range(len(imgs))]
    else:
        eef = result["eef_heights"] * 100
        overlay = [f"z={eef[t]:.1f}cm" if t < len(eef) else "" for t in range(len(imgs))]
    w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 256))
    for t in range(len(imgs)):
        f = cv2.cvtColor(imgs[t], cv2.COLOR_RGB2BGR)
        cv2.putText(f, f"{tag}  {overlay[t]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
    ap.add_argument("--n-rollouts", type=int, default=20,
                    help="DEPRECATED when --n-success is set; used as max-attempts fallback")
    ap.add_argument("--n-success", type=int, default=None,
                    help="run until this many SUCCESSFUL (on-plate) rollouts per condition; "
                         "all conditions end with the same N for a fair boxplot")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="hard cap on attempts per condition (default: n-success*6 or n-rollouts)")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--save-success-only", action="store_true",
                    help="save videos for SUCCESSFUL (task-completed / on-plate) rollouts only")
    ap.add_argument("--out-dir", default="outputs/libero_osc_baseline")
    ap.add_argument("--device", default=None)
    # ── steering ──
    ap.add_argument("--conditions", nargs="+", default=["none"],
                    help="none, caa_high, caa_low, keyword_high, keyword_low, physical_high, physical_low")
    ap.add_argument("--caa-path", default=None, help="expert-space CAA vectors (setup_caa)")
    ap.add_argument("--caa-alpha", type=float, default=6.0)
    ap.add_argument("--caa-layer-lo", type=int, default=None,
                    help="steer only expert layers [layer-lo, layer-hi) — middle layers "
                         "steer cleanly; steering ALL layers unnormalized swamps the policy")
    ap.add_argument("--caa-layer-hi", type=int, default=None)
    ap.add_argument("--caa-exclude-last", action="store_true",
                    help="skip the final (degenerate high-norm) expert layer")
    ap.add_argument("--caa-normalize", action="store_true",
                    help="unit-normalize each layer vector so alpha is a portable "
                         "fraction of the activation scale (recommended)")
    ap.add_argument("--caa-phase-gate", action="store_true",
                    help="steer ONLY during the carry sub-phase (bowl lifted, not yet "
                         "over plate) so grasp+place run unperturbed — lets a strong "
                         "alpha raise carry height without breaking task success")
    ap.add_argument("--neurons-json", default=None,
                    help="VLM keyword-neuron map JSON (libero_find_height_neurons.py output). "
                         "Neurons are in the frozen VLM backbone — same file works across all SmolVLA models.")
    ap.add_argument("--keyword-alpha", type=float, default=8.0)
    ap.add_argument("--keyword-top-n", type=int, default=None,
                    help="keep only top-N neurons per concept (None = all)")
    ap.add_argument("--keyword-mode", choices=["add", "set"], default="add")
    ap.add_argument("--physical-vectors", default=None, help="VLM-space contrast vectors (setup_physical_caa)")
    ap.add_argument("--physical-alpha", type=float, default=6.0)
    ap.add_argument("--physical-top-k", type=int, default=None)
    ap.add_argument("--dimas-path", default=None, help="DiMaS artifact (libero_build_dimas.py output)")
    ap.add_argument("--dimas-alpha", type=float, default=0.5,
                    help="DiMaS interpolation knob in [0,1] (0=off, 1=full transport)")
    ap.add_argument("--dimas-no-gate", action="store_true",
                    help="disable the DiMaS feature-absent gate (steer every rep)")
    ap.add_argument("--coast-path",  default=None, help="COAST artifact (libero_compute_coast.py output)")
    ap.add_argument("--coast-beta",  type=float, default=0.5,
                    help="COAST gate strength β ∈ [0,1]: 0=no steering, 1=full projection (default: 0.5)")
    ap.add_argument("--coast-layer-lo", type=int, default=None,
                    help="steer only expert layers [lo, hi) with COAST (None = all fitted)")
    ap.add_argument("--coast-layer-hi", type=int, default=None)
    ap.add_argument("--input-format", choices=["custom", "standard"], default="custom",
                    help="'standard' = paper checkpoint HuggingFaceVLA/smolvla_libero "
                         "(observation.images.image/image2 + state8); 'custom' = retrained OSC model.")
    ap.add_argument("--third-camera", default=None,
                    help="Third camera to render and pass to the model. Use 'sideview' for "
                         "three-cam models (svla_franka_pick_n_place_vla_steering_libero_three_cams_*). "
                         "None (default) passes a blank frame for camera3 (two-cam models).")
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device
    from libero.libero import benchmark
    from lerobot.scripts.libero_eval_steering import (
        setup_caa, setup_physical_caa, setup_keyword_neurons, setup_dimas,
        setup_coast, clear_steering, enable_determinism)

    global INPUT_FORMAT
    INPUT_FORMAT = args.input_format
    enable_determinism()   # reproducible, history-independent rollouts

    # Detect the steered behavioral feature (speed vs height) so plots, labels and
    # video overlays match what is actually being intervened on (DiMaS, arXiv 2607.14280).
    steer_feature = "height"
    if args.dimas_path and any(c.startswith("dimas") for c in args.conditions):
        try:
            steer_feature = torch.load(args.dimas_path, map_location="cpu",
                                       weights_only=False).get("feature", "height")
        except Exception:
            pass
    _DISP = {"speed": {"dimas_high": "dimas_faster", "dimas_low": "dimas_slower"},
             "height": {"dimas_high": "dimas_higher", "dimas_low": "dimas_lower"}}

    def disp_label(cond):
        return _DISP.get(steer_feature, {}).get(cond, cond)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  (deterministic algorithms ON)")

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

    # ── Derive target object + container from suite and BDDL filename ─────────
    import re as _re
    if args.suite == "libero_object":
        container_mode = "basket"
        m = _re.search(r'pick_up_the_(.+?)_and_place', fname)
        target_name = m.group(1) if m else None   # e.g. "alphabet_soup"
        target_keywords = target_name.split("_") if target_name else ["bowl"]
        print(f"libero_object: target={target_name}  keywords={target_keywords}")
    else:
        container_mode = "plate"
        target_keywords = None   # falls back to _bowl_keys inside run_rollout

    env = make_env(bddl_path, third_camera=args.third_camera)
    obs0 = env.reset()
    if container_mode == "basket":
        plate_key = next((k for k in obs0 if "basket" in k.lower()
                          and k.endswith("_pos") and "to_robot" not in k), None)
    else:
        plate_key = next((k for k in obs0 if "plate" in k.lower()
                          and k.endswith("_pos") and "to_robot" not in k), None)
    print(f"container_key={plate_key}  mode={container_mode}")

    def carry_peak_cm(eef_m):
        e = np.asarray(eef_m) * 100
        if len(e) > 4:
            lo, hi = int(0.20 * len(e)), int(0.80 * len(e))
            return float(np.max(e[lo:hi]))
        return float(np.max(e)) if len(e) else 0.0

    def carry_mean_cm(eef_m):
        # MEAN carry-phase EEF-z — the metric that separates dataset high (114.7cm)
        # from low (101.0cm). Peak is task-pinned (both ~117cm to reach the plate),
        # so mean-carry is the honest height-steering metric (see DiMaS notes).
        e = np.asarray(eef_m) * 100
        if len(e) > 4:
            lo, hi = int(0.20 * len(e)), int(0.80 * len(e))
            return float(np.mean(e[lo:hi]))
        return float(np.mean(e)) if len(e) else 0.0

    def carry_speed_cmps(eef_xyz):
        # MEAN carry-phase EEF speed (cm/step) = ||Δpos|| — the closed-loop analog of
        # DiMaS's 'speed' feature (‖Δxyz‖). Reliable BIDIRECTIONAL steer feature on
        # SmolVLA (unlike height, which is task-pinned). Multiply by control-Hz for cm/s.
        p = np.asarray(eef_xyz, dtype=np.float64) * 100
        if len(p) > 5:
            d = np.linalg.norm(np.diff(p, axis=0), axis=1)   # cm per control step
            lo, hi = int(0.20 * len(d)), int(0.80 * len(d))
            return float(np.mean(d[lo:hi]))
        return float("nan")

    def apply_condition(cond):
        if cond == "none":
            clear_steering(policy)
        elif cond == "caa_high":
            setup_caa(policy, args.caa_path, alpha=+args.caa_alpha,
                      layer_lo=args.caa_layer_lo, layer_hi=args.caa_layer_hi,
                      exclude_last=args.caa_exclude_last, normalize=args.caa_normalize)
        elif cond == "caa_low":
            setup_caa(policy, args.caa_path, alpha=-args.caa_alpha,
                      layer_lo=args.caa_layer_lo, layer_hi=args.caa_layer_hi,
                      exclude_last=args.caa_exclude_last, normalize=args.caa_normalize)
        elif cond == "keyword_high":
            if not args.neurons_json:
                raise ValueError("--neurons-json required for keyword_high/low")
            setup_keyword_neurons(policy, args.neurons_json, "high",
                                  alpha=args.keyword_alpha, top_n=args.keyword_top_n,
                                  mode=args.keyword_mode)
        elif cond == "keyword_low":
            if not args.neurons_json:
                raise ValueError("--neurons-json required for keyword_high/low")
            setup_keyword_neurons(policy, args.neurons_json, "low",
                                  alpha=args.keyword_alpha, top_n=args.keyword_top_n,
                                  mode=args.keyword_mode)
        elif cond == "physical_high":
            setup_physical_caa(policy, args.physical_vectors, "high",
                               alpha=args.physical_alpha, top_k=args.physical_top_k)
        elif cond == "physical_low":
            setup_physical_caa(policy, args.physical_vectors, "low",
                               alpha=args.physical_alpha, top_k=args.physical_top_k)
        elif cond == "dimas_high":
            setup_dimas(policy, args.dimas_path, "high",
                        alpha=args.dimas_alpha, gate=not args.dimas_no_gate)
        elif cond == "dimas_low":
            setup_dimas(policy, args.dimas_path, "low",
                        alpha=args.dimas_alpha, gate=not args.dimas_no_gate)
        elif cond in ("coast_high", "coast_success", "coast_pos"):
            if not args.coast_path:
                raise ValueError("--coast-path required for coast conditions")
            setup_coast(policy, args.coast_path, "pos", beta=args.coast_beta,
                        layer_lo=args.coast_layer_lo, layer_hi=args.coast_layer_hi)
        elif cond in ("coast_low", "coast_failure", "coast_neg"):
            if not args.coast_path:
                raise ValueError("--coast-path required for coast conditions")
            setup_coast(policy, args.coast_path, "neg", beta=args.coast_beta,
                        layer_lo=args.coast_layer_lo, layer_hi=args.coast_layer_hi)
        else:
            raise ValueError(f"unknown condition {cond}")

    # ── per-condition loop: run until n_success successes or max_attempts ──
    # When --n-success is set, every condition collects exactly that many
    # successful (on-plate) rollouts, giving equal-N boxes in the boxplot.
    n_success_target = args.n_success         # None → fallback to fixed-n mode
    max_attempts = args.max_attempts or (
        n_success_target * 6 if n_success_target else args.n_rollouts
    )

    peaks_success_by_cond, speed_all_by_cond, summary = {}, {}, {}
    for cond in args.conditions:
        print(f"\n{'='*56}\nCondition: {cond}\n{'='*56}")
        apply_condition(cond)
        n_grasp = n_plate = n_attempts = 0
        cond_peaks_all, cond_peaks_success = [], []
        cond_means_all, cond_means_success = [], []
        cond_speed_all, cond_speed_success = [], []
        cond_actsp_all, cond_actdz_all = [], []   # predicted-action feature (paper metric)
        ri = 0
        target = n_success_target if n_success_target else None

        pbar = tqdm(total=target or max_attempts,
                    desc=f"  {cond} {'successes' if target else 'rollouts'}",
                    unit="ok" if target else "roll", dynamic_ncols=True)
        t_cond_start = time.time()
        t_rollout_times = []

        while n_attempts < max_attempts:
            if target is not None and len(cond_peaks_success) >= target:
                break
            init = init_states[ri % len(init_states)]
            t_roll = time.time()
            r = run_rollout(env, init, policy, preprocessor, postprocessor,
                            device, args.task, args.max_steps, plate_key,
                            phase_gate=args.caa_phase_gate,
                            third_camera=args.third_camera,
                            target_keywords=target_keywords,
                            container_mode=container_mode)
            t_rollout_times.append(time.time() - t_roll)
            n_attempts += 1; ri += 1
            n_grasp += int(r["grasped"]); n_plate += int(r["on_plate"])
            pk = carry_peak_cm(r["eef_heights"])
            mk = carry_mean_cm(r["eef_heights"])
            sp = carry_speed_cmps(r["eef_xyz"])
            cond_peaks_all.append(pk)
            cond_means_all.append(mk)
            if sp == sp:
                cond_speed_all.append(sp)
            if len(r["act_speed"]):
                cond_actsp_all.append(float(np.mean(r["act_speed"])))
                cond_actdz_all.append(float(np.mean(r["act_dz"])))

            ok_str  = "OK" if r["on_plate"] else ("GRASP" if r["grasped"] else "FAIL")
            avg_t   = sum(t_rollout_times) / len(t_rollout_times)
            n_ok    = len(cond_peaks_success) + int(r["on_plate"])
            sr      = 100.0 * n_plate / n_attempts
            # ETA estimate: remaining items / success rate (for success mode), or remaining × avg_t
            if target is not None:
                remaining = target - n_ok
                eta_s = (remaining / max(n_plate / n_attempts, 0.05)) * avg_t if n_plate > 0 \
                        else (max_attempts - n_attempts) * avg_t
            else:
                eta_s = (max_attempts - n_attempts) * avg_t
            pbar.set_postfix({
                "att": n_attempts,
                "ok": f"{n_ok}" + (f"/{target}" if target else ""),
                "sr%": f"{sr:.0f}",
                "z": f"{pk:.0f}cm",
                "t/roll": f"{avg_t:.0f}s",
                "ETA": f"{eta_s/60:.1f}m",
                "last": ok_str,
            })

            if r["on_plate"]:
                cond_peaks_success.append(pk)
                cond_means_success.append(mk)
                if sp == sp:
                    cond_speed_success.append(sp)
                pbar.update(1)
                if args.save_video:
                    save_rollout_video(
                        r, f"{disp_label(cond)} r{ri:03d} OK",
                        out_dir / "videos" / disp_label(cond) / f"rollout_{ri:03d}_OK.mp4",
                        feature=steer_feature)
            else:
                if target is None:
                    pbar.update(1)
                if args.save_video and not args.save_success_only:
                    save_rollout_video(
                        r, f"{disp_label(cond)} r{ri:03d} {ok_str}",
                        out_dir / "videos" / disp_label(cond) / f"rollout_{ri:03d}_{ok_str}.mp4",
                        feature=steer_feature)
        pbar.close()
        elapsed_cond = time.time() - t_cond_start
        print(f"  Condition time: {elapsed_cond/60:.1f} min  "
              f"({sum(t_rollout_times)/len(t_rollout_times):.0f}s/rollout avg)")

        peaks_success_by_cond[cond] = cond_peaks_success
        speed_all_by_cond[cond] = cond_speed_all   # realized EEF displacement/step, ALL rollouts
        smu = float(np.mean(cond_peaks_success)) if cond_peaks_success else float("nan")
        mu_all = float(np.mean(cond_peaks_all)) if cond_peaks_all else float("nan")
        sd_all = float(np.std(cond_peaks_all)) if cond_peaks_all else float("nan")
        # MEAN-carry (height-steering metric) + MEAN-speed (speed-steering metric)
        cmean_all = float(np.mean(cond_means_all)) if cond_means_all else float("nan")
        cmean_succ = float(np.mean(cond_means_success)) if cond_means_success else float("nan")
        cspeed_all = float(np.mean(cond_speed_all)) if cond_speed_all else float("nan")
        cspeed_succ = float(np.mean(cond_speed_success)) if cond_speed_success else float("nan")
        # PREDICTED-ACTION feature means (what DiMaS actually steers — paper metric)
        pa_speed = float(np.mean(cond_actsp_all)) if cond_actsp_all else float("nan")
        pa_dz = float(np.mean(cond_actdz_all)) if cond_actdz_all else float("nan")
        summary[cond] = (n_grasp, n_plate, n_attempts, mu_all, sd_all, smu,
                         cmean_all, cmean_succ, cspeed_all, cspeed_succ, pa_speed, pa_dz)
        print(f"  attempts={n_attempts}  grasp={n_grasp}/{n_attempts} ({100*n_grasp/n_attempts:.0f}%)"
              f"  on_plate={n_plate}/{n_attempts} ({100*n_plate/n_attempts:.0f}%)"
              f"  carry-MEAN(succ n={len(cond_means_success)})={cmean_succ:.1f}cm"
              f"  carry-SPEED(all)={cspeed_all:.2f}cm/step  SPEED(succ)={cspeed_succ:.2f}")
        print(f"  PRED-ACTION(all): speed(||Δxyz||)={pa_speed:.4f}   dz={pa_dz:+.4f}   "
              f"<- DiMaS-steered feature")
    clear_steering(policy)
    env.close()

    # ── summary + separation ──
    # carry-peak(successful) is the HONEST steering metric: it only counts rollouts
    # that COMPLETED the task (bowl on plate), so a broken direction can't fake a
    # separation from failing/flailing rollouts.
    print("\n" + "=" * 104)
    print(f"OSC steering eval  ({args.policy_path})")
    print("Height metric = MEAN carry-z (separates dataset high 114.7cm / low 101.0cm); "
          "peak is task-pinned to plate (~117cm)")
    print(f"{'condition':<14}{'attempts':>9}{'grasp%':>8}{'success%':>10}"
          f"{'cMEAN(all)':>12}{'Δheight':>9}{'SPEED(all)':>12}{'Δspeed%':>9}")
    base = summary.get("none", (0,)*12)[6]        # baseline mean-carry (all)
    base_sp = summary.get("none", (0,)*12)[8]     # baseline mean-speed (all)
    for cond, vals in summary.items():
        (g, p, nattempts, mu, sd, smu, cmean_all, cmean_succ, csp_all, csp_succ) = vals[:10]
        dh = f"{cmean_all-base:+.1f}" if (cond != "none" and base == base and cmean_all == cmean_all) else "—"
        dsp = f"{100*(csp_all-base_sp)/base_sp:+.0f}%" if (cond != "none" and base_sp == base_sp and base_sp and csp_all == csp_all) else "—"
        print(f"{cond:<14}{nattempts:>9}{100*g/nattempts:>6.0f}%{100*p/nattempts:>9.0f}%"
              f"{cmean_all:>12.1f}{dh:>9}{csp_all:>12.2f}{dsp:>9}")
    print("(height in cm: dataset low=101.0 / high=114.7; speed in cm/control-step. "
          "For SPEED steering read the SPEED/Δspeed cols; for height read cMEAN/Δheight.)")
    # PREDICTED-ACTION feature table (the quantity DiMaS steers, per the paper)
    base_pas = summary.get("none", (float('nan'),)*12)[10]
    base_pdz = summary.get("none", (float('nan'),)*12)[11]
    print(f"\n{'PRED-ACTION':<14}{'speed||Δxyz||':>16}{'Δspeed%':>10}{'dz(signed)':>14}{'Δdz':>10}")
    for cond, vals in summary.items():
        pas, pdz = vals[10], vals[11]
        dps = f"{100*(pas-base_pas)/base_pas:+.0f}%" if (cond != "none" and base_pas == base_pas and base_pas) else "—"
        ddz = f"{pdz-base_pdz:+.4f}" if (cond != "none" and base_pdz == base_pdz) else "—"
        print(f"{cond:<14}{pas:>16.4f}{dps:>10}{pdz:>+14.4f}{ddz:>10}")
    print("=" * 104)

    # ── save JSON summary (always) ──────────────────────────────────────────────
    import json as _json
    base = summary.get("none", (0,)*12)[6]
    base_sp = summary.get("none", (0,)*12)[8]
    base_pas = summary.get("none", (float('nan'),)*12)[10]
    base_pdz = summary.get("none", (float('nan'),)*12)[11]
    summary_rows = []
    for cond, vals in summary.items():
        (g, p, nattempts, mu, sd, smu, cmean_all, cmean_succ, csp_all, csp_succ) = vals[:10]
        pas, pdz = vals[10], vals[11]
        summary_rows.append({
            "condition":       cond,
            "attempts":        nattempts,
            "grasp_pct":       round(100 * g / nattempts, 1),
            "success_pct":     round(100 * p / nattempts, 1),
            "carry_mean_all_cm":    round(cmean_all, 2),
            "carry_mean_succ_cm":   round(cmean_succ, 2) if cmean_succ == cmean_succ else None,
            "delta_height_cm":      round(cmean_all - base, 2) if cond != "none" else 0.0,
            "speed_all_cmps":       round(csp_all, 3),
            "speed_succ_cmps":      round(csp_succ, 3) if csp_succ == csp_succ else None,
            "delta_speed_pct":      round(100 * (csp_all - base_sp) / base_sp, 1) if (cond != "none" and base_sp) else 0.0,
            "pred_action_speed":    round(pas, 4) if pas == pas else None,
            "pred_action_dz":       round(pdz, 4) if pdz == pdz else None,
            "delta_pred_speed_pct": round(100 * (pas - base_pas) / base_pas, 1) if (cond != "none" and base_pas == base_pas and base_pas) else 0.0,
            "delta_pred_dz":        round(pdz - base_pdz, 4) if (cond != "none" and base_pdz == base_pdz) else 0.0,
            "peaks_success":        peaks_success_by_cond.get(cond, []),
        })
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as _f:
        _json.dump({"policy": args.policy_path, "task_idx": args.task_idx,
                    "results": summary_rows}, _f, indent=2)
    print(f"[✓] summary JSON → {summary_path}")

    # ── boxplot of the STEERED feature by condition (matches DiMaS Fig 2/3) ──────
    # speed run  -> distribution of realized EEF displacement per step (all rollouts)
    # height run -> distribution of carry-phase peak EEF-z (successful rollouts)
    if steer_feature == "speed":
        vals_by_cond = speed_all_by_cond
        ylabel = "EEF displacement per step (cm)  [all rollouts]"
        title = "Speed steering — EEF displacement distribution"
        subset = "all"
    else:
        vals_by_cond = peaks_success_by_cond
        ylabel = "Carry-phase peak EEF z (cm)  [successful rollouts]"
        title = "Height steering — carry-z distribution (task-completed)"
        subset = "success"
    if len(vals_by_cond) >= 1:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = {"none": "#555",
                "dimas_high": "#C0392B", "dimas_low": "#2E86C1"}   # red=increase, blue=decrease
        labels = list(vals_by_cond.keys())
        data = [vals_by_cond[c] if vals_by_cond[c] else [np.nan] for c in labels]
        fig, ax = plt.subplots(figsize=(max(4, 2 * len(labels)), 5))
        bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "white", "lw": 2})
        for patch, c in zip(bp["boxes"], labels):
            patch.set_facecolor(cols.get(c, "#888")); patch.set_alpha(0.85)
        all_vals = [v for d in data for v in d if v == v]
        if all_vals:
            ymin = min(all_vals)
            for i, c in enumerate(labels):
                ax.text(i + 1, ymin - 0.04 * (abs(ymin) + 1),
                        f"n={len([v for v in vals_by_cond[c] if v == v])}",
                        ha="center", fontsize=9, color="black")
        ax.set_xticklabels([disp_label(c) for c in labels], rotation=15, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{Path(args.policy_path).name}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "steering_separation_boxplot.png", dpi=200)
        print(f"[✓] boxplot ({steer_feature}, {subset}) → {out_dir/'steering_separation_boxplot.png'}")
    print(f"[✓] outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
