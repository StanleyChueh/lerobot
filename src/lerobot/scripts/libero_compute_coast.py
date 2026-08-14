#!/usr/bin/env python3
"""
Compute COAST (Contrastive Conceptor Activation Steering) artifacts for SmolVLA.

Unlike CAA's rank-1 mean-difference vector, COAST fits a SUBSPACE operator per
layer from closed-loop rollout activations, then composes them via Boolean
conceptor algebra into a contrastive gate.

Math (Sec 3.1 of arxiv:2605.17144):
  C = R(R + α⁻²I)⁻¹   where R = X̄ᵀX̄/N  (Jaeger 2014 conceptor)
  C_steer = C_high ∧ ¬C_low   (AND of high, NOT-low: keeps directions high but not low)
  AND: A ∧ B = (A⁻¹ + B⁻¹ − I)⁻¹
  NOT: ¬C = I − C

Inference gate (libero_osc_eval.py with --conditions coast_high/coast_low):
  M = (1−β)I + β·C_steer        (β = steering strength)
  h' = h @ M   (multiplicative on residual — not additive like CAA)

Advantages over CAA:
  • Captures multi-dimensional subspace, not just rank-1 mean shift
  • Suppresses shared low-variance noise; amplifies outcome-discriminative directions
  • Contrastive composition (∧ ¬C_low) cancels directions common to both heights

Data source: closed-loop rollouts in the LIBERO sim (not offline dataset).
  Split rollouts by measured carry-phase EEF-z → high/low conceptors.

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_compute_coast.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_three_cams_40k \\
    --output outputs/coast_three_cams_40k.pt \\
    --n-rollouts 40 --aperture 10.0 \\
    --task-idx 0 --max-steps 400 --n-action-steps 10

Speed tips:
  --n-rollouts 20  : faster; needs enough success AND failure rollouts for both conceptors
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Must be set before the CUDA context / first cuBLAS call for deterministic matmuls.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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

DEFAULT_TASK = ("pick the akita black bowl between the plate and the ramekin "
                "and place it on the plate")


# ── Env / obs helpers (mirror libero_osc_eval.py) ─────────────────────────────

def eef_state(obs) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([
        obs["robot0_eef_pos"].astype(np.float32),
        quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        obs["robot0_gripper_qpos"].astype(np.float32),
    ])


def build_obs(agent_hwc, wrist_hwc, state8):
    return {
        "observation.images.camera1":        agent_hwc,
        "observation.images.camera2":        wrist_hwc,
        "observation.images.camera3":        _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


def make_env(bddl_path: Path):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256, camera_widths=256,
    )


def find_demo_file(suite, task_fname):
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


def load_demo_init_states(demo_file: Path):
    with h5py.File(demo_file, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        return [f[f"data/{k}/states"][()][0] for k in keys]


def _bowl_keys(obs):
    return [k for k in obs if "bowl" in k.lower() and k.endswith("_pos") and "to_robot" not in k]


def _plate_key(obs):
    return next((k for k in obs if "plate" in k.lower()
                 and k.endswith("_pos") and "to_robot" not in k), None)


# ── Activation collection ──────────────────────────────────────────────────────

def _register_hooks(policy):
    """Hook lm_expert MLP outputs. Returns (handles, buf).
    buf[i] = list of (d,) GPU tensors, one per expert forward.

    IMPORTANT: accumulate on-GPU (NO per-step .cpu()). A .cpu() call inside the
    hook forces a CUDA sync every layer every step, which — under warn_only
    deterministic mode — shifts nondeterministic-kernel scheduling enough to
    PERTURB the model's own actions (observed: num_steps=1 collection dropped to
    14% vs 50% hook-free). Keeping the read on-GPU makes the hook truly
    non-perturbing; convert to numpy once, after the rollout."""
    layers = policy.model.vlm_with_expert.lm_expert.layers
    buf = [[] for _ in range(len(layers))]
    handles = []
    for i, layer in enumerate(layers):
        def _h(m, inp, out, _i=i):
            h = out[0] if isinstance(out, tuple) else out
            buf[_i].append(h.detach().mean(dim=1).squeeze(0))   # stays on GPU
        handles.append(layer.mlp.register_forward_hook(_h))
    return handles, buf


def run_collection_rollout(env, init_state, policy, preprocessor, postprocessor,
                           device, task, max_steps, collect_frames=False):
    """Run one rollout collecting lm_expert activations + outcome labels.

    Returns dict with:
        acts     : list[ndarray | None]  shape (N_calls, d) per layer
        carry_z  : float  mean carry-phase EEF-z (m)
        grasped  : bool
        on_plate : bool   task success (bowl on plate) — paper's success label
        frames   : list[ndarray] | None  agentview RGB frames (if collect_frames)
    """
    from lerobot.utils.control_utils import predict_action

    policy.reset(); preprocessor.reset(); postprocessor.reset()
    handles, buf = _register_hooks(policy)

    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))

    bowls = _bowl_keys(obs)
    plate_key = _plate_key(obs)
    b0 = {b: float(obs[b][2]) for b in bowls}
    bowl_peak = dict(b0)
    eef_z_trace = []
    frames = [] if collect_frames else None

    for _ in range(max_steps):
        agent = np.ascontiguousarray(obs["agentview_image"][::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
        if collect_frames:
            frames.append(agent.copy())
        policy._caa_gate = True
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
        eef_z_trace.append(float(obs["robot0_eef_pos"][2]))
        for b in bowls:
            bowl_peak[b] = max(bowl_peak[b], float(obs[b][2]))
        if done:
            break

    for h in handles:
        h.remove()

    z = np.array(eef_z_trace)
    T = len(z)
    lo, hi = int(0.2 * T), int(0.8 * T)
    carry_z = float(np.mean(z[lo:hi])) if hi > lo else float(np.mean(z))

    lifted = max(bowls, key=lambda b: bowl_peak[b] - b0[b]) if bowls else None
    grasped = bool(lifted is not None and bowl_peak[lifted] - b0[lifted] > 0.03)
    on_plate = False
    if lifted is not None and plate_key is not None and plate_key in obs:
        fb, tp = obs[lifted], obs[plate_key]
        dxy = float(np.linalg.norm(fb[:2] - tp[:2])); dz = float(fb[2] - tp[2])
        on_plate = bool(dxy < 0.06 and -0.02 < dz < 0.08 and fb[2] > 0.88)

    n_layers = len(policy.model.vlm_with_expert.lm_expert.layers)
    # Convert GPU-accumulated activations to numpy AFTER the rollout (single sync,
    # non-perturbing — see _register_hooks).
    layer_acts = [
        torch.stack(buf[i], dim=0).float().cpu().numpy() if buf[i] else None
        for i in range(n_layers)
    ]
    return {"acts": layer_acts, "carry_z": carry_z,
            "grasped": grasped, "on_plate": on_plate,
            "frames": frames, "eef_z_trace": eef_z_trace}


def save_collection_video(frames, eef_z_trace, out_path: Path, tag, fps=20.0):
    """Write one collection rollout to mp4 with an EEF-z overlay (256x256 RGB)."""
    import cv2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 256))
    for t in range(len(frames)):
        f = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        z = eef_z_trace[t] * 100 if t < len(eef_z_trace) else float("nan")
        cv2.putText(f, f"{tag}  z={z:.1f}cm", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)
        w.write(f)
    w.release()


# ── Conceptor algebra ──────────────────────────────────────────────────────────

def _make_conceptor(X: np.ndarray, aperture: float) -> np.ndarray:
    """C = R(R + α⁻²I)⁻¹  via eigendecomposition of R = XᵀX/N.

    X is mean-centred before computing R so the conceptor captures covariance
    rather than absolute activation level (same as COAST paper Eq. 1).
    Returns C as a (d, d) symmetric float64 matrix.
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    R = (Xc.T @ Xc) / len(Xc)
    eigvals, eigvecs = np.linalg.eigh(R.astype(np.float64))   # stable for PSD
    # conceptor weights: μᵢ = λᵢ / (λᵢ + α⁻²)
    weights = eigvals / (eigvals + aperture ** (-2))
    # C = V diag(μ) Vᵀ
    return (eigvecs * weights[np.newaxis, :]) @ eigvecs.T


def _conceptor_not(C: np.ndarray) -> np.ndarray:
    return np.eye(C.shape[0], dtype=np.float64) - C


def _conceptor_and(A: np.ndarray, B: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """A ∧ B = (A⁻¹ + B⁻¹ − I)⁻¹  with regularised pseudo-inverse."""
    d = A.shape[0]
    I = np.eye(d, dtype=np.float64)
    A_inv = np.linalg.pinv(A + eps * I, rcond=eps)
    B_inv = np.linalg.pinv(B + eps * I, rcond=eps)
    inner = A_inv + B_inv - I
    C = np.linalg.pinv(inner + eps * I, rcond=eps)
    return (C + C.T) * 0.5   # symmetrise


def _compress(C: np.ndarray, top_k: int):
    """Eigendecompose C, keep top-k components (by eigenvalue magnitude).

    Returns (eigvals_k, eigvecs_k): shapes (k,) and (d, k) float32.
    Storing only top-k cuts artifact size dramatically (e.g. 64 vs 2048).
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(np.abs(eigvals))[::-1][:top_k]
    return eigvals[idx].astype(np.float32), eigvecs[:, idx].astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compute COAST contrastive conceptor artifacts from LIBERO rollouts.")
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--suite",       default="libero_spatial")
    ap.add_argument("--task-idx",    type=int,   default=0)
    ap.add_argument("--task",        default=DEFAULT_TASK)
    ap.add_argument("--n-rollouts",  type=int,   default=40,
                    help="Total rollouts to collect (default: 40)")
    ap.add_argument("--split",       choices=["success", "height"], default="success",
                    help="'success' = paper-exact (C_success ∧ ¬C_failure, improves task "
                         "success rate); 'height' = split by carry EEF-z (height steering)")
    ap.add_argument("--positive-only", action="store_true",
                    help="paper's positive-only variant: C_steer = C_success (no ¬C_failure). "
                         "Tests whether projecting toward success alone suffices.")
    ap.add_argument("--max-steps",   type=int,   default=400)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--num-steps",   type=int,   default=None,
                    help="flow-matching denoising steps (config.num_steps); lower = "
                         "less-refined actions = lower baseline (more COAST headroom). "
                         "None keeps the model default. MUST match the eval setting.")
    ap.add_argument("--aperture",    type=float, default=10.0,
                    help="Conceptor aperture α — larger = less regularised (default: 10)")
    ap.add_argument("--output",      required=True)
    ap.add_argument("--device",      default=None)
    ap.add_argument("--save-video",  action="store_true",
                    help="render each collection rollout to mp4, filed under "
                         "<video-dir>/success|failure/ by outcome label")
    ap.add_argument("--video-dir",   default=None,
                    help="dir for --save-video mp4s (default: <output>_videos next to --output)")
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device
    from libero.libero import benchmark
    from lerobot.scripts.libero_eval_steering import enable_determinism

    enable_determinism()   # reproducible success/failure labels for clean conceptor fitting
    device = get_safe_torch_device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  (deterministic algorithms ON)")

    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    fname     = suite_obj.get_task_bddl_files()[args.task_idx]
    bddl_path = BDDL_ROOT / args.suite / fname
    print(f"Task {args.task_idx}: {fname}")

    demo_file = find_demo_file(args.suite, fname)
    if demo_file is None:
        raise FileNotFoundError(f"No demo HDF5 found under {DEMO_DIR / args.suite}")
    init_states = load_demo_init_states(demo_file)
    print(f"Loaded {len(init_states)} demo init states")

    print(f"Loading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.config.n_action_steps = args.n_action_steps
    if args.num_steps is not None:
        policy.config.num_steps = args.num_steps
        print(f"  flow-matching num_steps = {policy.config.num_steps}")
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy_path)

    n_layers = len(policy.model.vlm_with_expert.lm_expert.layers)
    print(f"lm_expert: {n_layers} layers")

    env = make_env(bddl_path)

    video_dir = None
    if args.save_video:
        video_dir = Path(args.video_dir) if args.video_dir else \
            Path(str(args.output).rsplit(".", 1)[0] + "_videos")
        print(f"Saving collection-rollout videos → {video_dir}/(success|failure)/")

    # ── Phase 1: collect rollouts ──────────────────────────────────────────────
    all_acts, all_z, all_grasp, all_plate = [], [], [], []

    print(f"\nCollecting {args.n_rollouts} rollouts …")
    t0 = time.time()
    for ri in tqdm(range(args.n_rollouts), desc="rollouts"):
        init = init_states[ri % len(init_states)]
        r = run_collection_rollout(
            env, init, policy, preprocessor, postprocessor,
            device, args.task, args.max_steps, collect_frames=args.save_video)
        all_acts.append(r["acts"])
        all_z.append(r["carry_z"])
        all_grasp.append(r["grasped"])
        all_plate.append(r["on_plate"])
        tqdm.write(f"  r{ri:03d}  carry_z={r['carry_z']*100:.1f}cm  "
                   f"grasp={int(r['grasped'])}  on_plate={int(r['on_plate'])}")
        if args.save_video and r["frames"]:
            label = "success" if r["on_plate"] else "failure"
            save_collection_video(
                r["frames"], r["eef_z_trace"],
                video_dir / label / f"rollout_{ri:03d}_{label}.mp4",
                tag=f"r{ri:03d} {label}")

    env.close()
    elapsed = time.time() - t0
    n_succ = sum(all_plate)
    print(f"\nCollection done: {args.n_rollouts} rollouts in {elapsed/60:.1f} min  "
          f"(grasp {sum(all_grasp)}/{args.n_rollouts}, success {n_succ}/{args.n_rollouts})")

    # ── Phase 2: split into positive (pos) / negative (neg) groups ─────────────
    z_arr = np.array(all_z)
    if args.split == "success":
        pos_idx = [i for i, ok in enumerate(all_plate) if ok]        # success rollouts
        neg_idx = [i for i, ok in enumerate(all_plate) if not ok]    # failure rollouts
        meaning = {"pos": "success", "neg": "failure"}
        print(f"\nSplit (success)  POS=success({len(pos_idx)})  NEG=failure({len(neg_idx)})")
        if len(pos_idx) < 3 or len(neg_idx) < 3:
            raise RuntimeError(
                f"Need >=3 success AND >=3 failure rollouts to fit contrastive conceptors "
                f"(got success={len(pos_idx)}, failure={len(neg_idx)}). "
                f"Baseline is too near-ceiling/floor on this task — pick a harder --task-idx "
                f"or collect more --n-rollouts.")
    else:  # height
        order = np.argsort(z_arr)
        k = max(1, len(order) // 2)
        pos_idx = list(order[-k:]); neg_idx = list(order[:k])
        meaning = {"pos": "high", "neg": "low"}
        print(f"\nSplit (height)  POS=high({len(pos_idx)}) z={np.mean(z_arr[pos_idx])*100:.1f}cm  "
              f"NEG=low({len(neg_idx)}) z={np.mean(z_arr[neg_idx])*100:.1f}cm")

    z_pos = float(np.mean(z_arr[pos_idx])) * 100
    z_neg = float(np.mean(z_arr[neg_idx])) * 100

    # ── Phase 3: fit conceptors per layer ─────────────────────────────────────
    tag = "positive-only (C_success)" if args.positive_only else "contrastive (C_pos ∧ ¬C_neg)"
    print(f"\nFitting conceptors [{tag}] (aperture={args.aperture}) …")

    def stack_layer(indices, li):
        parts = [all_acts[ri][li] for ri in indices if all_acts[ri][li] is not None]
        return np.concatenate(parts, axis=0) if parts else None

    coast_layers = {}
    for li in tqdm(range(n_layers), desc="layers"):
        X_p = stack_layer(pos_idx, li)
        X_n = stack_layer(neg_idx, li)
        if X_p is None or X_n is None or len(X_p) < 4 or len(X_n) < 4:
            continue

        C_p = _make_conceptor(X_p, args.aperture)
        C_n = _make_conceptor(X_n, args.aperture)

        if args.positive_only:
            # C_steer = C_success (steer toward success subspace only)
            C_steer_pos = C_p
            C_steer_neg = C_n
        else:
            # Contrastive: C_pos ∧ ¬C_neg  (keeps directions in pos, absent from neg)
            C_steer_pos = _conceptor_and(C_p, _conceptor_not(C_n))
            C_steer_neg = _conceptor_and(C_n, _conceptor_not(C_p))

        # Store the FULL (d,d) conceptor — applied as h' = (1-β)h + β·h·C_steer at
        # eval. Full matrix is exact (top-k eigen-compression drops the many
        # mid-eigenvalue directions of C_steer and distorts the gate); d≈720 so
        # 16 layers × 2 dirs is only ~66 MB.
        coast_layers[str(li)] = {
            "pos": torch.from_numpy(C_steer_pos.astype(np.float32)),
            "neg": torch.from_numpy(C_steer_neg.astype(np.float32)),
            "d": int(C_steer_pos.shape[0]),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind":        "coast",
        "split":       args.split,
        "meaning":     meaning,              # {"pos": "success"/"high", "neg": "failure"/"low"}
        "positive_only": args.positive_only,
        "n_layers":    n_layers,
        "aperture":    args.aperture,
        "z_pos_cm":    z_pos,
        "z_neg_cm":    z_neg,
        "n_success":   n_succ,
        "n_rollouts":  args.n_rollouts,
        "policy_path": args.policy_path,
        "layers":      coast_layers,
        "hook_point":  "lm_expert.layers[i].mlp  (forward hook on output)",
    }, out)
    # diagnostic: mean eigenvalue of C_steer_pos per layer (fraction of variance kept)
    tr = [float(np.trace(coast_layers[k]["pos"].numpy())) / coast_layers[k]["d"]
          for k in coast_layers]
    print(f"\n[✓] Saved {len(coast_layers)}/{n_layers} layer conceptors → {out}")
    print(f"    split={args.split}  meaning={meaning}  C_steer_pos mean-eigval "
          f"min={min(tr):.4f} max={max(tr):.4f} (per-layer avg retained variance)")
    steer_cond = "coast_pos" if args.split == "height" else "coast_success"
    print(f"\nEval command (steer toward {meaning['pos']}):")
    print(f"  python src/lerobot/scripts/libero_osc_eval.py \\")
    print(f"    --policy-path {args.policy_path} \\")
    print(f"    --conditions none {steer_cond} \\")
    print(f"    --coast-path {out} --coast-beta 0.5 \\")
    print(f"    --task-idx {args.task_idx} --n-rollouts 20 --max-steps {args.max_steps}")


if __name__ == "__main__":
    main()
