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
    --n-rollouts 40 --aperture 10.0 --top-k 64 \\
    --task-idx 0 --max-steps 400 --n-action-steps 10

Speed tips:
  --n-rollouts 20  : faster; needs at least ~10 rollouts per split for stable covariance
  --top-k 32       : smaller artifact, marginally less faithful
"""

import argparse
import sys
import time
from pathlib import Path

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


# ── Activation collection ──────────────────────────────────────────────────────

def _register_hooks(policy):
    """Hook lm_expert MLP outputs. Returns (handles, buf).
    buf[i] = list of (d,) float32 arrays, one per predict_action call."""
    layers = policy.model.vlm_with_expert.lm_expert.layers
    buf = [[] for _ in range(len(layers))]
    handles = []
    for i, layer in enumerate(layers):
        def _h(m, inp, out, _i=i):
            h = out[0] if isinstance(out, tuple) else out
            buf[_i].append(h.detach().mean(dim=1).squeeze(0).cpu().float().numpy())
        handles.append(layer.mlp.register_forward_hook(_h))
    return handles, buf


def run_collection_rollout(env, init_state, policy, preprocessor, postprocessor,
                           device, task, max_steps):
    """Run one rollout collecting lm_expert activations + carry EEF-z.

    Returns:
        layer_acts : list[ndarray | None]  shape (N_calls, d) per layer
        carry_z    : float  mean carry-phase EEF-z (m)
        grasped    : bool
    """
    from lerobot.utils.control_utils import predict_action

    policy.reset(); preprocessor.reset(); postprocessor.reset()
    handles, buf = _register_hooks(policy)

    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))

    bowls = _bowl_keys(obs)
    b0 = {b: float(obs[b][2]) for b in bowls}
    bowl_peak = dict(b0)
    eef_z_trace = []

    for _ in range(max_steps):
        agent = np.ascontiguousarray(obs["agentview_image"][::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
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
    grasped = bool(bowls and max(bowl_peak[b] - b0[b] for b in bowls) > 0.03)

    n_layers = len(policy.model.vlm_with_expert.lm_expert.layers)
    layer_acts = [np.stack(buf[i], axis=0) if buf[i] else None for i in range(n_layers)]
    return layer_acts, carry_z, grasped


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
                    help="Total rollouts; top/bottom halves become high/low (default: 40)")
    ap.add_argument("--max-steps",   type=int,   default=400)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--aperture",    type=float, default=10.0,
                    help="Conceptor aperture α — larger = less regularised (default: 10)")
    ap.add_argument("--top-k",       type=int,   default=64,
                    help="Eigenvectors to keep in compressed C_steer (default: 64)")
    ap.add_argument("--output",      required=True)
    ap.add_argument("--device",      default=None)
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device
    from libero.libero import benchmark

    device = get_safe_torch_device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

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
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy_path)

    n_layers = len(policy.model.vlm_with_expert.lm_expert.layers)
    print(f"lm_expert: {n_layers} layers")

    env = make_env(bddl_path)

    # ── Phase 1: collect rollouts ──────────────────────────────────────────────
    all_acts  = []   # list[list[ndarray|None]]  — [rollout][layer]
    all_z     = []   # float per rollout
    all_grasp = []

    print(f"\nCollecting {args.n_rollouts} rollouts …")
    t0 = time.time()
    for ri in tqdm(range(args.n_rollouts), desc="rollouts"):
        init = init_states[ri % len(init_states)]
        acts, cz, grasped = run_collection_rollout(
            env, init, policy, preprocessor, postprocessor,
            device, args.task, args.max_steps)
        all_acts.append(acts)
        all_z.append(cz)
        all_grasp.append(grasped)
        tqdm.write(f"  r{ri:03d}  carry_z={cz*100:.1f}cm  grasp={grasped}")

    env.close()
    elapsed = time.time() - t0
    print(f"\nCollection done: {args.n_rollouts} rollouts in {elapsed/60:.1f} min  "
          f"(grasp {sum(all_grasp)}/{args.n_rollouts})")

    # ── Phase 2: split by carry EEF-z ─────────────────────────────────────────
    z_arr = np.array(all_z)
    order = np.argsort(z_arr)
    k     = max(1, len(order) // 2)
    high_idx = list(order[-k:])
    low_idx  = list(order[:k])

    z_high = float(np.mean(z_arr[high_idx])) * 100
    z_low  = float(np.mean(z_arr[low_idx]))  * 100
    print(f"\nSplit  HIGH({k} rollouts) z={z_high:.1f}cm  "
          f"LOW({k} rollouts) z={z_low:.1f}cm  Δ={z_high-z_low:.1f}cm")
    if z_high - z_low < 2.0:
        print("  WARNING: very small height gap — conceptors may not separate well. "
              "Try more rollouts or a different task.")

    # ── Phase 3: fit conceptors per layer ─────────────────────────────────────
    print(f"\nFitting conceptors (aperture={args.aperture}, top_k={args.top_k}) …")

    def stack_layer(indices, li):
        parts = [all_acts[ri][li] for ri in indices if all_acts[ri][li] is not None]
        return np.concatenate(parts, axis=0) if parts else None

    coast_layers = {}
    for li in tqdm(range(n_layers), desc="layers"):
        X_h = stack_layer(high_idx, li)
        X_l = stack_layer(low_idx,  li)
        if X_h is None or X_l is None or len(X_h) < 4 or len(X_l) < 4:
            continue

        C_h  = _make_conceptor(X_h, args.aperture)
        C_l  = _make_conceptor(X_l, args.aperture)

        # Contrastive conceptors
        C_steer_high = _conceptor_and(C_h, _conceptor_not(C_l))  # high ∧ ¬low
        C_steer_low  = _conceptor_and(C_l, _conceptor_not(C_h))  # low  ∧ ¬high

        ev_h, V_h = _compress(C_steer_high, args.top_k)
        ev_l, V_l = _compress(C_steer_low,  args.top_k)

        coast_layers[str(li)] = {
            "high_eigvals": torch.from_numpy(ev_h),  # (k,)
            "high_eigvecs": torch.from_numpy(V_h),   # (d, k)
            "low_eigvals":  torch.from_numpy(ev_l),
            "low_eigvecs":  torch.from_numpy(V_l),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind":        "coast",
        "n_layers":    n_layers,
        "aperture":    args.aperture,
        "top_k":       args.top_k,
        "z_high_cm":   z_high,
        "z_low_cm":    z_low,
        "policy_path": args.policy_path,
        "layers":      coast_layers,
        "hook_point":  "lm_expert.layers[i].mlp  (forward hook on output)",
    }, out)
    norms_h = [coast_layers[k]["high_eigvals"].abs().max().item() for k in coast_layers]
    norms_l = [coast_layers[k]["low_eigvals"].abs().max().item()  for k in coast_layers]
    print(f"\n[✓] Saved {len(coast_layers)}/{n_layers} layer conceptors → {out}")
    print(f"    C_steer_high: max eigval min={min(norms_h):.4f}  max={max(norms_h):.4f}")
    print(f"    C_steer_low:  max eigval min={min(norms_l):.4f}  max={max(norms_l):.4f}")
    print(f"\nEval command:")
    print(f"  python src/lerobot/scripts/libero_osc_eval.py \\")
    print(f"    --policy-path {args.policy_path} \\")
    print(f"    --conditions none coast_high coast_low \\")
    print(f"    --coast-path {out} --coast-beta 0.5 \\")
    print(f"    --task-idx {args.task_idx} --n-rollouts 20 --max-steps {args.max_steps}")


if __name__ == "__main__":
    main()
