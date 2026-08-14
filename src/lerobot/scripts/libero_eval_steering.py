#!/usr/bin/env python3
"""
LIBERO Activation-Steering Evaluation for SmolVLA
──────────────────────────────────────────────────
Offline dataset-conditioned evaluation: camera images replay from collected
LIBERO HDF5 episodes; joint state evolves autonomously via policy predictions;
EEF height is computed at each step via exact Franka DH forward kinematics.

Three steering conditions compared on the SAME visual context per rollout:
  none     — baseline (no steering)
  caa      — dense per-layer CAA steering (from libero_compute_caa.py)
  keyword  — per-neuron activation steering (from lerobot_reord_top_token.py)

Outputs (in --out-dir):
  eef_traces_{cond}.npz        raw EEF-z traces, one row per rollout (metres)
  rollout_metrics_{cond}.csv   per-rollout stats (peak, mean, carry phase)
  summary_{cond}.json          aggregate stats with 95% bootstrap CI
  eef_trajectory_comparison.png  paper-ready figure (all conditions)
  videos/{cond}/rollout_NNN.mp4  side-by-side video: policy input | EEF bar

Usage:

  # 1) Baseline only (fast sanity check)
  conda run -n lerobot python src/lerobot/scripts/libero_eval_steering.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero \\
    --hdf5 /home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5 \\
           /home/bruce/datasets/libero_height_demos/libero_spatial/low/task_00.hdf5 \\
    --task "Pick up the black bowl and place it on the plate." \\
    --conditions none \\
    --n-rollouts 10 --out-dir outputs/libero_eval

  # 2) All conditions (full paper run)
  conda run -n lerobot python src/lerobot/scripts/libero_eval_steering.py \\
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero \\
    --hdf5 /home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5 \\
           /home/bruce/datasets/libero_height_demos/libero_spatial/low/task_00.hdf5 \\
    --task "Pick up the black bowl and place it on the place." \\
    --conditions none caa keyword_high keyword_low \\
    --caa-path outputs/libero_caa_vectors.pt \\
    --caa-alpha 2.0 \\
    --neurons-json outputs/libero_height_neurons.json \\
    --keyword-alpha 4.0 \\
    --n-rollouts 30 --save-video --out-dir outputs/libero_eval_full
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── Franka DH forward kinematics ──────────────────────────────────────────────

_PI = np.pi
_FRANKA_DH = [
    [0,        0.333,   0      ],
    [0,        0,      -_PI/2  ],
    [0,        0.316,   _PI/2  ],
    [0.0825,   0,       _PI/2  ],
    [-0.0825,  0.384,  -_PI/2  ],
    [0,        0,       _PI/2  ],
    [0.088,    0.107,   _PI/2  ],
]
_FRANKA_EEF_Z = 0.1034


def franka_fk_z(q: np.ndarray) -> float:
    """EEF z-height (metres) from 7-DoF Franka joint angles (radians)."""
    T = np.eye(4)
    for i, (a, d, alpha) in enumerate(_FRANKA_DH):
        t = float(q[i])
        ct, st = np.cos(t), np.sin(t)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T = T @ np.array([
            [ct, -st*ca,  st*sa,  a*ct],
            [st,  ct*ca, -ct*sa,  a*st],
            [0,   sa,     ca,     d   ],
            [0,   0,      0,      1   ],
        ])
    return float(T[2, 3] + T[2, 2] * _FRANKA_EEF_Z)


def fit_fk_offset(episode: dict) -> float:
    """
    Compute the FK z-offset (world_z - base_z) from the first stored frame.
    The Franka in LIBERO is mounted on a table, so world_z = fk_z + ~0.855 m.
    Using a per-episode offset ensures any minor variation is accounted for.
    """
    return float(episode["eef_pos"][0, 2] - franka_fk_z(episode["joint_pos"][0]))


def validate_fk(hdf5_path: str, n: int = 30):
    """Validate calibrated FK against stored eef_pos; return mean Δz error (cm)."""
    with h5py.File(hdf5_path, "r") as f:
        g = f[sorted(k for k in f.keys() if k.startswith("ep_"))[0]]
        jp = g["joint_pos"][()]
        ep = g["eef_pos"][()]
    offset = ep[0, 2] - franka_fk_z(jp[0])
    errs = [abs(franka_fk_z(jp[i]) + offset - ep[i, 2]) * 100 for i in range(min(n, len(jp)))]
    err = float(np.mean(errs))
    print(f"  Calibrated FK (offset={offset*100:.1f} cm): "
          f"mean Δz={err:.2f} cm  max={float(np.max(errs)):.2f} cm  [n={min(n,len(jp))}]")
    return err


# ── Observation format ─────────────────────────────────────────────────────────

_EMPTY_256 = np.zeros((256, 256, 3), dtype=np.uint8)
_EMPTY_480 = np.zeros((480, 640, 3), dtype=np.uint8)


def build_obs(img: np.ndarray, state8: np.ndarray, task: str) -> dict:
    # state8: (8,) = [panda_joint_1..7, gripper] — must be 8D to match normalizer stats
    return {
        "observation.images.agentview":      img,          # preprocessor renames → camera1
        "observation.images.camera2":        _EMPTY_256,
        "observation.images.camera3":        _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.images.empty_camera_1": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


# ── HDF5 episode loading ───────────────────────────────────────────────────────

def load_all_episodes(hdf5_paths: list[str]) -> list[dict]:
    """Return list of {imgs, joint_pos, gripper, eef_pos, arc_type} dicts."""
    episodes = []
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            arc_type = f.attrs.get("arc_type", Path(path).parent.name)
            for key in sorted(k for k in f.keys() if k.startswith("ep_")):
                g = f[key]
                episodes.append({
                    "imgs":      g["agentview_image"][()],   # (T, 256, 256, 3)
                    "joint_pos": g["joint_pos"][()],          # (T, 7)
                    "gripper":   g["actions"][:, 6:7],        # (T, 1) gripper cmd [-1=open, +1=close]
                    "eef_pos":   g["eef_pos"][()],            # (T, 3)
                    "arc_type":  arc_type,
                    "src_file":  path,
                    "src_key":   key,
                })
    print(f"  Loaded {len(episodes)} episodes from {len(hdf5_paths)} file(s)")
    return episodes


# ── Steering infrastructure ────────────────────────────────────────────────────

def enable_determinism(seed: int = 0):
    """Make SmolVLA rollouts reproducible + history-independent.

    Non-deterministic CUDA kernels (atomic reductions in attention/matmuls) flip
    borderline LIBERO rollouts run-to-run, so the SAME (init_state, steering) can
    succeed or fail depending on execution history. That breaks any paired
    baseline-vs-steering comparison. Enabling deterministic algorithms makes each
    rollout a pure function of (init_state, steering). Call BEFORE the first CUDA
    op; also set env CUBLAS_WORKSPACE_CONFIG=:4096:8 before the process starts."""
    import os
    import random
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _expert_layers(policy):
    return policy.model.vlm_with_expert.lm_expert.layers


def _vlm_text_layers(policy):
    return policy.model.vlm_with_expert.get_vlm_model().text_model.layers


def clear_steering(policy):
    for attr in ("_caa_handles", "_kw_handles", "_phys_handles", "_dimas_handles",
                 "_coast_handles"):
        if hasattr(policy, attr):
            for h in getattr(policy, attr):
                try:
                    h.remove()
                except Exception:
                    pass
            setattr(policy, attr, [])
    if hasattr(policy, "clear_activation_steering"):
        try:
            policy.clear_activation_steering()
        except Exception:
            pass


def setup_caa(policy, caa_path: str, alpha: float,
              layer_lo: int | None = None, layer_hi: int | None = None,
              exclude_last: bool = False, normalize: bool = False):
    """Inject the CAA delta at each lm_expert layer's MLP OUTPUT — the SAME point
    where libero_compute_caa_osc.register_mlp_hooks() captured the contrast.

    SmolVLA's fused VLM+expert forward decomposes each layer manually (no
    layer.forward call), so a layer-level residual hook never fires; layer.mlp is
    the reachable injection point, and the delta added here propagates into the
    residual stream via the forward's `out_emb += after_first_residual`. Capture
    and injection MUST use the same module (layer.mlp).

    Layer selection / scaling (standard CAA/RepE practice):
      layer_lo, layer_hi : restrict steering to layers [layer_lo, layer_hi) —
                           middle layers steer cleanly; early/late layers tend to
                           scramble the output. None = no bound on that side.
      exclude_last       : skip the final expert layer (often a degenerate,
                           high-norm contrast that corrupts the action head).
      normalize          : unit-normalize each layer's vector so alpha means the
                           same thing regardless of the model's activation scale
                           (portable alpha; recommended when norms vary a lot)."""
    clear_steering(policy)
    obj = torch.load(caa_path, map_location="cpu", weights_only=False)
    if "vectors" not in obj:
        raise KeyError(f"'vectors' key missing in {caa_path}")
    layers = _expert_layers(policy)
    n_layers = len(layers)
    lo = 0 if layer_lo is None else layer_lo
    hi = n_layers if layer_hi is None else layer_hi
    handles = []
    used = []
    for layer_key, vec in obj["vectors"].items():
        idx = int(layer_key)
        if not (0 <= idx < n_layers):
            continue
        if not (lo <= idx < hi):
            continue
        if exclude_last and idx == n_layers - 1:
            continue
        v = vec.float().cpu()
        if v.ndim == 1:
            v = v.unsqueeze(0)
        if normalize:
            nrm = v.norm()
            if nrm > 1e-8:
                v = v / nrm

        def _hook(module, inputs, output, _v=v, _a=alpha, _pol=policy):
            # Phase gate: when a caller sets policy._caa_gate=False (e.g. outside the
            # carry sub-phase), pass activations through unperturbed so grasp/place
            # run at native behavior and only the carry height is steered.
            if not getattr(_pol, "_caa_gate", True):
                return output
            h = output[0] if isinstance(output, tuple) else output
            delta = _a * _v.to(h.device, h.dtype)
            if isinstance(output, tuple):
                return (h + delta,) + output[1:]
            return h + delta

        handles.append(layers[idx].mlp.register_forward_hook(_hook))
        used.append(idx)
    policy._caa_handles = handles
    print(f"  [CAA] {len(handles)} hooks (layer.mlp), alpha={alpha}, "
          f"layers={used}, normalize={normalize}")


def setup_dimas(policy, artifact_path: str, direction: str, alpha: float,
                gate: bool = True, step_m=None):
    """DiMaS (distribution-matching) steering — the OT alternative to linear CAA.

    Ref: Khayatan et al., arXiv:2607.14280. At the target action-expert layer's
    mlp output, pool the token reps to p, standardize, and:
      - GATE: only steer reps where the target feature is ABSENT (a linear
        classifier says p is in the source tail), leaving already-correct reps alone.
      - TRANSPORT: nearest-neighbor project p onto the source tail samples, take
        that sample's barycentric OT image in the target tail, and interpolate
        p_new = (1-alpha) p + alpha * T(P(p))  (alpha in [0,1] preserves success).
      - Add the resulting delta (broadcast over tokens) to the layer.mlp output.

    Unlike CAA's single fixed vector, the shift here DEPENDS on where p lands in
    the source distribution — that per-sample adaptivity is what lets it match the
    high/low distributions that a fixed linear shift cannot.

    direction: "high" (steer toward high feature) or "low".
    """
    clear_steering(policy)
    art = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if art.get("kind") != "dimas":
        raise ValueError(f"{artifact_path} is not a DiMaS artifact (kind={art.get('kind')})")
    if direction not in ("high", "low"):
        raise ValueError("direction must be 'high' or 'low'")

    layers = _expert_layers(policy)
    li = int(art["layer"])
    if not (0 <= li < len(layers)):
        raise ValueError(f"artifact layer {li} out of range for this model")

    dev = next(policy.parameters()).device
    mean = art["mean"].to(dev).float()
    std = art["std"].to(dev).float()
    source = art[direction]["source"].to(dev).float()   # standardized source-tail reps
    target = art[direction]["target"].to(dev).float()   # their barycentric OT images
    gate_w = art["gate_w"].to(dev).float()
    gate_b = float(art["gate_b"])
    feature = art["feature"]

    # Gate fires (=steer) when the target feature is ABSENT:
    #   direction high  -> steer reps predicted LOW  -> p(high) < 0.5
    #   direction low   -> steer reps predicted HIGH -> p(high) > 0.5
    steer_when_high = (direction == "low")

    # Per-DENOISING-STEP injection (paper injects at ONE step m, not all steps).
    # sample_actions runs exactly num_steps expert forwards per chunk, so a free-
    # running call counter mod num_steps recovers the current denoising step index.
    num_steps = int(getattr(policy.config, "num_steps", 10))
    _state = {"call": 0}

    def _hook(module, inputs, output, _src=source, _tgt=target, _a=alpha,
              _mean=mean, _std=std, _gw=gate_w, _gb=gate_b, _gate=gate,
              _steer_when_high=steer_when_high, _step_m=step_m, _ns=num_steps):
        cur_step = _state["call"] % _ns
        _state["call"] += 1
        h = output[0] if isinstance(output, tuple) else output   # (B,T,d)
        if _step_m is not None and cur_step != _step_m:
            return output                                        # skip other steps
        p = h.mean(dim=1)                                        # (B,d)
        p_std = (p - _mean) / _std
        # gate
        if _gate:
            logit = p_std @ _gw + _gb
            fire = (logit > 0) if _steer_when_high else (logit < 0)   # (B,)
        else:
            fire = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
        # nearest-neighbor transport in standardized space
        d2 = torch.cdist(p_std, _src)              # (B, Nsrc)
        j = d2.argmin(dim=1)                       # (B,)
        tgt_std = _tgt[j]                          # (B,d)
        p_new_std = (1 - _a) * p_std + _a * tgt_std
        p_new = p_new_std * _std + _mean
        delta = (p_new - p) * fire.float()[:, None]   # zero where gate off
        h2 = h + delta[:, None, :]
        if isinstance(output, tuple):
            return (h2,) + output[1:]
        return h2

    handle = layers[li].mlp.register_forward_hook(_hook)
    policy._dimas_handles = [handle]
    print(f"  [DiMaS-{direction}] layer={li}, feature={feature}, alpha={alpha}, "
          f"gate={gate}, step_m={step_m}/{num_steps}, src={tuple(source.shape)}")


def setup_keyword_neurons(
    policy, neurons_json: str, direction: str, alpha: float,
    top_n: int | None = None, mode: str = "add", bidirectional: bool = False,
):
    """
    Keyword-neuron steering on VLM text_model.layers[i].mlp.down_proj inputs.

    mode='add'  (recommended): activation[..., neuron_ids] += alpha
                 Always pushes in direction regardless of natural activation level.
    mode='set'  (paper exact): activation[..., neuron_ids]  = alpha
                 Uses policy.set_activation_steering(); may reduce if natural > alpha.

    bidirectional=True: also subtract alpha from opposing-concept neurons for
                 maximum contrast (keyword_high suppresses low neurons, and vice versa).
    """
    clear_steering(policy)
    with open(neurons_json) as f:
        raw = json.load(f)
    if direction not in raw:
        raise ValueError(f"Direction '{direction}' not in {neurons_json}. Keys: {list(raw.keys())}")

    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model

    def _build_pruned(concept, _top_n):
        nmap = {int(k): v for k, v in raw.get(concept, {}).items()}
        if _top_n is not None:
            flat = [(l, n) for l, ns in nmap.items() for n in ns][:_top_n]
            nmap = {}
            for l, n in flat:
                nmap.setdefault(l, []).append(n)
        pruned = {}
        for li, idxs in nmap.items():
            if not (0 <= li < len(text_model.layers)):
                continue
            d_ff = text_model.layers[li].mlp.down_proj.weight.shape[1]
            valid = [n for n in idxs if 0 <= n < d_ff]
            if valid:
                pruned[li] = valid
        return pruned

    target_map   = _build_pruned(direction, top_n)
    opposite     = "low" if direction == "high" else "high"
    opposite_map = _build_pruned(opposite, top_n) if bidirectional else {}

    if mode == "set":
        # Built-in SET: activation = alpha for target neurons
        policy.set_activation_steering(
            steering_neurons=target_map,
            alpha=alpha,
            record_debug=False,
            enable_steering=True,
        )
        # Bidirectional suppression via custom hook (SET opposing to -alpha)
        kw_handles = []
        for li, idxs in opposite_map.items():
            idxs_t = torch.tensor(idxs, dtype=torch.long)
            def _suppress(module, inputs, _i=idxs_t, _a=alpha):
                inp = list(inputs)
                act = inp[0].clone()
                act[..., _i.to(act.device)] = -_a
                inp[0] = act
                return tuple(inp)
            kw_handles.append(
                text_model.layers[li].mlp.down_proj.register_forward_pre_hook(_suppress)
            )
        policy._kw_handles = kw_handles
    else:
        # ADD mode: register pre-hooks that do +=
        kw_handles = []
        for li, idxs in target_map.items():
            idxs_t = torch.tensor(idxs, dtype=torch.long)
            def _add(module, inputs, _i=idxs_t, _a=alpha):
                inp = list(inputs)
                act = inp[0].clone()
                act[..., _i.to(act.device)] += _a
                inp[0] = act
                return tuple(inp)
            kw_handles.append(
                text_model.layers[li].mlp.down_proj.register_forward_pre_hook(_add)
            )
        for li, idxs in opposite_map.items():
            idxs_t = torch.tensor(idxs, dtype=torch.long)
            def _sub(module, inputs, _i=idxs_t, _a=alpha):
                inp = list(inputs)
                act = inp[0].clone()
                act[..., _i.to(act.device)] -= _a
                inp[0] = act
                return tuple(inp)
            kw_handles.append(
                text_model.layers[li].mlp.down_proj.register_forward_pre_hook(_sub)
            )
        policy._kw_handles = kw_handles

    nt = sum(len(v) for v in target_map.values())
    no = sum(len(v) for v in opposite_map.values())
    print(f"  [KEYWORD-{direction}] mode={mode}, target={nt}n/{len(target_map)}L "
          f"suppress={no}n, alpha={alpha:.1f}, bidir={bidirectional}")


def setup_physical_caa(policy, vectors_path: str, direction: str, alpha: float,
                       top_k: int | None = None):
    """
    Sparse-CAA physical-neuron steering (the principled maximum of the
    physical-neuron method): ADD sign*alpha*(mean_high − mean_low) to the
    VLM text_model.layers[i].mlp.down_proj INPUT activation.

    Unlike fixed-alpha SET, each neuron is weighted by its ACTUAL contrast
    magnitude+sign — so both high and low directions are captured faithfully.

    direction: "high" → +alpha*vec  ;  "low" → −alpha*vec
    top_k: if set, keep only the top-k neurons per layer by |contrast| (sparse);
           None uses the full dense VLM-space contrast vector.
    """
    clear_steering(policy)
    obj = torch.load(vectors_path, map_location="cpu", weights_only=False)
    vectors = obj["vectors"] if "vectors" in obj else obj
    sign = +1.0 if direction == "high" else -1.0

    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    handles = []
    total_n = 0
    for layer_key, vec in vectors.items():
        li = int(layer_key)
        if not (0 <= li < len(text_model.layers)):
            continue
        v = vec.float().flatten().clone()
        if top_k is not None and top_k < v.numel():
            # keep only the top-k by |contrast|, zero the rest (sparse)
            idx = torch.topk(v.abs(), top_k).indices
            mask = torch.zeros_like(v)
            mask[idx] = 1.0
            v = v * mask
        delta = (sign * alpha * v)
        nz = int((v != 0).sum().item())
        total_n += nz

        def _add(module, inputs, _d=delta):
            act = inputs[0].clone()
            act = act + _d.to(act.device, act.dtype)
            return (act,) + tuple(inputs[1:])

        handles.append(text_model.layers[li].mlp.down_proj.register_forward_pre_hook(_add))
    policy._phys_handles = handles
    print(f"  [PHYSICAL-CAA-{direction}] {len(handles)} layers, {total_n} active neurons, "
          f"alpha={alpha:.1f}, top_k={top_k}")


def setup_coast(policy, coast_path: str, direction: str, beta: float,
                layer_lo: int | None = None, layer_hi: int | None = None):
    """COAST multiplicative conceptor gating on lm_expert MLP outputs.

    Math (arxiv:2605.17144):
      C_steer fitted offline as C_pos ∧ ¬C_neg  (Boolean conceptor algebra)
      M = (1−β)I + β·C_steer
      h' = h @ Mᵀ ⇒ per-token: h'ᵀ = M hᵀ  (multiplicative — key difference from CAA)

    direction: which conceptor to gate with, as an alias:
      "pos"/"success"/"high" → steer toward the positive group (success or high)
      "neg"/"failure"/"low"  → steer toward the negative group
    beta ∈ [0,1] — 0 = no steering, 1 = full projection.
    layer_lo/layer_hi: restrict steering to expert layers [lo, hi) (None = all fitted).
    """
    clear_steering(policy)
    import torch as _torch
    art = _torch.load(coast_path, map_location="cpu", weights_only=False)
    if art.get("kind") != "coast":
        raise ValueError(f"Not a COAST artifact (kind={art.get('kind')}): {coast_path}")

    alias = {"pos": "pos", "success": "pos", "high": "pos",
             "neg": "neg", "failure": "neg", "low": "neg"}
    if direction not in alias:
        raise ValueError(f"direction must be one of {list(alias)}, got {direction}")
    key = alias[direction]

    layers = _expert_layers(policy)
    n_layers = len(layers)
    lo = 0 if layer_lo is None else layer_lo
    hi = n_layers if layer_hi is None else layer_hi
    handles, used = [], []

    for layer_key, data in art["layers"].items():
        li = int(layer_key)
        if not (0 <= li < n_layers) or not (lo <= li < hi):
            continue
        if key not in data:
            continue
        C = data[key].float()   # (d, d)  full contrastive conceptor C_steer

        def _hook(module, inputs, output, _C=C, _b=beta, _pol=policy):
            if not getattr(_pol, "_caa_gate", True):
                return output
            h = output[0] if isinstance(output, tuple) else output   # (B, T, d)
            C = _C.to(h.device, h.dtype)
            # Multiplicative gate M = (1−β)I + β·C_steer ⇒ h' = h·M = (1−β)h + β·(h·C)
            Csteer_h = h @ C                             # (B, T, d)
            h_new = (1.0 - _b) * h + _b * Csteer_h
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        handles.append(layers[li].mlp.register_forward_hook(_hook))
        used.append(li)

    policy._coast_handles = handles
    meaning = art.get("meaning", {})
    tgt = meaning.get(key, key)
    print(f"  [COAST→{tgt}] {len(handles)} layers {used}, beta={beta:.3f}, "
          f"split={art.get('split','?')}, positive_only={art.get('positive_only', False)}")


# ── Rollout ────────────────────────────────────────────────────────────────────

def run_one_rollout(
    episode: dict,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
) -> dict:
    """
    One closed-loop (on joints) / open-loop (on vision) rollout.

    Vision:  stored agentview frames replayed in order.
    State:   joint_pos[:6] from policy prediction at each step.
    Action:  policy prediction → 8D [joint_1..7_next, gripper_next].
    EEF-z:   Franka DH FK on predicted joint_1..7.
    """
    from lerobot.utils.control_utils import predict_action

    imgs      = episode["imgs"]        # (T, 256, 256, 3)
    joint_ref = episode["joint_pos"]   # (T, 7)
    gripper_ref = episode["gripper"]   # (T, 1)
    T = len(imgs)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # Calibrate FK: add offset so FK z matches stored world-frame z
    fk_offset = fit_fk_offset(episode)

    # Init 8D state: [7 joints, 1 gripper] — matches normalizer stats (8D)
    current_state8 = np.concatenate([
        joint_ref[0].astype(np.float64),
        gripper_ref[0].astype(np.float64),
    ])  # (8,)

    eef_heights = []
    pred_joints = []

    for t in range(T):
        img = imgs[t]
        obs = build_obs(img, current_state8.astype(np.float32), task)

        with torch.no_grad():
            action = predict_action(
                observation=obs,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
                task=task,
            )

        if torch.is_tensor(action):
            act_np = action.detach().cpu().numpy().reshape(-1)
        else:
            act_np = np.asarray(action, dtype=np.float64).reshape(-1)

        # action is 8D: [joint_1..7_next, gripper_next] — becomes next state
        current_state8 = act_np[:8].astype(np.float64)
        next_joints7   = current_state8[:7]
        eef_z = franka_fk_z(next_joints7) + fk_offset   # calibrated world-frame z

        eef_heights.append(eef_z)
        pred_joints.append(next_joints7.copy())

    return {
        "eef_heights": np.array(eef_heights, dtype=np.float64),
        "pred_joints": np.array(pred_joints, dtype=np.float64),
        "ref_eef_z":   episode["eef_pos"][:, 2].astype(np.float64),
        "T": T,
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def rollout_metrics(ri: int, result: dict, arc_type: str) -> dict:
    h  = result["eef_heights"]
    hc = h * 100
    T  = len(h)
    lo, hi = int(0.20 * T), int(0.80 * T)
    carry  = hc[lo:hi]
    safe   = lambda a, fn: float(fn(a)) if len(a) else float("nan")
    # Joint-level carry means (more sensitive than FK height, no FK error)
    pj = result.get("pred_joints")  # (T, 7) or None
    j1_mean = float(pj[lo:hi, 1].mean()) if pj is not None and hi > lo else float("nan")
    j3_mean = float(pj[lo:hi, 3].mean()) if pj is not None and hi > lo else float("nan")
    return {
        "rollout": ri, "arc_type": arc_type, "n_steps": T,
        "eef_max_cm":          safe(hc,    np.max),
        "eef_mean_cm":         safe(hc,    np.mean),
        "carry_peak_cm":       safe(carry, np.max),
        "carry_mean_cm":       safe(carry, np.mean),
        "carry_std_cm":        safe(carry, np.std),
        "eef_final_cm":        float(hc[-1]) if T > 0 else float("nan"),
        "ref_carry_peak_cm":   float(result["ref_eef_z"][lo:hi].max() * 100) if T > 0 else float("nan"),
        "carry_joint1_mean":   j1_mean,   # shoulder — correlates with height
        "carry_joint3_mean":   j3_mean,   # elbow  — correlates with height
    }


def bootstrap_ci(vals: np.ndarray, n=3000, seed=42):
    rng = np.random.default_rng(seed)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def cohen_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((b.mean() - a.mean()) / sp) if sp else float("nan")


def aggregate(rows: list[dict], key="carry_peak_cm"):
    vals = np.array([r[key] for r in rows if key in r], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {}
    lo, hi = bootstrap_ci(vals)
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "std":  float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
        "median": float(np.median(vals)),
        "ci95_lo": lo, "ci95_hi": hi,
        "min": float(vals.min()), "max": float(vals.max()),
    }


# ── Video ──────────────────────────────────────────────────────────────────────

def save_rollout_video(
    result: dict,
    episode: dict,
    condition: str,
    rollout_idx: int,
    out_path: Path,
    fps: float = 15.0,
):
    imgs = episode["imgs"]   # (T, 256, 256, 3)
    eef  = result["eef_heights"]
    ref  = result["ref_eef_z"]
    T    = len(eef)
    W, H = 512, 256

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W*2, H)
    )

    peak_pred = float(np.max(eef) * 100)
    peak_ref  = float(np.max(ref) * 100)

    # Zoomed bar scale: show relevant height range instead of 0-140 cm
    all_z = np.concatenate([eef * 100, ref * 100])
    bar_min_z = max(0.0, float(np.min(all_z)) - 10.0)   # 10 cm below minimum
    bar_max_z = float(np.max(all_z)) + 8.0               # 8 cm above maximum

    for t in range(T):
        # Left: policy input image
        left = cv2.resize(
            cv2.cvtColor(imgs[min(t, len(imgs)-1)], cv2.COLOR_RGB2BGR), (W, H)
        )
        # Right: EEF height bar chart
        right = np.ones((H, W, 3), dtype=np.uint8) * 30

        z_pred_cm = eef[t] * 100
        z_ref_cm  = ref[min(t, len(ref)-1)] * 100
        bar_h     = H - 70
        bar_w     = 50
        z_range   = bar_max_z - bar_min_z

        def _draw_bar(x_left, value_cm, color_bgr, label):
            frac = np.clip((value_cm - bar_min_z) / z_range, 0.0, 1.0)
            fill = int(bar_h * frac)
            y_bot = H - 40
            cv2.rectangle(right, (x_left, y_bot - bar_h), (x_left+bar_w, y_bot), (60,60,60), -1)
            cv2.rectangle(right, (x_left, y_bot - fill),  (x_left+bar_w, y_bot), color_bgr, -1)
            cv2.putText(right, f"{value_cm:.1f}cm",
                        (x_left, y_bot - bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(right, label,
                        (x_left, y_bot + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (200,200,200), 1, cv2.LINE_AA)

        _draw_bar(80,  z_pred_cm, (0, 200, 80),  "pred")
        _draw_bar(180, z_ref_cm,  (80, 140, 255), "ref")

        # Scale labels
        y_bot = H - 40
        cv2.putText(right, f"{bar_max_z:.0f}", (10, y_bot - bar_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (140,140,140), 1, cv2.LINE_AA)
        cv2.putText(right, f"{bar_min_z:.0f}", (10, y_bot),         cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (140,140,140), 1, cv2.LINE_AA)
        cv2.putText(right, f"t={t}/{T}", (W-80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1, cv2.LINE_AA)
        cv2.putText(right, f"cond={condition}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,100), 1, cv2.LINE_AA)
        cv2.putText(right, f"peak={peak_pred:.1f}cm", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,220,80), 1, cv2.LINE_AA)

        frame = np.concatenate([left, right], axis=1)
        writer.write(frame)

    writer.release()


# ── EEF height trajectory plot (paper figure) ─────────────────────────────────

def plot_comparison(
    traces_by_cond: dict[str, np.ndarray],  # {condition: (n_rollouts, T)}
    out_path: Path,
    ref_traces: dict[str, np.ndarray] | np.ndarray | None = None,
    metrics_by_cond: dict[str, list] | None = None,
):
    """
    Paper-ready EEF height trajectory comparison with zoomed y-axis.
    Three subplots: (1) trajectory bands, (2) carry-peak box plots,
    (3) delta-from-baseline bar chart (if 'none' condition present).

    ref_traces: either a dict {"high": arr, "low": arr} of demo references
                (plotted as separate dashed lines to show the target gap),
                or a single array (legacy), or None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {
        "none":          "#555555",
        "caa":           "#E05A2B",
        "caa_high":      "#C0392B",
        "caa_low":       "#E8A87C",
        "keyword_high":  "#2B8BE0",
        "keyword_low":   "#E0B82B",
        "physical_high": "#1ABC9C",
        "physical_low":  "#8E44AD",
    }
    LABELS = {
        "none":          "Baseline",
        "caa":           "CAA",
        "caa_high":      "CAA (high)",
        "caa_low":       "CAA (low)",
        "keyword_high":  "Keyword-high",
        "keyword_low":   "Keyword-low",
        "physical_high": "Physical-high",
        "physical_low":  "Physical-low",
    }

    has_delta = metrics_by_cond is not None and "none" in metrics_by_cond
    n_panels = 3 if has_delta else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))

    # ── Left: mean ± std bands zoomed to actual data range ───────────────────
    ax = axes[0]
    N = 100
    all_means_cm = []       # steered/baseline means (drive the zoom)
    demo_means_cm = []       # demo high/low means (shown but do not force zoom)

    # Demo references: separate high (red) and low (blue) dashed lines
    REF_STYLE = {
        "high": ("#C0392B", "Demo HIGH (dataset)"),
        "low":  ("#2471A3", "Demo LOW (dataset)"),
    }
    if isinstance(ref_traces, dict):
        for arc, arr in ref_traces.items():
            if arr is None or len(arr) == 0:
                continue
            rm = np.nanmean(arr * 100, axis=0)
            rs = np.nanstd(arr * 100, axis=0)
            xs = np.linspace(0, 1, len(rm))
            color, lbl = REF_STYLE.get(arc, ("gray", f"Demo {arc}"))
            ax.plot(xs, rm, "--", color=color, lw=1.6, alpha=0.9, label=lbl)
            ax.fill_between(xs, rm - rs, rm + rs, color=color, alpha=0.08)
            demo_means_cm.append(rm)
    elif ref_traces is not None:
        rm = np.nanmean(ref_traces * 100, axis=0)
        xs = np.linspace(0, 1, len(rm))
        ax.plot(xs, rm, "--", color="lightgray", lw=1.2, alpha=0.8, label="Demo ref")
        demo_means_cm.append(rm)

    for cond, traces in traces_by_cond.items():
        resampled = []
        for tr in traces:
            valid = tr[np.isfinite(tr)]
            if len(valid) > 1:
                resampled.append(np.interp(np.linspace(0, 1, N),
                                           np.linspace(0, 1, len(valid)),
                                           valid * 100))
        if not resampled:
            continue
        arr = np.stack(resampled)
        mn, sd = arr.mean(0), arr.std(0)
        xs = np.linspace(0, 1, N)
        color = COLORS.get(cond, "black")
        ax.plot(xs, mn, color=color, lw=2.0, label=LABELS.get(cond, cond))
        ax.fill_between(xs, mn - sd, mn + sd, color=color, alpha=0.18)
        all_means_cm.append(mn)

    # Zoom y-axis to include BOTH steered lines and demo high/low references,
    # so the target gap (what steering must achieve) is always visible.
    zoom_src = all_means_cm + [
        np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(m)), m)
        for m in demo_means_cm
    ]
    if zoom_src:
        carry_vals = np.concatenate([m[int(0.2*N):int(0.8*N)] for m in zoom_src])
        y_lo = float(np.nanmin(carry_vals)) - 3.0
        y_hi = float(np.nanmax(carry_vals)) + 3.0
        ax.set_ylim(y_lo, y_hi)

    ax.axvspan(0.20, 0.80, color="#FFFFAA", alpha=0.25, label="Carry phase")
    ax.set_xlabel("Normalised timestep", fontsize=11)
    ax.set_ylabel("EEF z-height (cm)", fontsize=11)
    ax.set_title("EEF Height Trajectory", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # ── Middle: box plot of carry-phase peak heights (zoomed) ────────────────
    ax2 = axes[1]
    box_data, box_labels, box_colors = [], [], []
    all_peaks_flat = []

    def _carry_peaks(traces_arr):
        out = []
        for tr in traces_arr:
            valid = np.asarray(tr)[np.isfinite(tr)] * 100
            if len(valid) > 4:
                lo = int(0.20 * len(valid)); hi = int(0.80 * len(valid))
                out.append(float(np.max(valid[lo:hi])))
        return out

    # Dataset reference boxes first (ground-truth high/low demo range)
    if isinstance(ref_traces, dict):
        for arc in ("low", "high"):
            arr = ref_traces.get(arc)
            if arr is None or len(arr) == 0:
                continue
            peaks = _carry_peaks(arr)
            if peaks:
                box_data.append(peaks)
                box_labels.append(f"Dataset {arc.upper()}")
                box_colors.append(REF_STYLE.get(arc, ("gray", ""))[0])
                all_peaks_flat.extend(peaks)

    for cond, traces in traces_by_cond.items():
        peaks = _carry_peaks(traces)
        if peaks:
            box_data.append(peaks)
            box_labels.append(LABELS.get(cond, cond))
            box_colors.append(COLORS.get(cond, "black"))
            all_peaks_flat.extend(peaks)

    if box_data:
        bp = ax2.boxplot(box_data, patch_artist=True,
                         medianprops={"color": "white", "lw": 2},
                         whiskerprops={"lw": 1.2},
                         capprops={"lw": 1.2})
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax2.set_xticklabels(box_labels, fontsize=9, rotation=15, ha="right")
        # Zoom y-axis to actual peak range
        if all_peaks_flat:
            p_lo = min(all_peaks_flat) - 1.0
            p_hi = max(all_peaks_flat) + 1.0
            ax2.set_ylim(p_lo, p_hi)
    ax2.set_ylabel("Carry-phase peak EEF z (cm)", fontsize=11)
    ax2.set_title("Peak Height Distribution", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    # ── Right: paired delta from baseline ────────────────────────────────────
    if has_delta:
        ax3 = axes[2]
        base_peaks = np.array([r["carry_peak_cm"] for r in metrics_by_cond["none"]])
        delta_means, delta_stds, delta_labels, delta_colors = [], [], [], []
        for cond in traces_by_cond:
            if cond == "none" or cond not in metrics_by_cond:
                continue
            steered = np.array([r["carry_peak_cm"] for r in metrics_by_cond[cond]])
            n = min(len(base_peaks), len(steered))
            deltas = steered[:n] - base_peaks[:n]
            delta_means.append(float(deltas.mean()))
            delta_stds.append(float(deltas.std(ddof=1) / math.sqrt(n) if n > 1 else 0))
            delta_labels.append(LABELS.get(cond, cond))
            delta_colors.append(COLORS.get(cond, "black"))

        if delta_means:
            xs3 = np.arange(len(delta_means))
            bars = ax3.bar(xs3, delta_means, yerr=delta_stds, capsize=5,
                           color=delta_colors, alpha=0.75, error_kw={"lw": 1.5})
            ax3.axhline(0, color="black", lw=0.8, ls="--")
            ax3.set_xticks(xs3)
            ax3.set_xticklabels(delta_labels, fontsize=9, rotation=15, ha="right")
            ax3.set_ylabel("Δ carry-peak vs baseline (cm)", fontsize=11)
            ax3.set_title("Steering Effect (mean ± SE)", fontsize=12)
            ax3.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Figure: {out_path}")


# ── CSV / JSON helpers ─────────────────────────────────────────────────────────

def save_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="LIBERO offline activation-steering evaluation for SmolVLA."
    )
    ap.add_argument("--policy-path",  default="ethanCSL/svla_franka_pick_n_place_vla_steering_libero")
    ap.add_argument("--hdf5",         nargs="+", required=True, help="One or more HDF5 files (high + low)")
    ap.add_argument("--task",         default="Pick up the black bowl and place it on the plate.")
    ap.add_argument("--conditions",   nargs="+",
                    default=["none", "caa", "keyword_high", "keyword_low"],
                    help="Steering conditions to evaluate")
    ap.add_argument("--n-rollouts",   type=int, default=20,
                    help="Total rollouts per condition (cycles through episodes)")
    ap.add_argument("--caa-path",     default=None, help="Path to libero_caa_vectors.pt")
    ap.add_argument("--caa-alpha",    type=float, default=2.0)
    ap.add_argument("--neurons-json", default=None, help="Path to libero_height_neurons.json")
    ap.add_argument("--keyword-alpha",type=float, default=8.0,
                    help="Steering magnitude (ADD mode: always pushes in direction; SET: absolute)")
    ap.add_argument("--keyword-top-n", type=int, default=10,
                    help="Keep only top-N neurons per concept")
    ap.add_argument("--steering-mode", choices=["add", "set"], default="add",
                    help="add: activation+=alpha (recommended); set: activation=alpha (paper exact)")
    ap.add_argument("--bidirectional", action="store_true",
                    help="Also suppress opposing-concept neurons at -alpha for stronger contrast")
    ap.add_argument("--save-video",   action="store_true")
    ap.add_argument("--video-rollouts", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fps",          type=float, default=15.0)
    ap.add_argument("--out-dir",      default="outputs/libero_eval")
    ap.add_argument("--device",       default=None)
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_safe_torch_device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # FK validation
    print("\nValidating Franka DH FK ...")
    validate_fk(args.hdf5[0])

    # Load episodes
    print("\nLoading HDF5 episodes ...")
    episodes = load_all_episodes(args.hdf5)
    if not episodes:
        raise ValueError("No episodes found.")

    # Policy
    print(f"\nLoading policy: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy_cfg = policy.config
    policy_cfg.device = str(device)
    policy.eval().to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
    )

    # ── Condition loop ─────────────────────────────────────────────────────────
    all_traces:   dict[str, list] = {}
    all_metrics:  dict[str, list] = {}
    ref_traces_by_arc: dict[str, list] = {}   # {"high": [...], "low": [...]}
    ref_collected = False

    for cond in args.conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond}")
        print(f"{'='*60}")

        # Setup steering
        if cond == "none":
            clear_steering(policy)
        elif cond == "caa":
            if not args.caa_path:
                print("  [skip] --caa-path not provided")
                continue
            setup_caa(policy, args.caa_path, alpha=args.caa_alpha)
        elif cond == "caa_high":
            # ADD the (high - low) CAA vector → steers toward high carry
            if not args.caa_path:
                print("  [skip] --caa-path not provided")
                continue
            setup_caa(policy, args.caa_path, alpha=+args.caa_alpha)
        elif cond == "caa_low":
            # SUBTRACT the (high - low) CAA vector → steers toward low carry
            if not args.caa_path:
                print("  [skip] --caa-path not provided")
                continue
            setup_caa(policy, args.caa_path, alpha=-args.caa_alpha)
        elif cond == "keyword_high":
            if not args.neurons_json:
                print("  [skip] --neurons-json not provided")
                continue
            setup_keyword_neurons(
                policy, args.neurons_json, "high",
                alpha=args.keyword_alpha, top_n=args.keyword_top_n,
                mode=args.steering_mode, bidirectional=args.bidirectional,
            )
        elif cond == "keyword_low":
            if not args.neurons_json:
                print("  [skip] --neurons-json not provided")
                continue
            setup_keyword_neurons(
                policy, args.neurons_json, "low",
                alpha=args.keyword_alpha, top_n=args.keyword_top_n,
                mode=args.steering_mode, bidirectional=args.bidirectional,
            )
        else:
            print(f"  [skip] Unknown condition '{cond}'")
            continue

        cond_metrics = []
        cond_traces  = []

        for ri in tqdm(range(args.n_rollouts), desc=f"  {cond} rollouts"):
            ep = episodes[ri % len(episodes)]
            result = run_one_rollout(ep, policy, preprocessor, postprocessor, device, args.task)

            metrics = rollout_metrics(ri, result, ep["arc_type"])
            cond_metrics.append(metrics)
            cond_traces.append(result["eef_heights"])

            if not ref_collected:
                # Split demo reference by arc_type so high/low dataset
                # trajectories can be plotted separately (target gap).
                ref_traces_by_arc.setdefault(ep["arc_type"], []).append(result["ref_eef_z"])

            if args.save_video and ri in args.video_rollouts:
                vpath = out_dir / "videos" / cond / f"rollout_{ri:03d}.mp4"
                save_rollout_video(result, ep, cond, ri, vpath, fps=args.fps)

        if not ref_collected:
            ref_collected = True

        all_traces[cond]  = cond_traces
        all_metrics[cond] = cond_metrics

        # Per-condition outputs
        save_csv(out_dir / f"rollout_metrics_{cond}.csv", cond_metrics)

        carry_agg = aggregate(cond_metrics, "carry_peak_cm")
        mean_agg  = aggregate(cond_metrics, "carry_mean_cm")
        summary = {
            "condition": cond,
            "n_rollouts": len(cond_metrics),
            "carry_peak_cm": carry_agg,
            "carry_mean_cm": mean_agg,
        }
        # Cohen's d vs baseline (only if baseline already computed)
        if cond != "none" and "none" in all_metrics:
            base_peaks = np.array([r["carry_peak_cm"] for r in all_metrics["none"]], dtype=np.float64)
            this_peaks = np.array([r["carry_peak_cm"] for r in cond_metrics], dtype=np.float64)
            summary["cohens_d_vs_baseline"] = cohen_d(base_peaks, this_peaks)

        with open(out_dir / f"summary_{cond}.json", "w") as f:
            json.dump(summary, f, indent=2)

        if carry_agg:
            print(f"\n  carry_peak: mean={carry_agg['mean']:.2f} ± {carry_agg['std']:.2f} cm, "
                  f"CI95=[{carry_agg['ci95_lo']:.2f}, {carry_agg['ci95_hi']:.2f}]  n={carry_agg['n']}")

    # ── Save raw traces ────────────────────────────────────────────────────────
    for cond, traces in all_traces.items():
        if not traces:
            continue
        max_T = max(len(t) for t in traces)
        arr = np.full((len(traces), max_T), np.nan, dtype=np.float64)
        for i, t in enumerate(traces):
            arr[i, :len(t)] = t
        np.savez_compressed(out_dir / f"eef_traces_{cond}.npz",
                            traces_m=arr, traces_cm=arr*100)

    # ── Comparison figure ──────────────────────────────────────────────────────
    if all_traces:
        # Build arrays {cond: (n, max_T)}
        trace_arrays = {}
        for cond, traces in all_traces.items():
            max_T = max(len(t) for t in traces)
            arr = np.full((len(traces), max_T), np.nan, dtype=np.float64)
            for i, t in enumerate(traces):
                arr[i, :len(t)] = t
            trace_arrays[cond] = arr

        # Build separate demo-reference arrays per arc_type (high / low)
        ref_arr_by_arc = {}
        for arc, traces in ref_traces_by_arc.items():
            if not traces:
                continue
            max_T = max(len(t) for t in traces)
            arr = np.full((len(traces), max_T), np.nan, dtype=np.float64)
            for i, t in enumerate(traces):
                arr[i, :len(t)] = t
            ref_arr_by_arc[arc] = arr

        plot_comparison(
            trace_arrays,
            out_dir / "eef_trajectory_comparison.png",
            ref_arr_by_arc,
            metrics_by_cond=all_metrics,
        )

    # ── Cross-condition summary table ──────────────────────────────────────────
    if len(all_metrics) > 1:
        print("\n" + "="*80)
        print("Cross-condition Summary (carry-phase peak EEF height, cm)")
        print("="*80)
        print(f"{'Condition':<18} {'Mean':>7} {'±Std':>6} {'CI95_lo':>8} {'CI95_hi':>8} "
              f"{'Cohen-d':>8} {'J1-mean':>8}")
        print("-"*80)
        base_peaks = None
        base_j1 = None
        for cond, metrics in all_metrics.items():
            peaks = np.array([r["carry_peak_cm"] for r in metrics])
            agg   = aggregate(metrics, "carry_peak_cm")
            j1vals = np.array([r.get("carry_joint1_mean", float("nan")) for r in metrics])
            j1_mean_str = f"{np.nanmean(j1vals):+.4f}" if np.any(np.isfinite(j1vals)) else "   n/a"
            d = ""
            if cond == "none":
                base_peaks = peaks
                base_j1 = np.nanmean(j1vals)
            elif base_peaks is not None:
                d = f"{cohen_d(base_peaks, peaks):+.3f}"
            print(f"  {cond:<16} {agg.get('mean', 0):7.2f} "
                  f"{agg.get('std',0):6.2f} "
                  f"{agg.get('ci95_lo',0):8.2f} "
                  f"{agg.get('ci95_hi',0):8.2f} "
                  f"{d:>8}  {j1_mean_str}")
        if base_j1 is not None:
            print(f"\n  (J1-mean = mean shoulder-joint angle during carry; "
                  f"baseline J1={base_j1:+.4f} rad)")

    print(f"\n[✓] All outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
