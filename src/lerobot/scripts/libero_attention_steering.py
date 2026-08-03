#!/usr/bin/env python3
"""
libero_attention_steering.py
────────────────────────────
Visualize the SmolVLA action-expert CROSS-ATTENTION over the agentview image
under none / caa_high / caa_low steering, on the SAME carry frame.

Purpose: show the model's INTERNAL response to height steering — i.e. that the
steering coherently changes (a) where the action expert attends in the scene and
(b) the commanded vertical action dz — rather than the carry height changing by luck.

The action expert cross-attends to the VLM prefix tokens (images/language/state).
We record those weights (`record_attn`), average over action-query tokens and heads,
pull out the agentview image tokens, reshape to a 2D map, and overlay it.

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_attention_steering.py \
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_60k \
    --caa-path outputs/caa_osc_60k_hf.pt --caa-alpha 3.0 \
    --high-hdf5 /home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5 \
    --frame-frac 0.5 --out outputs/attn_steer_60k.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lerobot.scripts.libero_eval_steering import setup_caa, clear_steering

_EMPTY_256 = np.zeros((256, 256, 3), dtype=np.uint8)
_EMPTY_480 = np.zeros((480, 640, 3), dtype=np.uint8)


def build_obs(agent, wrist, state8):
    return {
        "observation.images.camera1":        agent,
        "observation.images.camera2":        wrist,
        "observation.images.camera3":        _EMPTY_256,
        "observation.images.empty_camera_0": _EMPTY_480,
        "observation.state": state8.astype(np.float32),
    }


def eef_state_from_hdf5(eef_pos, eef_quat, gripper_qpos):
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([eef_pos.astype(np.float32),
                           quat2axisangle(eef_quat).astype(np.float32),
                           gripper_qpos.astype(np.float32)])


def extract_image_heat(attn_matrix, token_layout, image_index=0):
    """attn_matrix [Q,K] → mean over queries → per-image-token 1D heat for the camera."""
    mean_attn = attn_matrix.mean(dim=0)  # [K]
    for seg in token_layout:
        if seg.get("type") == "image" and seg.get("image_index") == image_index:
            return mean_attn[seg["start"]:seg["end"]]
    # fallback: first image segment
    for seg in token_layout:
        if seg.get("type") == "image":
            return mean_attn[seg["start"]:seg["end"]]
    return None


def heat_to_2d(heat_1d, size=256):
    heat = heat_1d.float().detach().cpu().numpy()
    n = heat.size
    side = int(round(n ** 0.5))
    if side * side != n:
        # factor closest to square
        for h in range(side, 0, -1):
            if n % h == 0:
                side_h, side_w = h, n // h
                break
        grid = heat.reshape(side_h, side_w)
    else:
        grid = heat.reshape(side, side)
    return cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)


def overlay(agent_rgb, heat2d):
    h = (heat2d - heat2d.min()) / (heat2d.max() - heat2d.min() + 1e-8)
    hm = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(agent_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base, 0.55, hm, 0.45, 0)


def get_cross_heat(vlm_with_expert, image_index=0, size=256):
    recs = getattr(vlm_with_expert, "attn_records", {}) or {}
    layers = sorted({k[0] for k in recs if k[1] == "expert_cross"})
    if not layers:
        return None
    attn_list = recs.get((layers[-1], "expert_cross"), [])
    if not attn_list:
        return None
    attn = attn_list[-1]                       # last denoising step
    if attn.ndim == 4:
        attn = attn.mean(dim=1)                # mean over heads → [B,Q,K]
    attn_matrix = attn[0].float()              # [Q,K]
    layout = getattr(vlm_with_expert, "_last_prefix_token_layout", None)
    if layout is None:
        return None
    heat_1d = extract_image_heat(attn_matrix, layout, image_index)
    return heat_to_2d(heat_1d, size) if heat_1d is not None and heat_1d.numel() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--caa-path", required=True)
    ap.add_argument("--caa-alpha", type=float, default=3.0)
    ap.add_argument("--high-hdf5",
                    default="/home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5")
    ap.add_argument("--episode", default="ep_000")
    ap.add_argument("--frame-frac", type=float, default=0.5, help="representative frame for the overlay (0-1)")
    ap.add_argument("--n-frames", type=int, default=15, help="carry frames to aggregate dz + attention over")
    ap.add_argument("--n-eps", type=int, default=4, help="episodes to aggregate over")
    ap.add_argument("--task", default="pick the akita black bowl between the plate "
                    "and the ramekin and place it on the plate")
    ap.add_argument("--out", default="outputs/attn_steer.png")
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy_path)

    vlm = policy.model.vlm_with_expert
    vlm.record_attn = True

    # gather carry frames across a few episodes
    frames = []
    repr_agent = None
    with h5py.File(args.high_hdf5, "r") as f:
        eps = sorted(k for k in f.keys() if k.startswith("ep_"))[:args.n_eps]
        for ei, ek in enumerate(eps):
            g = f[ek]
            T = len(g["agentview_image"])
            ids = np.linspace(int(0.2 * T), int(0.8 * T) - 1, args.n_frames).astype(int)
            for t in ids:
                a_img = g["agentview_image"][t]
                frames.append((a_img, g["eye_in_hand_image"][t],
                               eef_state_from_hdf5(g["eef_pos"][t], g["eef_quat"][t], g["gripper_qpos"][t])))
            if ei == 0:
                repr_agent = g["agentview_image"][int(args.frame_frac * T)]
    print(f"aggregating over {len(frames)} carry frames ({args.n_eps} eps × {args.n_frames})")

    conditions = [("none", 0.0), ("caa_high", +args.caa_alpha), ("caa_low", -args.caa_alpha)]
    dz_by_cond, heat_by_cond = {}, {}
    for name, a in conditions:
        (clear_steering(policy) if a == 0.0 else setup_caa(policy, args.caa_path, alpha=a))
        dzs, heats = [], []
        for agent, wrist, s8 in frames:
            policy.reset()
            vlm.attn_records = {}
            with torch.no_grad():
                act = predict_action(observation=build_obs(agent, wrist, s8), policy=policy,
                                     device=device, preprocessor=pre, postprocessor=post,
                                     use_amp=False, task=args.task)
            act = (act.detach().cpu().numpy() if torch.is_tensor(act) else np.asarray(act)).reshape(-1)
            dzs.append(float(act[2]))
            h = get_cross_heat(vlm, image_index=0, size=256)
            if h is not None:
                heats.append(h)
        dz_by_cond[name] = np.array(dzs)
        heat_by_cond[name] = np.mean(heats, axis=0) if heats else None
        print(f"  {name}: dz mean={np.mean(dzs):+.4f} ± {np.std(dzs):.4f}  (n={len(dzs)})")
    clear_steering(policy)

    # ── statistical "not luck" check ──
    hi, lo = dz_by_cond["caa_high"], dz_by_cond["caa_low"]
    frac = float(np.mean(hi > lo))
    print(f"\n  dz(caa_high) > dz(caa_low) on {frac*100:.0f}% of frames  "
          f"(mean Δ = {np.mean(hi)-np.mean(lo):+.4f})")

    # ── figure: overlays (mean attention) + high−low difference map ──
    def panel(name):
        h = heat_by_cond[name]
        p = overlay(repr_agent, h) if h is not None else cv2.cvtColor(repr_agent, cv2.COLOR_RGB2BGR)
        cv2.putText(p, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(p, f"dz={np.mean(dz_by_cond[name]):+.3f}", (6, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return p

    panels = [panel(c) for c, _ in conditions]
    # difference map (caa_high − caa_low): where steering shifts attention
    if heat_by_cond["caa_high"] is not None and heat_by_cond["caa_low"] is not None:
        d = heat_by_cond["caa_high"] - heat_by_cond["caa_low"]
        m = np.abs(d).max() + 1e-8
        dn = ((d / m) * 127 + 128).astype(np.uint8)
        dmap = cv2.applyColorMap(dn, cv2.COLORMAP_BWR if hasattr(cv2, "COLORMAP_BWR") else cv2.COLORMAP_JET)
        dpanel = cv2.addWeighted(cv2.cvtColor(repr_agent, cv2.COLOR_RGB2BGR), 0.5, dmap, 0.5, 0)
        cv2.putText(dpanel, "high - low", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        panels.append(dpanel)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.hstack(panels))
    print(f"[✓] saved {args.out}  (none | caa_high | caa_low | high−low diff; dz = mean commanded vertical action)")


if __name__ == "__main__":
    main()
