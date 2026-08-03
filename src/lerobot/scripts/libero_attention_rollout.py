#!/usr/bin/env python3
"""
libero_attention_rollout.py
───────────────────────────
One CLOSED-LOOP rollout with the SmolVLA action-expert cross-attention heatmap
overlaid on the RGB agentview at every frame → an .mp4 you can watch.

n_action_steps is forced to 1 so the policy re-queries every step, giving a fresh
attention map per frame (fully reactive closed loop).

Usage:
  conda run -n lerobot python src/lerobot/scripts/libero_attention_rollout.py \
    --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_60k \
    --condition none --task-idx 0 --rollout-idx 0 --max-steps 300 \
    --out outputs/attn_rollout_60k_none.mp4
  # under steering:
  ... --condition caa_high --caa-path outputs/caa_osc_60k_hf.pt --caa-alpha 3.0
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lerobot.scripts.libero_osc_eval import (
    make_env, eef_state, build_obs, find_demo_file, load_demo_init_states,
    BDDL_ROOT, _bowl_keys,
)
from lerobot.scripts.libero_attention_steering import get_cross_heat, overlay
from lerobot.scripts.libero_eval_steering import setup_caa, clear_steering


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--condition", default="none", choices=["none", "caa_high", "caa_low"])
    ap.add_argument("--caa-path", default=None)
    ap.add_argument("--caa-alpha", type=float, default=3.0)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--task", default="pick the akita black bowl between the plate "
                    "and the ramekin and place it on the plate")
    ap.add_argument("--rollout-idx", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--out", default="outputs/attn_rollout.mp4")
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device
    from libero.libero import benchmark

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device)
    policy.config.n_action_steps = 1            # fresh attention every frame
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy_path)

    vlm = policy.model.vlm_with_expert
    vlm.record_attn = True

    # steering condition
    if args.condition == "none":
        clear_steering(policy)
    elif args.condition == "caa_high":
        setup_caa(policy, args.caa_path, alpha=+args.caa_alpha)
    elif args.condition == "caa_low":
        setup_caa(policy, args.caa_path, alpha=-args.caa_alpha)

    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    fname = suite_obj.get_task_bddl_files()[args.task_idx]
    env = make_env(BDDL_ROOT / args.suite / fname)
    inits = load_demo_init_states(find_demo_file(args.suite, fname))
    init = inits[args.rollout_idx % len(inits)]

    policy.reset(); pre.reset(); post.reset()
    env.reset()
    obs = env.set_init_state(init)
    for _ in range(5):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))

    bowls = _bowl_keys(obs)
    b0 = {b: float(obs[b][2]) for b in bowls}
    bpk = dict(b0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (256, 256))

    for t in range(args.max_steps):
        agent = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        s8 = eef_state(obs)
        vlm.attn_records = {}
        with torch.no_grad():
            act = predict_action(observation=build_obs(agent, wrist, s8), policy=policy,
                                 device=device, preprocessor=pre, postprocessor=post,
                                 use_amp=False, task=args.task)
        act = (act.detach().cpu().numpy() if torch.is_tensor(act) else np.asarray(act)).reshape(-1)
        heat = get_cross_heat(vlm, image_index=0, size=256)
        frame = overlay(agent, heat) if heat is not None else cv2.cvtColor(agent, cv2.COLOR_RGB2BGR)

        z = float(obs["robot0_eef_pos"][2]) * 100
        cv2.putText(frame, f"{args.condition}  z={z:.1f}cm", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"dz={act[2]:+.3f}  attn=action-expert cross", (6, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

        obs, _, done, _ = env.step(act[:7].astype(np.float32))
        for b in bowls:
            bpk[b] = max(bpk[b], float(obs[b][2]))
        if done:
            break

    writer.release()
    env.close()
    lifted = max(bowls, key=lambda b: bpk[b] - b0[b]) if bowls else None
    grasped = lifted is not None and bpk[lifted] - b0[lifted] > 0.03
    print(f"[✓] saved {args.out}  ({t+1} frames, grasped={grasped})")


if __name__ == "__main__":
    main()
