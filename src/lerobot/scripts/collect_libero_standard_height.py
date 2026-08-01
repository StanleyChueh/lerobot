#!/usr/bin/env python3
"""
collect_libero_standard_height.py
─────────────────────────────────
Re-collect LIBERO height-steering demos in the STANDARD LeRobot/LIBERO format so
the base model can actually complete the task in closed-loop (unlike the old
single-camera, joint-action, 60-episode model).

Standard LIBERO format (matches HuggingFaceVLA/libero, Pi0.5 ~97% success):
  observation.images.agentview           (256x256x3)  [third-person]
  observation.images.robot0_eye_in_hand  (256x256x3)  [WRIST — key for grasping]
  observation.state          8D  [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
  action                     7D  OSC-pose delta [dx,dy,dz,drx,dry,drz, gripper]

High/low steering contrast: each human demo is replayed through the (reliable)
grasp, then a gentle OSC waypoint controller carries the bowl at ARC_Z[high|low]
and places it on the plate. Same instruction for high & low → the model learns a
height-steerable representation. Object positions are randomized per episode
(each LIBERO human demo starts from a randomized reset).

Writes a LeRobot v2 dataset directly (ready to train).

Usage:
  conda run -n lerobot python src/lerobot/scripts/collect_libero_standard_height.py \\
    --repo-id ethanCSL/libero_spatial_height_std --suite libero_spatial \\
    --task-idx 0 --n-eps 50 --arc both               # one task
  conda run -n lerobot python src/lerobot/scripts/collect_libero_standard_height.py \\
    --repo-id ethanCSL/libero_spatial_height_std --suite libero_spatial \\
    --n-eps 50 --arc both                             # all 10 tasks
  ... --smoke   # tiny 2-demo test
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

BDDL_ROOT = Path("/home/bruce/anaconda3/envs/libero_sim/lib/python3.10/"
                 "site-packages/libero/libero/bddl_files")
DEMO_DIR = Path("/home/bruce/datasets/libero_demos")

ARC_Z = {"high": 1.30, "low": 1.10}   # EEF world-z during carry
GRIPPER_OPEN, GRIPPER_CLOSE = -1.0, 1.0
PLACE_Z_OFFSET = 0.03
HANDOFF_LIFT = 0.02
HANDOFF_MAX_EXTRA = 30
CARRY_KP, CARRY_CLIP = 1.5, 0.5

KEYWORD_TO_OBS_PREFIX = {
    "black_bowl": "akita_black_bowl", "bowl": "akita_black_bowl", "cube": "cube",
    "can": "can", "bottle": "wine_bottle", "mug": "mug", "plate": "plate",
    "cheese": "cream_cheese", "cookies": "cookies", "ramekin": "ramekin",
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


def find_pos_key(obs, keyword):
    keyword = keyword.lower()
    for k in obs:
        if keyword in k.lower() and k.endswith("_pos"):
            return k
    for word in keyword.split("_"):
        if len(word) < 3:
            continue
        for k in obs:
            if word in k.lower() and k.endswith("_pos"):
                return k
    return None


def parse_task_objects(bddl_fname):
    name = bddl_fname.replace(".bddl", "").lower()
    m = re.search(r"pick_up_the_(\w+).*?(?:place_it_(?:on|in)_(?:the_)?(\w+))", name)
    if m:
        return m.group(1), m.group(2)
    tokens = name.split("_")
    try:
        pick = tokens[tokens.index("the") + 1]
    except (ValueError, IndexError):
        pick = "bowl"
    return pick, tokens[-1]


def get_object_and_target(obs, bddl_fname):
    pick_kw, place_kw = parse_task_objects(bddl_fname)
    obj_key = find_pos_key(obs, pick_kw)
    tgt_key = find_pos_key(obs, place_kw)
    if obj_key is None:
        raise RuntimeError(f"pick key not found for '{pick_kw}'")
    if tgt_key is None or tgt_key == obj_key:
        raise RuntimeError(f"target key not found for '{place_kw}'")
    return obj_key, tgt_key


def eef_state(obs):
    """Standard 8D LIBERO state: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)."""
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([
        obs["robot0_eef_pos"].astype(np.float32),
        quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        obs["robot0_gripper_qpos"].astype(np.float32),
    ])


def move_to(env, obs, target_xyz, gripper, hold_xy=None,
            max_steps=120, tol=0.012, kp=CARRY_KP, clip=CARRY_CLIP):
    """Gentle OSC P-controller for the carry/place phase. Records (obs, action)."""
    frames = []
    for _ in range(max_steps):
        e = obs["robot0_eef_pos"]
        tgt = np.asarray(target_xyz, float).copy()
        if hold_xy is not None:
            tgt[0], tgt[1] = hold_xy
        err = tgt - e
        if np.linalg.norm(err) < tol:
            break
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(kp * err, -clip, clip)
        action[6] = gripper
        frames.append((obs, action.copy()))
        obs, _, done, _ = env.step(action)
        if done:
            break
    return obs, frames


def collect_episode(env, arc_type, bddl_fname, demo_actions, demo_init, obj_key, tgt_key):
    """Replay grasp → waypoint carry at ARC_Z → place. Returns (frames, success).
    frames = list of (obs, action)."""
    carry_z = ARC_Z[arc_type]
    obs = env.reset()
    obs = env.set_init_state(demo_init)
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    tgt_pos = obs[tgt_key].copy()
    grip_seq = demo_actions[:, 6]
    close_idx = np.where(grip_seq > 0)[0]
    if len(close_idx) == 0:
        return None, False
    close_t = int(close_idx[0])

    frames = []

    def do(action):
        nonlocal obs
        frames.append((obs, np.asarray(action, dtype=np.float32).copy()))
        obs, _, done, _ = env.step(action)
        return done

    # Phase A: replay demo through grasp
    for t in range(min(close_t + 1, len(demo_actions))):
        do(demo_actions[t])
    bowl_grasp_z = obs[obj_key][2]

    # Phase B: continue until bowl airborne
    extra = 0
    while extra < HANDOFF_MAX_EXTRA and (close_t + 1 + extra) < len(demo_actions):
        do(demo_actions[close_t + 1 + extra])
        extra += 1
        if obs[obj_key][2] > bowl_grasp_z + HANDOFF_LIFT:
            break
    handoff_qpos = obs["robot0_gripper_qpos"][0]
    handoff_bowl_z = obs[obj_key][2]

    def carry_to(target, hold_xy=None, **kw):
        nonlocal obs
        obs, fr = move_to(env, obs, target, GRIPPER_CLOSE, hold_xy=hold_xy, **kw)
        frames.extend(fr)

    xy0 = obs["robot0_eef_pos"][:2].copy()
    # Phase C: lift to carry_z
    carry_to([xy0[0], xy0[1], carry_z], hold_xy=xy0, max_steps=90)
    carry_eef_z = float(obs["robot0_eef_pos"][2])
    # Phase D: transport above target
    carry_to([tgt_pos[0], tgt_pos[1], carry_z], max_steps=140)
    # Phase E: lower
    place_z = tgt_pos[2] + PLACE_Z_OFFSET
    carry_to([tgt_pos[0], tgt_pos[1], place_z], hold_xy=tgt_pos[:2], max_steps=90)
    # Phase F: release
    for _ in range(25):
        a = np.zeros(7, dtype=np.float32); a[6] = GRIPPER_OPEN
        do(a)
    # Phase G: retract
    carry_to([tgt_pos[0], tgt_pos[1], carry_z], hold_xy=tgt_pos[:2], max_steps=40)

    final_bowl = obs[obj_key]
    dist_xy = float(np.linalg.norm(final_bowl[:2] - tgt_pos[:2]))
    grasp_ok = handoff_qpos > 0.0012 and handoff_bowl_z > bowl_grasp_z + 0.01
    success = grasp_ok and dist_xy < 0.08 and final_bowl[2] > 0.90
    return frames, success


def collect_episode_natural(env, demo_actions, demo_init, obj_key, tgt_key):
    """Replay the FULL human demo (natural grasp→carry→place) and record
    (obs, action) at each step, where action = the demo's own OSC 7D command.

    This is the cleanest official-LIBERO training data: real teleop task
    completion at natural height, no scripted lift/carry. Use this for the BASE
    model (paper arXiv:2509.00328 trains the base VLA on normal demos and derives
    the height-steering vector separately from high/low groups). Returns
    (frames, success); frames = list of (obs, action)."""
    env.reset()
    obs = env.set_init_state(demo_init)
    tgt_pos = obs[tgt_key].copy()
    frames = []
    for t in range(len(demo_actions)):
        a = np.asarray(demo_actions[t], dtype=np.float32)
        frames.append((obs, a.copy()))
        obs, _, done, _ = env.step(a)
        if done:
            break
    final_bowl = obs[obj_key]
    dist_xy = float(np.linalg.norm(final_bowl[:2] - tgt_pos[:2]))
    success = dist_xy < 0.08 and final_bowl[2] > 0.90
    return frames, success


def build_features(use_videos=True):
    img = {"dtype": "video" if use_videos else "image", "shape": (256, 256, 3),
           "names": ["height", "width", "channel"]}
    return {
        # LIBERO camera names (agentview = third-person, robot0_eye_in_hand = wrist).
        # These match your training rename_map (agentview→camera1, robot0_eye_in_hand→camera2).
        "observation.images.agentview":          dict(img),
        "observation.images.robot0_eye_in_hand": dict(img),
        "observation.state":  {"dtype": "float32", "shape": (8,), "names": None},
        "action":             {"dtype": "float32", "shape": (7,), "names": None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="ethanCSL/libero_spatial_height_std")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-idx", type=int, default=None, help="single task 0-9; omit=all")
    ap.add_argument("--n-eps", type=int, default=50, help="human demos per (task, arc)")
    ap.add_argument("--arc", choices=["high", "low", "both", "natural"], default="both",
                    help="high/low/both = scripted carry heights (for steering-vector data); "
                         "natural = replay full human demo (clean data for the BASE model)")
    ap.add_argument("--root", default=None, help="local dataset root (default HF cache)")
    ap.add_argument("--smoke", action="store_true", help="tiny 2-demo test")
    ap.add_argument("--push", action="store_true", help="push dataset to HF hub when done")
    ap.add_argument("--no-videos", action="store_true",
                    help="store frames as images (PNG) instead of encoding video "
                         "(much faster collection; larger on disk; trains fine)")
    ap.add_argument("--img-threads", type=int, default=4, help="parallel image-writer threads")
    args = ap.parse_args()

    if args.smoke:
        args.n_eps = 2
        if args.task_idx is None:
            args.task_idx = 0

    from libero.libero import benchmark
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    suite = benchmark.get_benchmark_dict()[args.suite]()
    bddl_files = suite.get_task_bddl_files()
    task_indices = [args.task_idx] if args.task_idx is not None else list(range(len(bddl_files)))
    arcs = ["high", "low"] if args.arc == "both" else [args.arc]
    natural = args.arc == "natural"

    print(f"Creating LeRobot dataset: {args.repo_id}")
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=20, features=build_features(use_videos=not args.no_videos),
        root=args.root, robot_type="panda", use_videos=not args.no_videos,
        image_writer_threads=args.img_threads,
    )

    n_ok = n_total = 0
    for task_idx in task_indices:
        fname = bddl_files[task_idx]
        bddl_path = BDDL_ROOT / args.suite / fname
        env = make_env(bddl_path)
        instruction = env.language_instruction
        print(f"\n=== Task {task_idx}: {instruction} ===")

        demo_file = find_demo_file(args.suite, fname)
        if demo_file is None:
            print(f"  [skip] no human demo for {fname}"); env.close(); continue
        with h5py.File(demo_file, "r") as df:
            keys = sorted(df["data"].keys(), key=lambda k: int(k.split("_")[1]))
            demos = [(df[f"data/{k}/actions"][()], df[f"data/{k}/states"][()][0]) for k in keys]

        obs0 = env.reset()
        try:
            obj_key, tgt_key = get_object_and_target(obs0, fname)
        except RuntimeError as e:
            print(f"  [skip] {e}"); env.close(); continue

        for arc in arcs:
            n_demo = min(args.n_eps, len(demos))
            for ep_i in range(n_demo):
                da, di = demos[ep_i]
                try:
                    if natural:
                        frames, success = collect_episode_natural(env, da, di, obj_key, tgt_key)
                    else:
                        frames, success = collect_episode(env, arc, fname, da, di, obj_key, tgt_key)
                except Exception as e:
                    print(f"    ep {ep_i} {arc}: EXCEPTION {e}"); continue
                if frames is None:
                    continue
                n_total += 1; n_ok += int(success)
                for obs, action in frames:
                    ds.add_frame({
                        "observation.images.agentview":          obs["agentview_image"].astype(np.uint8),
                        "observation.images.robot0_eye_in_hand": obs["robot0_eye_in_hand_image"].astype(np.uint8),
                        "observation.state": eef_state(obs),
                        "action": action.astype(np.float32),
                        "task": instruction,
                    })
                ds.save_episode()
                if ep_i < 3 or ep_i % 10 == 0:
                    print(f"    task{task_idx} {arc} ep{ep_i}: {'OK' if success else 'FAIL'} "
                          f"({len(frames)} frames)")
        env.close()

    print(f"\n[✓] Collected {n_total} episodes, {n_ok} successful "
          f"({100*n_ok/max(n_total,1):.0f}%)")
    print(f"    dataset: {ds.root}")
    if args.push:
        print("Pushing to hub ...")
        ds.push_to_hub()
    print(f"\nTrain with:  lerobot-train --policy.type=smolvla "
          f"--dataset.repo_id={args.repo_id} --policy.load_vlm_weights=true ...")


if __name__ == "__main__":
    main()
