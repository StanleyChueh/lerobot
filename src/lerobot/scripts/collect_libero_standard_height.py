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


def _scene_object_xy(obs):
    """xy of all task objects (bowls/plate/ramekin/...) — always in the camera view."""
    xy = []
    for k, v in obs.items():
        if not k.endswith("_pos") or not hasattr(v, "__len__") or len(v) != 3:
            continue
        if "to_" in k or any(s in k for s in ("robot", "eef", "gripper", "hand")):
            continue
        xy.append(np.asarray(v[:2], dtype=float))
    return np.array(xy) if xy else None


def randomize_object_xy(env, obs, obj_pos_key, radius, rng, margin=0.02, min_sep=0.12):
    """Move a free object (e.g. the plate) to a random xy within `radius` (m) of its
    current position, CLAMPED into the bounding box of the scene's objects (± margin,
    so it stays in view) AND kept at least `min_sep` (m) from every OTHER object so it
    never overlaps/stacks on the bowl or ramekin (which would make grasping pull both
    objects off the table). Keeps z + orientation. Returns True if it moved.

    Decorrelates target position from carry height (like the real-robot recipe) so the
    model must visually servo to the target instead of memorizing one fixed place spot."""
    base = obj_pos_key[:-4] if obj_pos_key.endswith("_pos") else obj_pos_key  # "plate_1"
    jname = base + "_joint0" if (base + "_joint0") in env.sim.model.joint_names else None
    if jname is None:
        for jn in env.sim.model.joint_names:
            if base.split("_")[0] in jn and "joint" in jn:
                jname = jn
                break
    if jname is None:
        return False
    try:
        addr = env.sim.model.get_joint_qpos_addr(jname)
        start = addr[0] if isinstance(addr, tuple) else addr
        cur = env.sim.data.qpos[start:start + 2].copy()
        # other objects (exclude the one being moved) — the plate must stay clear of these
        others = []
        for k, v in obs.items():
            if not k.endswith("_pos") or not hasattr(v, "__len__") or len(v) != 3:
                continue
            if k == obj_pos_key or "to_" in k or any(s in k for s in ("robot", "eef", "gripper", "hand")):
                continue
            others.append(np.asarray(v[:2], dtype=float))
        allxy = _scene_object_xy(obs)
        lo, hi = ((allxy.min(0) - margin, allxy.max(0) + margin)
                  if allxy is not None and len(allxy) >= 2 else (None, None))
        # sample a spot that is in-view AND ≥ min_sep from every other object
        for _ in range(50):
            cand = cur + rng.uniform(-radius, radius, size=2)
            if lo is not None:
                cand = np.clip(cand, lo, hi)
            if others and min(np.linalg.norm(cand - o) for o in others) < min_sep:
                continue
            env.sim.data.qpos[start:start + 2] = cand
            env.sim.forward()
            return True
        return False   # no non-overlapping spot found → keep original (already valid)
    except Exception:
        return False


def collect_episode(env, carry_z, demo_actions, demo_init, obj_key, tgt_key,
                    randomize_plate=0.0, rng=None):
    """Replay grasp → waypoint carry at carry_z → place. Returns (frames, success).
    frames = list of (obs, action). If randomize_plate>0, the plate is moved to a
    random xy (radius m) after reset, and the scripted place targets the new spot."""
    obs = env.reset()
    obs = env.set_init_state(demo_init)
    if randomize_plate > 0 and rng is not None:
        randomize_object_xy(env, obs, tgt_key, randomize_plate, rng)
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    tgt_pos = obs[tgt_key].copy()   # read AFTER the plate move → place targets new spot
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
    # bowl must end ON the plate (xy within 6cm, resting height) — matches the eval's on_plate
    success = grasp_ok and dist_xy < 0.06 and final_bowl[2] > 0.90
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
    ap.add_argument("--randomize-plate", type=float, default=0.0,
                    help="move the plate to a random xy within this radius (m) per episode "
                         "(decorrelates target position from carry height; e.g. 0.10). "
                         "Only applies to scripted arcs, not --arc natural.")
    ap.add_argument("--arc-z-high", type=float, default=ARC_Z["high"],
                    help="carry EEF world-z (m) for the HIGH arc (default 1.30; try 1.20 for less-extreme)")
    ap.add_argument("--arc-z-low", type=float, default=ARC_Z["low"],
                    help="carry EEF world-z (m) for the LOW arc (default 1.10; try 1.08 for less-extreme)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for plate randomization")
    ap.add_argument("--max-retries", type=int, default=8,
                    help="with --randomize-plate, retry a new plate position up to this many "
                         "times until the task COMPLETES; only completed episodes are saved")
    args = ap.parse_args()

    if args.smoke:
        args.n_eps = 2
        if args.task_idx is None:
            args.task_idx = 0

    arc_z = {"high": args.arc_z_high, "low": args.arc_z_low}
    rng = np.random.default_rng(args.seed)
    if args.randomize_plate > 0 and args.arc == "natural":
        print("[!] --randomize-plate is ignored for --arc natural (the demo's fixed place "
              "motion can't reach a moved plate). Use it with high/low/both.")

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

    n_ok = n_skip = 0
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
            n_target = min(args.n_eps, len(demos))
            can_retry = args.randomize_plate > 0 and not natural

            def _save(frames):
                for obs, action in frames:
                    ds.add_frame({
                        "observation.images.agentview":          obs["agentview_image"].astype(np.uint8),
                        "observation.images.robot0_eye_in_hand": obs["robot0_eye_in_hand_image"].astype(np.uint8),
                        "observation.state": eef_state(obs),
                        "action": action.astype(np.float32),
                        "task": instruction,
                    })
                ds.save_episode()

            def _try(ep_i):
                da, di = demos[ep_i]
                try:
                    if natural:
                        return collect_episode_natural(env, da, di, obj_key, tgt_key)
                    return collect_episode(env, arc_z[arc], da, di, obj_key, tgt_key,
                                           randomize_plate=args.randomize_plate, rng=rng)
                except Exception as e:
                    print(f"    ep {ep_i} {arc}: EXCEPTION {e}")
                    return None, False

            saved = 0
            ok_demos = []
            # ── pass 1: try each unique demo (with per-demo retries) ──
            for ep_i in range(n_target):
                frames, success = None, False
                for _ in range(args.max_retries if can_retry else 1):
                    frames, success = _try(ep_i)
                    if success and frames is not None:
                        break
                if success and frames is not None:
                    _save(frames); saved += 1; n_ok += 1; ok_demos.append(ep_i)
                    if ep_i < 3 or ep_i % 10 == 0:
                        print(f"    task{task_idx} {arc} ep{ep_i}: OK ({len(frames)} frames)")
                else:
                    print(f"    task{task_idx} {arc} ep{ep_i}: fail after retries → will top up from a good demo")
            # ── top-up: reach the target count using KNOWN-good demos + fresh randomization ──
            if can_retry and saved < n_target and ok_demos:
                cap = (n_target - saved) * args.max_retries * 3
                att = j = 0
                while saved < n_target and att < cap:
                    ep_i = ok_demos[j % len(ok_demos)]; j += 1; att += 1
                    frames, success = _try(ep_i)
                    if success and frames is not None:
                        _save(frames); saved += 1; n_ok += 1
                        print(f"    task{task_idx} {arc} top-up (demo {ep_i}): OK  [{saved}/{n_target}]")
            if saved < n_target:
                n_skip += (n_target - saved)
                print(f"    [!] task{task_idx} {arc}: only {saved}/{n_target} (could not reach target)")
        env.close()

    print(f"\n[✓] Saved {n_ok} episodes — ALL task-completed"
          + (f"  ({n_skip} short of target — even top-up failed)" if n_skip else " (target count reached)"))
    print(f"    dataset: {ds.root}")
    if args.push:
        print("Pushing to hub ...")
        ds.push_to_hub()
    print(f"\nTrain with:  lerobot-train --policy.type=smolvla "
          f"--dataset.repo_id={args.repo_id} --policy.load_vlm_weights=true ...")


if __name__ == "__main__":
    main()
