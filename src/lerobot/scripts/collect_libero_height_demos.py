#!/usr/bin/env python3
"""
collect_libero_height_demos.py

HYBRID pick-and-place demo collector for LIBERO with controlled EEF arc height.
Used for: CAA steering vector computation (high-arc vs low-arc contrast).

Why hybrid:
    Hand-scripting a bowl grasp is fragile (the gripper closes on empty cavity).
    Instead we REPLAY each LIBERO human demo through the grasp + initial lift
    (guaranteed successful grasp, bowl airborne), then HAND OFF to a waypoint
    controller for the variable-height carry + place. Only the carry height
    differs between "high" and "low" — everything else matches real human demos.

    high arc: EEF z=1.30 during carry  (bowl carried ~1.25 m, ~28 cm above table)
    low  arc: EEF z=1.10 during carry  (bowl carried ~1.05 m, ~ 8 cm above table)

Requires the LIBERO human demos to be downloaded first (see replay_libero_demo.py):
    demos live under /home/bruce/datasets/libero_demos/<suite>/<task>_demo.hdf5

Usage:
    python collect_libero_height_demos.py --arc both --n-eps 50 --task-idx 0
    python collect_libero_height_demos.py --arc both --n-eps 50            # all 10 tasks
    python collect_libero_height_demos.py --arc high --n-eps 3 --task-idx 0 --save-video
    python collect_libero_height_demos.py --inspect --task-idx 0
"""

import argparse
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

BDDL_ROOT = Path(
    "/home/bruce/anaconda3/envs/libero_sim/lib/python3.10/"
    "site-packages/libero/libero/bddl_files"
)
OUTPUT_DIR = Path("/home/bruce/datasets/libero_height_demos")
DEMO_DIR   = Path("/home/bruce/datasets/libero_demos")   # LIBERO human demos

# EEF world-z during horizontal carry phase (table objects sit at z≈0.970)
ARC_Z = {
    "high": 1.30,   # bowl carried ~28 cm above table — clear high arc
    "low":  1.10,   # bowl carried ~ 8 cm above table — clearly lower
}

# Robosuite Panda gripper (LIBERO convention): -1 = open, +1 = close  (verified empirically)
GRIPPER_OPEN  = -1.0
GRIPPER_CLOSE =  1.0

PLACE_Z_OFFSET   = 0.03    # release height above target surface
HANDOFF_LIFT     = 0.02    # bowl must rise this much above grasp-z before handoff
HANDOFF_MAX_EXTRA = 30     # cap replay steps after gripper closes, before handoff
CARRY_KP         = 1.5     # gentle gain for controller carry (avoids dropping bowl)
CARRY_CLIP       = 0.5     # per-step action clip during carry

VERBOSE = False   # set via --verbose; prints per-phase eef diagnostics


# ── Env factory ──────────────────────────────────────────────────────────────

def make_env(bddl_path: Path):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=["agentview"],
        camera_heights=256,
        camera_widths=256,
    )


# ── Human-demo file lookup ────────────────────────────────────────────────────

def find_demo_file(suite: str, task_fname: str) -> Path | None:
    """Locate the LIBERO human-demo HDF5 for a task."""
    stem = task_fname.replace(".bddl", "")
    candidates = [
        DEMO_DIR / suite / f"{stem}_demo.hdf5",
        DEMO_DIR / suite / f"{stem}.hdf5",
    ]
    for c in candidates:
        if c.exists():
            return c
    for p in (DEMO_DIR / suite).glob("*.hdf5") if (DEMO_DIR / suite).exists() else []:
        if stem[:30] in p.stem:
            return p
    return None


# ── Object / target position lookup ─────────────────────────────────────────

# Maps BDDL filename keywords → obs-key prefix that holds its position
KEYWORD_TO_OBS_PREFIX = {
    "black_bowl": "akita_black_bowl",
    "bowl":       "akita_black_bowl",
    "cube":       "cube",
    "can":        "can",
    "bottle":     "wine_bottle",
    "wine_bottle":"wine_bottle",
    "mug":        "mug",
    "plate":      "plate",
    "rack":       "rack",
    "stove":      "flat_stove",
    "cabinet":    "wooden_cabinet",
    "ramekin":    "glazed_rim_porcelain_ramekin",
    "cream_cheese": "cream_cheese",
    "cookies":    "cookies",
}


def find_pos_key(obs: dict, keyword: str) -> str | None:
    """Return the obs key whose name contains `keyword` and ends with `_pos`."""
    keyword = keyword.lower()
    for k in obs:
        if keyword in k.lower() and k.endswith("_pos"):
            return k
    # try abbreviated match on each word in keyword
    for word in keyword.split("_"):
        if len(word) < 3:
            continue
        for k in obs:
            if word in k.lower() and k.endswith("_pos"):
                return k
    return None


def parse_task_objects(bddl_fname: str) -> tuple[str, str]:
    """
    Extract (pick_keyword, place_keyword) from BDDL filename.
    E.g. "pick_up_the_black_bowl_...and_place_it_on_the_plate.bddl"
         → ("black_bowl", "plate")
    """
    name = bddl_fname.replace(".bddl", "").lower()

    # Pattern: "pick_up_the_<obj>_..._place_it_on/in_the_<target>"
    m = re.search(r"pick_up_the_(\w+).*?(?:place_it_(?:on|in)_(?:the_)?(\w+))", name)
    if m:
        return m.group(1), m.group(2)

    # Fallback: first token after "pick_up_the" and last token
    tokens = name.split("_")
    try:
        i = tokens.index("the") + 1
        pick = tokens[i]
    except (ValueError, IndexError):
        pick = "bowl"
    place = tokens[-1]
    return pick, place


def get_object_and_target(obs: dict, bddl_fname: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    """
    Return (obj_pos, target_pos, obj_key, tgt_key).
    Raises RuntimeError if not found.
    """
    pick_kw, place_kw = parse_task_objects(bddl_fname)

    obj_key = find_pos_key(obs, pick_kw)
    tgt_key = find_pos_key(obs, place_kw)

    if obj_key is None:
        raise RuntimeError(
            f"Cannot find pick-object obs key for keyword '{pick_kw}'. "
            f"Available pos keys: {[k for k in obs if k.endswith('_pos')]}"
        )
    if tgt_key is None or tgt_key == obj_key:
        raise RuntimeError(
            f"Cannot find target obs key for keyword '{place_kw}'. "
            f"Available pos keys: {[k for k in obs if k.endswith('_pos')]}"
        )

    return obs[obj_key].copy(), obs[tgt_key].copy(), obj_key, tgt_key


# ── Waypoint controller ───────────────────────────────────────────────────────

def move_to(env, obs, target_xyz, gripper, hold_xy=None,
            max_steps=120, tol=0.012, kp=CARRY_KP, clip=CARRY_CLIP):
    """
    Gentle P-controller for the CARRY phase (bowl already in hand).
    Low gain + small clip so the grasped object is not shaken loose.
    If hold_xy is given, xy is servoed to that fixed point (only z travels).
    Returns (final_obs, obs_list, act_list).
    """
    obs_list, act_list = [], []
    for _ in range(max_steps):
        e = obs["robot0_eef_pos"]
        tgt = target_xyz.copy()
        if hold_xy is not None:
            tgt[0], tgt[1] = hold_xy
        err = tgt - e
        if np.linalg.norm(err) < tol:
            break
        action = np.zeros(7)
        action[:3] = np.clip(kp * err, -clip, clip)
        action[6]  = gripper
        obs, _, done, _ = env.step(action)
        obs_list.append(obs)
        act_list.append(action.copy())
        if done:
            break
    return obs, obs_list, act_list


# ── Episode (hybrid: replay human grasp → controller carry) ────────────────────

def collect_episode(env, arc_type, bddl_fname, demo_actions, demo_init_state,
                    obj_key, tgt_key):
    """
    Hybrid pick-and-place at a controlled arc height.

    1. set_init_state to the human demo's start, replay demo actions through the
       grasp until the bowl is airborne  → guaranteed successful grasp.
    2. Hand off to the gentle waypoint controller: lift to carry_z, transport
       above target, lower, release, retract.

    Returns dict {images, eef_pos, joint_pos, actions, peak_eef_z,
                  carry_eef_z, success} or None on failure.
    """
    carry_z = ARC_Z[arc_type]

    obs = env.reset()
    obs = env.set_init_state(demo_init_state)
    # settle
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    tgt_pos = obs[tgt_key].copy()
    grip_seq = demo_actions[:, 6]
    close_idx = np.where(grip_seq > 0)[0]
    if len(close_idx) == 0:
        if VERBOSE:
            print("      [skip] demo never closes gripper")
        return None
    close_t = int(close_idx[0])

    all_obs, all_acts = [obs], [np.zeros(7)]

    def do(action):
        nonlocal obs
        obs, _, done, _ = env.step(action)
        all_obs.append(obs)
        all_acts.append(np.asarray(action, dtype=np.float32).copy())
        return done

    # ── Phase A: replay human demo through the grasp ──
    for t in range(min(close_t + 1, len(demo_actions))):
        do(demo_actions[t])

    grasp_z = obs["robot0_eef_pos"][2]
    bowl_grasp_z = obs[obj_key][2]

    # ── Phase B: continue replaying until bowl is airborne (or cap) ──
    extra = 0
    while extra < HANDOFF_MAX_EXTRA and (close_t + 1 + extra) < len(demo_actions):
        do(demo_actions[close_t + 1 + extra])
        extra += 1
        if obs[obj_key][2] > bowl_grasp_z + HANDOFF_LIFT:
            break

    handoff_qpos = obs["robot0_gripper_qpos"][0]
    handoff_bowl_z = obs[obj_key][2]

    # ── Controller helper (gentle, gripper held CLOSED) ──
    def carry_to(target, hold_xy=None, label="", **kw):
        nonlocal obs
        obs, ob, ac = move_to(env, obs, np.asarray(target, float),
                              GRIPPER_CLOSE, hold_xy=hold_xy, **kw)
        all_obs.extend(ob)
        all_acts.extend(ac)
        if VERBOSE:
            e = obs["robot0_eef_pos"]
            print(f"      [{label:12s}] eef={np.round(e,3)} "
                  f"bowl_z={obs[obj_key][2]*100:.1f}cm "
                  f"qpos={obs['robot0_gripper_qpos'][0]:.4f}")

    xy0 = obs["robot0_eef_pos"][:2].copy()

    # ── Phase C: lift straight up to carry_z (hold xy) ──
    carry_to([xy0[0], xy0[1], carry_z], hold_xy=xy0, label="C-lift", max_steps=90)
    carry_eef_z = float(obs["robot0_eef_pos"][2])

    # ── Phase D: transport above target at carry_z ──
    carry_to([tgt_pos[0], tgt_pos[1], carry_z], label="D-transport", max_steps=140)

    # ── Phase E: lower toward target ──
    place = tgt_pos.copy(); place[2] += PLACE_Z_OFFSET
    carry_to([tgt_pos[0], tgt_pos[1], place[2]],
             hold_xy=tgt_pos[:2], label="E-lower", max_steps=90)

    # ── Phase F: release ──
    for _ in range(25):
        a = np.zeros(7); a[6] = GRIPPER_OPEN
        do(a)

    # ── Phase G: retract up ──
    carry_to([tgt_pos[0], tgt_pos[1], carry_z], hold_xy=tgt_pos[:2],
             label="G-retract", max_steps=50)

    # ── Package ──
    images       = np.stack([o["agentview_image"]  for o in all_obs]).astype(np.uint8)
    eef_pos_traj = np.stack([o["robot0_eef_pos"]  for o in all_obs]).astype(np.float32)
    joint_pos    = np.stack([o["robot0_joint_pos"] for o in all_obs]).astype(np.float32)
    T = len(all_obs)
    actions = np.zeros((T, 7), dtype=np.float32)
    n = min(len(all_acts), T)
    actions[:n] = np.stack(all_acts[:n])

    peak_z = float(eef_pos_traj[:, 2].max())

    # Success: bowl ended up near the plate xy and was lifted clear off the table
    final_bowl = obs[obj_key]
    dist_xy = float(np.linalg.norm(final_bowl[:2] - tgt_pos[:2]))
    # grasp_ok: bowl was airborne at handoff (empty-air qpos≈0.0006; a held wall≳0.0012)
    grasp_ok = handoff_qpos > 0.0012 and handoff_bowl_z > bowl_grasp_z + 0.01
    success = grasp_ok and dist_xy < 0.08 and final_bowl[2] > 0.90

    if VERBOSE:
        print(f"      grasp_ok={grasp_ok} handoff_qpos={handoff_qpos:.4f} "
              f"final_bowl_xy_dist={dist_xy*100:.1f}cm success={success}")

    return {
        "images":     images,
        "eef_pos":    eef_pos_traj,
        "joint_pos":  joint_pos,
        "actions":    actions,
        "peak_eef_z": peak_z,
        "carry_eef_z": carry_eef_z,
        "success":    success,
    }


# ── Video writer ─────────────────────────────────────────────────────────────

def save_video(images: np.ndarray, eef_pos_traj: np.ndarray,
               path: Path, fps: int = 20):
    """Save episode frames as MP4 with EEF z-height overlay."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = images.shape[1], images.shape[2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )
    for t, (img, eef) in enumerate(zip(images, eef_pos_traj)):
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        z_cm = eef[2] * 100
        cv2.putText(frame, f"z={z_cm:.1f}cm  t={t:04d}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        writer.write(frame)
    writer.release()


# ── HDF5 writer ───────────────────────────────────────────────────────────────

def save_hdf5(episodes: list[dict], path: Path, arc_type: str, task_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["arc_type"]  = arc_type
        f.attrs["task_name"] = task_name
        f.attrs["n_episodes"] = len(episodes)
        f.attrs["mean_peak_eef_z"] = float(
            np.mean([ep["peak_eef_z"] for ep in episodes])
        )
        for i, ep in enumerate(episodes):
            g = f.create_group(f"ep_{i:03d}")
            g.create_dataset("agentview_image", data=ep["images"],    compression="gzip", compression_opts=4)
            g.create_dataset("eef_pos",         data=ep["eef_pos"])
            g.create_dataset("joint_pos",        data=ep["joint_pos"])
            g.create_dataset("actions",          data=ep["actions"])
            g.attrs["peak_eef_z"] = ep["peak_eef_z"]
            g.attrs["success"]    = bool(ep["success"])
    print(f"  Saved {len(episodes)} ep → {path}")


# ── Inspect mode ──────────────────────────────────────────────────────────────

def inspect(env, bddl_fname: str):
    obs = env.reset()
    print("\n── Obs keys (non-image) ──")
    for k, v in obs.items():
        if not k.endswith("image"):
            shape = v.shape if hasattr(v, "shape") else type(v)
            val   = np.round(v, 3) if isinstance(v, np.ndarray) and v.size <= 4 else "..."
            print(f"  {k:45s} {str(shape):15s} {val}")

    print("\n── Detected pick/place ──")
    try:
        obj_pos, tgt_pos, obj_key, tgt_key = get_object_and_target(obs, bddl_fname)
        print(f"  Pick object key : {obj_key}  →  {np.round(obj_pos, 3)}")
        print(f"  Place target key: {tgt_key}  →  {np.round(tgt_pos, 3)}")
    except RuntimeError as e:
        print(f"  ERROR: {e}")

    lo, hi = env.action_spec
    print(f"\n── Action spec ──  dim={lo.shape}  low={lo}  high={hi}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", choices=["high", "low", "both"], default="both")
    ap.add_argument("--n-eps",    type=int, default=50)
    ap.add_argument("--task-idx", type=int, default=None,
                    help="Single task index 0-9; omit for all 10")
    ap.add_argument("--suite",    default="libero_spatial")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--inspect",    action="store_true")
    ap.add_argument("--save-video", action="store_true",
                    help="Save MP4 video for every episode")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-phase EEF convergence diagnostics")
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    from libero.libero import benchmark
    suite_obj  = benchmark.get_benchmark_dict()[args.suite]()
    bddl_files = suite_obj.get_task_bddl_files()

    task_indices = ([args.task_idx] if args.task_idx is not None
                    else list(range(len(bddl_files))))
    arc_types    = ["high", "low"] if args.arc == "both" else [args.arc]
    output_dir   = Path(args.output_dir)

    for task_idx in task_indices:
        fname     = bddl_files[task_idx]
        bddl_path = BDDL_ROOT / args.suite / fname
        task_name = fname.replace(".bddl", "")

        print(f"\n{'='*65}")
        print(f"Task {task_idx:2d}: {task_name}")

        env = make_env(bddl_path)

        if args.inspect:
            inspect(env, fname)
            env.close()
            if args.task_idx is not None:
                sys.exit(0)
            continue

        # Load human demos for this task (source of grasps)
        demo_file = find_demo_file(args.suite, fname)
        if demo_file is None:
            print(f"  [skip] no human demo HDF5 found for this task under {DEMO_DIR/args.suite}")
            print(f"         download with: python src/lerobot/scripts/replay_libero_demo.py "
                  f"--suite {args.suite} --task-idx {task_idx}")
            env.close()
            continue

        with h5py.File(demo_file, "r") as df:
            demo_keys = sorted(df["data"].keys(), key=lambda k: int(k.split("_")[1]))
            demos = []
            for dk in demo_keys:
                demos.append((
                    df[f"data/{dk}/actions"][()],
                    df[f"data/{dk}/states"][()][0],   # initial sim state
                ))
        print(f"  Loaded {len(demos)} human demos from {demo_file.name}")

        # Determine obj/tgt keys once from a reset obs
        obs0 = env.reset()
        try:
            _, _, obj_key, tgt_key = get_object_and_target(obs0, fname)
        except RuntimeError as e:
            print(f"  [skip] {e}")
            env.close()
            continue

        for arc in arc_types:
            print(f"\n  Arc: {arc.upper()}  (carry z = {ARC_Z[arc]:.3f} m)")
            episodes  = []
            n_success = 0

            for ep_i in range(min(args.n_eps, len(demos))):
                demo_actions, demo_init = demos[ep_i]
                try:
                    ep = collect_episode(env, arc, fname, demo_actions, demo_init,
                                         obj_key, tgt_key)
                except Exception as e:
                    print(f"    ep {ep_i:03d}: EXCEPTION {e}")
                    continue
                if ep is None:
                    continue

                n_success += ep["success"]
                mark = "✓" if ep["success"] else "✗"
                print(
                    f"    ep {ep_i:03d}: {mark}  "
                    f"peak_z={ep['peak_eef_z']*100:.1f}cm  "
                    f"carry_z={ep['carry_eef_z']*100:.1f}cm  "
                    f"T={len(ep['images'])}"
                )

                if args.save_video:
                    vid_path = (output_dir / args.suite / "videos" /
                                f"task_{task_idx:02d}_{arc}_ep{ep_i:03d}.mp4")
                    save_video(ep["images"], ep["eef_pos"], vid_path)
                    print(f"           video → {vid_path}")

                episodes.append(ep)

            sr = n_success / max(len(episodes), 1)
            print(f"  → success {n_success}/{len(episodes)} ({sr:.0%})"
                  f"  mean_peak_z={np.mean([e['peak_eef_z'] for e in episodes])*100:.1f}cm"
                  f"  mean_carry_z={np.mean([e['carry_eef_z'] for e in episodes])*100:.1f}cm")

            if episodes:
                out = output_dir / args.suite / arc / f"task_{task_idx:02d}.hdf5"
                save_hdf5(episodes, out, arc_type=arc, task_name=task_name)

        env.close()

    print("\nAll done.")


if __name__ == "__main__":
    main()
