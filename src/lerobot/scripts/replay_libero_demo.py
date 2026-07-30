#!/usr/bin/env python3
"""
replay_libero_demo.py

Download LIBERO human demo for one task and save a full MP4 replay video.
Shows what the task is SUPPOSED to look like (ground-truth human demo).

Usage:
    python replay_libero_demo.py --task-idx 0 --demo-idx 0
    python replay_libero_demo.py --task-idx 0 --demo-idx 0 --suite libero_spatial
"""

import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np


BDDL_ROOT = Path(
    "/home/bruce/anaconda3/envs/libero_sim/lib/python3.10/"
    "site-packages/libero/libero/bddl_files"
)
DEMO_DIR  = Path("/home/bruce/datasets/libero_demos")
VIDEO_DIR = Path("/home/bruce/datasets/libero_demos/videos")


# ── Download ──────────────────────────────────────────────────────────────────

def download_demos(suite: str, dest: Path):
    """Download LIBERO demo HDF5 files from HuggingFace."""
    from libero.libero.utils.download_utils import download_from_huggingface
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {suite} demos from HuggingFace → {dest}")
    download_from_huggingface(dataset_name=suite, download_dir=str(dest))
    print("Download complete.")


def find_demo_file(suite: str, task_fname: str, dest: Path) -> Path | None:
    """Look for the HDF5 demo file for a given task."""
    task_stem = task_fname.replace(".bddl", "")
    # LIBERO names the demo file: <task_stem>_demo.hdf5
    candidates = [
        dest / suite / f"{task_stem}_demo.hdf5",
        dest / f"{suite}" / f"{task_stem}.hdf5",
        dest / f"{task_stem}_demo.hdf5",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Search recursively
    for p in dest.rglob("*.hdf5"):
        if task_stem[:30] in p.stem:
            return p
    return None


# ── Replay ────────────────────────────────────────────────────────────────────

def replay_and_save(bddl_path: Path, demo_hdf5: Path,
                    demo_idx: int, out_video: Path, fps: int = 20):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=["agentview"],
        camera_heights=256,
        camera_widths=256,
    )

    with h5py.File(demo_hdf5, "r") as f:
        demo_key = f"data/demo_{demo_idx}"
        if demo_key not in f:
            available = list(f["data"].keys())
            print(f"Demo index {demo_idx} not found. Available: {available}")
            env.close()
            return

        actions = f[f"{demo_key}/actions"][()]
        states  = f[f"{demo_key}/states"][()]
        print(f"Demo {demo_idx}: {len(actions)} steps, {len(states)} states")

    # Restore initial sim state and replay actions
    env.reset()
    obs = env.set_init_state(states[0])

    frames = []
    for t, action in enumerate(actions):
        obs, _, done, _ = env.step(action)
        img  = obs["agentview_image"]
        eef  = obs["robot0_eef_pos"]
        z_cm = eef[2] * 100

        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"z={z_cm:.1f}cm  t={t:04d}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, "HUMAN DEMO", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        frames.append(frame)
        if done:
            break

    env.close()

    # Save video
    out_video.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()

    peak_z = max(cv2.getTextSize("", cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]
                 for _ in range(1))  # dummy — compute from frames
    eef_zs = [f[0, 8] for f in frames]  # rough extraction not reliable here

    print(f"Saved {len(frames)}-frame video → {out_video}")
    print(f"Open with:  xdg-open {out_video}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite",     default="libero_spatial")
    ap.add_argument("--task-idx",  type=int, default=0)
    ap.add_argument("--demo-idx",  type=int, default=0)
    ap.add_argument("--demo-dir",  default=str(DEMO_DIR))
    ap.add_argument("--no-download", action="store_true",
                    help="Skip download (assume demos already present)")
    args = ap.parse_args()

    from libero.libero import benchmark
    suite_obj  = benchmark.get_benchmark_dict()[args.suite]()
    bddl_files = suite_obj.get_task_bddl_files()
    fname      = bddl_files[args.task_idx]
    bddl_path  = BDDL_ROOT / args.suite / fname
    task_stem  = fname.replace(".bddl", "")

    demo_dest  = Path(args.demo_dir)

    print(f"Task {args.task_idx}: {task_stem}")
    print(f"BDDL: {bddl_path}")

    # Download if needed
    demo_file = find_demo_file(args.suite, fname, demo_dest)
    if demo_file is None and not args.no_download:
        download_demos(args.suite, demo_dest)
        demo_file = find_demo_file(args.suite, fname, demo_dest)

    if demo_file is None:
        print(f"\nERROR: Could not find demo HDF5 for task '{task_stem}'")
        print(f"Expected location: {demo_dest / args.suite / (task_stem + '_demo.hdf5')}")
        print("Try downloading manually:")
        print(f"  cd ~/CSL/LIBERO && python benchmark_scripts/download_libero_datasets.py "
              f"--datasets {args.suite} --use-huggingface")
        return

    print(f"Demo file: {demo_file}")

    out_video = VIDEO_DIR / f"human_demo_{args.suite}_task{args.task_idx:02d}_demo{args.demo_idx}.mp4"
    replay_and_save(bddl_path, demo_file, args.demo_idx, out_video)


if __name__ == "__main__":
    main()
