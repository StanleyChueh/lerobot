#!/usr/bin/env python3
"""
convert_libero_object_to_lerobot.py
────────────────────────────────────
Convert official LIBERO libero_object HDF5 demos → LeRobot v2 dataset for
fine-tuning smolvla_base on tasks 0-7, keeping tasks 8-9 as genuinely unseen
objects (chocolate pudding, orange juice).

HDF5 source layout (per task file):
  data/demo_N/obs/agentview_rgb      (T, 128, 128, 3)  — third-person cam
  data/demo_N/obs/eye_in_hand_rgb    (T, 128, 128, 3)  — wrist cam
  data/demo_N/obs/ee_states          (T, 6)  — eef_pos(3) + eef_axisangle(3)
  data/demo_N/obs/gripper_states     (T, 2)  — gripper open/close
  data/demo_N/actions                (T, 7)  — 7D OSC delta

LeRobot output:
  observation.images.agentview           (256, 256, 3)  — renamed → camera1 in train
  observation.images.robot0_eye_in_hand  (256, 256, 3)  — renamed → camera2 in train
  observation.state                      (8,)           — ee_states + gripper_states
  action                                 (7,)

libero_object task order:
  0  alphabet_soup     1  cream_cheese    2  salad_dressing  3  bbq_sauce
  4  ketchup           5  tomato_sauce    6  butter          7  milk
  8  chocolate_pudding (HELD OUT)         9  orange_juice    (HELD OUT)

Usage:
  # Convert tasks 0-7 (train split, ≈400 demos)
  conda run -n lerobot python src/lerobot/scripts/convert_libero_object_to_lerobot.py \\
    --output-dir /home/bruce/datasets/lerobot_libero_object_8tasks \\
    --tasks 0 1 2 3 4 5 6 7

  # Quick smoke test (2 demos from task 0)
  conda run -n lerobot python src/lerobot/scripts/convert_libero_object_to_lerobot.py \\
    --output-dir /home/bruce/datasets/lerobot_libero_object_8tasks \\
    --tasks 0 --max-demos 2

  # All 10 tasks (train + held-out; useful to inspect everything)
  conda run -n lerobot python src/lerobot/scripts/convert_libero_object_to_lerobot.py \\
    --output-dir /home/bruce/datasets/lerobot_libero_object_all \\
    --tasks 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DEMO_DIR = Path("/home/bruce/datasets/libero_demos/libero_object")

# libero_object task order (from benchmark.get_benchmark_dict()['libero_object'])
TASKS = [
    (0, "pick up the alphabet soup and place it in the basket",
     "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"),
    (1, "pick up the cream cheese and place it in the basket",
     "pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5"),
    (2, "pick up the salad dressing and place it in the basket",
     "pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5"),
    (3, "pick up the bbq sauce and place it in the basket",
     "pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5"),
    (4, "pick up the ketchup and place it in the basket",
     "pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5"),
    (5, "pick up the tomato sauce and place it in the basket",
     "pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5"),
    (6, "pick up the butter and place it in the basket",
     "pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5"),
    (7, "pick up the milk and place it in the basket",
     "pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5"),
    (8, "pick up the chocolate pudding and place it in the basket",
     "pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5"),
    (9, "pick up the orange juice and place it in the basket",
     "pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5"),
]


def resize_and_flip(img_hwc: np.ndarray, target_size: int = 256, flip: bool = True) -> np.ndarray:
    """Resize 128→256 and flip upright (MuJoCo renders bottom-up)."""
    if img_hwc.shape[0] != target_size or img_hwc.shape[1] != target_size:
        img_hwc = cv2.resize(img_hwc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    if flip:
        img_hwc = np.ascontiguousarray(img_hwc[::-1])
    return img_hwc.astype(np.uint8)


def build_features(use_videos: bool = True) -> dict:
    img = {"dtype": "video" if use_videos else "image",
           "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.agentview":          dict(img),
        "observation.images.robot0_eye_in_hand": dict(img),
        "observation.state":  {"dtype": "float32", "shape": (8,), "names": None},
        "action":             {"dtype": "float32", "shape": (7,), "names": None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="Local output directory for the LeRobot dataset")
    ap.add_argument("--tasks", nargs="+", type=int, default=list(range(8)),
                    help="Task indices to include (default: 0-7 = train split)")
    ap.add_argument("--max-demos", type=int, default=None,
                    help="Max demos per task (default: all 50)")
    ap.add_argument("--no-flip", action="store_true",
                    help="Skip the vertical flip (use if images are already upright)")
    ap.add_argument("--no-videos", action="store_true",
                    help="Save frames as images instead of video (larger, faster)")
    ap.add_argument("--img-threads", type=int, default=4)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                    help="Push the finished dataset to this HF Hub repo (e.g. youruser/libero_object_8tasks)")
    args = ap.parse_args()

    flip = not args.no_flip
    output_dir = Path(args.output_dir)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Creating dataset at {output_dir}")
    ds = LeRobotDataset.create(
        repo_id=output_dir.name,
        fps=args.fps,
        features=build_features(use_videos=not args.no_videos),
        root=str(output_dir),
        robot_type="panda",
        use_videos=not args.no_videos,
        image_writer_threads=args.img_threads,
    )

    task_rows = [t for t in TASKS if t[0] in args.tasks]
    total_eps = 0

    for task_idx, instruction, hdf5_name in task_rows:
        hdf5_path = DEMO_DIR / hdf5_name
        if not hdf5_path.exists():
            print(f"  [skip] {hdf5_path} not found")
            continue

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(
                [k for k in f["data"].keys() if k.startswith("demo_")],
                key=lambda k: int(k.split("_")[1])
            )
            if args.max_demos:
                demo_keys = demo_keys[:args.max_demos]

            print(f"\n=== Task {task_idx}: {instruction}  ({len(demo_keys)} demos) ===")

            for dk in demo_keys:
                g = f[f"data/{dk}"]
                T = g["actions"].shape[0]

                agentview   = g["obs/agentview_rgb"][()]    # (T, 128, 128, 3)
                eye_in_hand = g["obs/eye_in_hand_rgb"][()]  # (T, 128, 128, 3)
                ee_states   = g["obs/ee_states"][()]        # (T, 6)
                gripper     = g["obs/gripper_states"][()]   # (T, 2)
                actions     = g["actions"][()]              # (T, 7)

                state8 = np.concatenate([ee_states, gripper], axis=1).astype(np.float32)

                for t in range(T):
                    ds.add_frame({
                        "observation.images.agentview":
                            resize_and_flip(agentview[t], flip=flip),
                        "observation.images.robot0_eye_in_hand":
                            resize_and_flip(eye_in_hand[t], flip=flip),
                        "observation.state": state8[t],
                        "action": actions[t].astype(np.float32),
                        "task": instruction,
                    })
                ds.save_episode()
                total_eps += 1

    print(f"\n[✓] Saved {total_eps} episodes → {output_dir}")

    if args.push_to_hub:
        print(f"Pushing to HF Hub: {args.push_to_hub}")
        ds.push_to_hub(repo_id=args.push_to_hub)
        print(f"[✓] Pushed → https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
