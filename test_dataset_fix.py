#!/usr/bin/env python

from pathlib import Path
import shutil
import pandas as pd

root = Path(
    "/home/bruce/.cache/huggingface/lerobot/ethanCSL/"
    "svla_koch_pick_n_place_vla_steering_height_experiment_setup"
)

episodes_root = root / "meta" / "episodes"

rename_cols = {
    "videos/observation.images.side/chunk_index": "videos/observation.images.wrist/chunk_index",
    "videos/observation.images.side/file_index": "videos/observation.images.wrist/file_index",
    "videos/observation.images.side/from_timestamp": "videos/observation.images.wrist/from_timestamp",
    "videos/observation.images.side/to_timestamp": "videos/observation.images.wrist/to_timestamp",

    "videos/observation.images.wrist/chunk_index": "videos/observation.images.top/chunk_index",
    "videos/observation.images.wrist/file_index": "videos/observation.images.top/file_index",
    "videos/observation.images.wrist/from_timestamp": "videos/observation.images.top/from_timestamp",
    "videos/observation.images.wrist/to_timestamp": "videos/observation.images.top/to_timestamp",
}

backup_root = root / "meta" / "episodes_backup_before_video_key_fix"

if backup_root.exists():
    shutil.rmtree(backup_root)

shutil.copytree(episodes_root, backup_root)
print("Backup saved to:", backup_root)

for parquet_path in sorted(episodes_root.glob("*/*.parquet")):
    df = pd.read_parquet(parquet_path)

    missing = [c for c in rename_cols if c not in df.columns]
    if missing:
        print(f"\nSkipping missing columns warning in {parquet_path}:")
        for c in missing:
            print("  missing:", c)

    df = df.rename(columns={k: v for k, v in rename_cols.items() if k in df.columns})

    # Remove duplicate columns if pandas created any during rename.
    df = df.loc[:, ~df.columns.duplicated()]

    df.to_parquet(parquet_path, index=False)
    print("Updated:", parquet_path)

print("\nDone.")
