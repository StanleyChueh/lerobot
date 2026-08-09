#!/usr/bin/env python

from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    repo_id = "ethanCSL/svla_franka_pick_n_place_vla_steering_libero_height_two_cams"
    dataset_dir = Path("~/.cache/huggingface/lerobot") / repo_id

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    api = HfApi()
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_dir,
    )

if __name__ == "__main__":
    main()