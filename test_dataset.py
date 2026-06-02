from pathlib import Path
import json
import pandas as pd

root = Path("/home/bruce/.cache/huggingface/lerobot/ethanCSL/svla_koch_pick_n_place_vla_steering_height_experiment_setup")

info = json.loads((root / "meta/info.json").read_text())
print("features:")
for k, v in info["features"].items():
    if "image" in k or "camera" in k or v.get("dtype") == "video":
        print(" ", k, v.get("dtype"))

print("\nepisode metadata columns:")
ep_files = sorted((root / "meta/episodes").glob("*/*.parquet"))
ep = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
for c in ep.columns:
    if c.startswith("videos/"):
        print(" ", c)

print("\nvideo folders:")
for p in sorted((root / "videos").glob("*")):
    print(" ", p.name)
