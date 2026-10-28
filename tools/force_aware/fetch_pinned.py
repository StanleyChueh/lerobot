"""Explicit public-data acquisition for the offline audit (no weights or hardware)."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

REVISIONS = {
    "openarm_pringles_lerobot_real_v00": "2c1bc860347c44fc0fbae7567469d36ee49baa8d",
    "openarm_pringles_lerobot_real_10_v00": "cebdee98115ab6827a4ec2655335c536b5dd085b",
    "openarm_insertbattery_lerobot_real_v00": "24eb901651f35bbab3ff7c375ce2dcd05abb1c16",
}


def read(url: str) -> bytes:
    """Read a public HTTPS artifact."""
    with urlopen(url, timeout=120) as response:  # public HTTPS URLs constructed below
        return response.read()


def main() -> None:
    """Fetch explicitly pinned datasets into a separate audit directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-video", action="store_true", help="Download first body-camera shard too")
    args = parser.parse_args()
    for name, sha in REVISIONS.items():
        repo = "ethanCSL/" + name
        api = json.loads(read(f"https://huggingface.co/api/datasets/{repo}/revision/{sha}"))
        if api["sha"] != sha:
            raise ValueError("Dataset revision mismatch")
        root = args.output / name
        root.mkdir(parents=True, exist_ok=True)
        (root / "api.json").write_text(json.dumps(api, indent=2))
        video_selected = False
        for item in sorted(api["siblings"], key=lambda x: x["rfilename"]):
            path = Path(item["rfilename"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe repository path")
            selected = path.parts[0] in ("meta", "data") or str(path) == "README.md"
            if (
                args.body_video
                and str(path).startswith("videos/observation.images.body_cam/")
                and not video_selected
            ):
                selected = video_selected = True
            if selected:
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(read(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{path}"))


if __name__ == "__main__":
    main()
