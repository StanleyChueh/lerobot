"""Offline orchestration of OpenArm diagnostics; supply a separately pinned source checkout.

No downloads, dataset writes, training, or hardware imports. Run with --help.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    """Run pinned external diagnostics and freeze subset descriptors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotations", type=Path, help="Upstream real_v00 human image annotations")
    args = parser.parse_args()
    sys.path.insert(0, str(args.diagnostics_source.resolve()))
    from tools.dataset_diagnostics.checkpoint import compare_checkpoint, load_checkpoint_meta
    from tools.dataset_diagnostics.compat import _action_lead
    from tools.dataset_diagnostics.loader import load_dataset
    from tools.dataset_diagnostics.openarm import run_quality
    from tools.dataset_diagnostics.technical import run_technical_checks

    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    views = {}
    for root in sorted(args.data_root.glob("openarm_*")):
        view = load_dataset(root=root, label=root.name)
        views[root.name] = view
        checks = run_technical_checks(view, check_videos=False)
        # Handover-specific rules are invalid for insertion.
        if "pringles" in root.name:
            quality, _, quality_checks, _ = run_quality(view)
            quality.to_csv(args.output / (root.name + "_quality.csv"))
            checks += quality_checks
        if args.checkpoint and "real_10" in root.name:
            checks += compare_checkpoint(view, load_checkpoint_meta(str(args.checkpoint), offline=True))
        report[root.name] = {
            "sha": json.loads((root / "api.json").read_text())["sha"],
            "episodes": view.info["total_episodes"],
            "frames": len(view.data),
            "video_scope": "Metadata spans only; full video decoding NOT checked",
            "action_lead_frames_proxy": _action_lead(view, list(range(14))),
            "start_joint_std": np.stack([x[0] for x in view.vectors("observation.state").values()]).std(0),
            "checks": [c.to_dict() for c in checks],
        }
    pool = views["openarm_pringles_lerobot_real_v00"]
    small = views["openarm_pringles_lerobot_real_10_v00"]

    def digest(view: Any, episode: int) -> str:
        rows = view.data[view.data.episode_index == episode]
        return hashlib.sha256(
            b"".join(np.stack(rows[k]).tobytes() for k in ("observation.state", "action"))
        ).hexdigest()

    hashes = {e: digest(pool, e) for e in range(50)}
    report["real_10_matches"] = {
        e: [p for p, h in hashes.items() if h == digest(small, e)] for e in range(10)
    }
    # Freeze subsets before outcomes are available. Random subsets are independent draws, may overlap.
    report["controlled_subsets"] = {
        str(seed): sorted(np.random.default_rng(seed).choice(50, size=10, replace=False).tolist())
        for seed in (17, 29, 43)
    }
    quality = pd.read_csv(args.output / (pool.label + "_quality.csv")).set_index("episode_index")
    report["subset_descriptors"] = {}
    for label, ids in {"original": list(range(10)), **report["controlled_subsets"]}.items():
        q = quality.loc[ids]
        report["subset_descriptors"][label] = {
            "episodes": ids,
            "frames": int(q.length.sum()),
            "flagged_for_review": int(q.exclude.sum()),
            "median_overlap_frames": float(q.handover_overlap.median()),
            "median_right_path_joint_units": float(q.path_right.median()),
        }
    if args.annotations:
        from tools.dataset_diagnostics.coverage import kernel_effective_n

        annotations = pd.read_csv(args.annotations).set_index("episode_index")
        for label, ids in {
            "original": list(range(10)),
            "pool": list(range(50)),
            **report["controlled_subsets"],
        }.items():
            points = annotations.loc[ids, ["u", "v"]].to_numpy()
            report.setdefault("image_start_descriptors", {})[label] = {
                "min_uv_px": points.min(0),
                "max_uv_px": points.max(0),
                "kernel_effective_n_48px": kernel_effective_n(points, 48),
                "scope": "Inherited human annotations; reference-point consistency not revalidated; no metric calibration",
            }
    (args.output / "audit.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=lambda x: (
                x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else str(x)
            ),
        )
    )


if __name__ == "__main__":
    main()
