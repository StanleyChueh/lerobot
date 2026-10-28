# Validation and reproduction

The tested source base is `f10cbf97a0577fd4652dc55aff43eaf3c607a2df`; executable files match audited `530b253f262d6af92ea3b91d62abcae0295af59f` (README-only difference). Final feature commit is reported in the delivery response. No existing tracked source file was modified. All new training-independent code is under `tools/force_aware`; no baseline registration/configuration was changed.

## Commands executed successfully

From repository root, offline unit tests:

```bash
UV_CACHE_DIR=/tmp/force-aware-uv uv run --no-sync python -m unittest discover -s tests/force_aware -v
```

Result: **6 passed**. Uses NumPy and CPU PyTorch already installed in `.venv`.

CPU regression command (temporary pytest/matplotlib overlay fetched under `/tmp`, project dependencies unchanged):

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/force-aware-uv uv run --offline --no-sync --with pytest --with matplotlib python -m pytest tests/force_aware tests/processor/test_smolvla_processor.py -q -rs
```

Result: **13 passed, 4 skipped**. Four GPU-only cases skip: three require CUDA and one requires at least two GPUs. No GPU training or full VLM forward pass occurred. New tests are synthetic interface checks, not physical force measurements.

Actual reused diagnostics command:

```bash
UV_CACHE_DIR=/tmp/force-aware-uv uv run --offline --no-sync --with pytest --with matplotlib python tools/force_aware/audit.py \
  --diagnostics-source /tmp/force-aware-openarm/StanleyChueh-lerobot_openarm-9abbe12 \
  --data-root /tmp/force-aware-data \
  --output docs/force_aware/evidence \
  --checkpoint /home/csl/.cache/huggingface/hub/models--ethanCSL--openarm_pringles_lerobot_real_smolvla_10_v00/snapshots/8593bf2539eb663a536aac77c2ab6f33219a54ba \
  --annotations docs/force_aware/evidence/upstream_real_v00_annotations.csv
```

Result: completed. Baseline technical checks pass; real_10 stale input metadata WARN and two minor quality warnings; real_50 quality FAIL flags 14 demonstrations for review. This FAIL is a diagnostic heuristic outcome, not a failing software test. Video checks in this report explicitly cover metadata spans only.

Video command, executed for each downloaded first body-camera shard:

```bash
ffprobe -v error -count_frames -show_entries stream=codec_name,r_frame_rate,nb_read_frames,duration -of json /tmp/force-aware-data/openarm_pringles_lerobot_real_10_v00/videos/observation.images.body_cam/chunk-000/file-000.mp4
ffmpeg -v error -y -ss 1 -i /tmp/force-aware-data/openarm_pringles_lerobot_real_10_v00/videos/observation.images.body_cam/chunk-000/file-000.mp4 -frames:v 1 /tmp/openarm_pringles_lerobot_real_10_v00.png
```

See `evidence/video_probe.json` for all three actual results. Images were inspected for real_10 and insertion. No all-camera freshness/latency claim follows.

Ruff initially reported 14 style/docstring issues; these were fixed. Final checks:

```bash
UV_CACHE_DIR=/tmp/force-aware-uv uv run --no-sync --with ruff ruff check tools/force_aware tests/force_aware
UV_CACHE_DIR=/tmp/force-aware-uv uv run --no-sync --with ruff ruff format --check tools/force_aware tests/force_aware
UV_CACHE_DIR=/tmp/force-aware-uv uv run --no-sync python -m compileall -q tools/force_aware tests/force_aware
git diff --check
```

Result: all pass. The committed-diff check initially detected inherited CRLF annotation line endings; normalized the copied CSV to LF without changing values and rechecked.

## Data/source acquisition

Initial downloads queried public dataset APIs, recorded immutable SHAs, then downloaded all meta/data files and the first body-camera shard at those SHAs with Python urllib. The rerunnable pinned equivalent is:

```bash
UV_CACHE_DIR=/tmp/force-aware-uv uv run --no-sync python tools/force_aware/fetch_pinned.py --output /tmp/force-aware-data --body-video
```

The new fetch helper has compile validation only; its exact end-to-end rerun was not needed after the equivalent acquisition succeeded. Original blob SHA256 values are in `evidence/provenance.json`.

OpenArm source was fetched using `curl -L --fail --max-time 60 https://api.github.com/repos/StanleyChueh/lerobot_openarm/tarball -o /tmp/force-aware-openarm.tar.gz`, then only diagnostics, quality script, research reports/annotations and selected source files were extracted. For reproduction, pin the archive URL to commit `9abbe126128ce17f66c98c726688dac70b84a966`. Do not run its manifest wholesale: that includes sim datasets and other tasks outside this audit.

NeuralActuator `infer_actuator.py` inspected SHA256: `629d1f58103ecf047b7a9bf1cc462b43af40190f270629f58c3d006514217b8d`. It was read, not executed. Full paper pinned to arXiv v2. Dataset and checkpoint small-file hashes, Python/package versions are retained in provenance.

## Failed/limited attempts

- Git branch creation initially denied by read-only `.git`; approved escalation created the isolated branch.
- Sandbox DNS prevented initial public downloads; approved network escalation succeeded. Browser could not retrieve dataset raw JSON; direct public HTTPS did.
- Base `.venv` lacks pytest and matplotlib. Temporary `uv --with` overlay resolved this without updating `uv.lock` or project dependencies.
- First pytest attempt failed on an unrelated ROS auto-loaded plugin (`lark` missing). `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` resolved it; hardware/ROS plugins are unnecessary for these tests.
- Initial diagnostics on metadata-only downloads marked missing local video files FAIL. Final report disables file-presence checks and states its metadata-only scope; separately decoded body shards are documented above. This was not evidence that remote videos were missing.
- A final UV registry refresh hit sandbox DNS; reran with cached dependencies and `--offline`.
- Full optional-policy suite, full checkpoint restore, production telemetry contract, all-camera decoding and physical evaluation are not tested. These limitations gate production approval.

## Completion boundary

Offline phase deliverables and provisional tests are reviewable. Cross-repository integration is deliberately pending the finalized schema. Missing run identities, trial logs, calibration and force labels are scientific blockers, not inferred negative findings. Decision: **GO offline; NO-GO production/causal claims**.
