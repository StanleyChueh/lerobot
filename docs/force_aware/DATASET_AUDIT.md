# Dataset audit — 2026-10-28

Decision: **GO for offline diagnosis; NO-GO for causal success attribution or force integration.**
All numbers below concern real recordings. No sim_10 or generated simulation results enter the comparison.

## Provenance and scope

Working base: `update_dev_v2`, `f10cbf97a0577fd4652dc55aff43eaf3c607a2df`.
Requested audited reference: `530b253f262d6af92ea3b91d62abcae0295af59f`.
`git diff 530b253f262d6af92ea3b91d62abcae0295af59f HEAD --stat` before changes showed only 26 README lines absent locally; executable code is identical. Work branch: `research/force-aware-offline-audit`.

Reused, unchanged: [OpenArm diagnostics](https://github.com/StanleyChueh/lerobot_openarm/tree/9abbe126128ce17f66c98c726688dac70b84a966/tools/dataset_diagnostics), including loader, technical checks, checkpoint checks, gripper quality adapter, action-lead proxy and coverage kernel. Source obtained as a read-only archive in `/tmp`; no hardware imports or cross-repository changes. The wrapper is [audit.py](../../tools/force_aware/audit.py).

| Dataset (owner `ethanCSL`) | Immutable Hub revision | Episodes | Frames | Duration at 30 Hz |
|---|---|---:|---:|---:|
| `openarm_pringles_lerobot_real_v00` | `2c1bc860347c44fc0fbae7567469d36ee49baa8d` | 50 | 12115 | 403.83 s |
| `openarm_pringles_lerobot_real_10_v00` | `cebdee98115ab6827a4ec2655335c536b5dd085b` | 10 | 2575 | 85.83 s |
| `openarm_insertbattery_lerobot_real_v00` | `24eb901651f35bbab3ff7c375ce2dcd05abb1c16` | 50 | 38178 | 1272.60 s |

Actual metadata, **all tabular trajectory shards**, episode tables, tasks and one body-camera shard per dataset were downloaded. Original info/stats, checks, quality CSVs and SHA256 manifest are in [evidence](evidence/provenance.json). Raw trajectories/videos remain in `/tmp/force-aware-data`; they are not committed. There are no torque, velocity, contact or force-label columns in the declared schemas.

## Schemas, clocks and samples

All three: v3.0, `robot_type=openarm_umeow`, float32 action/state `[16]`, identical ordered names:
`RJ1,LJ1,RJ2,LJ2,RJ3,LJ3,RJ4,LJ4,RJ5,LJ5,RJ6,LJ6,RJ7,LJ7,RJ8,LJ8` (each suffixed `.pos`). Indices 14/15 are grippers. Never replace this interleaved order with left-arm/right-arm blocks.

Metadata does not explicitly encode physical units. The OpenArm source and prior audit describe arm radians and gripper raw motor position coordinates; gripper values are not metres or Cartesian force. Treat this as source-supported interpretation pending the final schema and calibration. Fixture units in the adapter tests are invented, not hardware declarations.

Camera keys: `observation.images.body_cam`, `.wrist_cam`, `.right_wrist_cam`; all declared 480×640 RGB, AV1 at 30 Hz. The verified real_10 training mapping is right_wrist→camera1, wrist→camera2, body→camera3. Wrist handedness must be confirmed visually during any later deployment parity check.

Tabular schema, finite state/action, episode lengths, frame continuity and monotonic 1/30 s timestamp grid pass for all datasets. Declared video episode spans match tabular lengths. The action→state best-lag proxy is **2 frames** in all three (~67 ms), minimizing recorded arm position discrepancy over 0–6 frames. This is servo tracking evidence, not measured camera delay or proof that labels should be shifted. Acquisition/arrival timestamps and actual deployed command timing are unavailable.

`ffprobe -count_frames` decoded the first body-camera shards: real_10 **2575**, real_50 **12115**, battery **7067** frames; 30/1 Hz AV1. The battery figure is only its first shard. One image at video time 1 s per dataset was decoded; the real_10 image shows an upright Pringles can between two arms; battery shows a loose battery and holder. These spot checks do not establish task completion or frame freshness. Wrist decoding, later battery shards, duplicated visual content, capture timestamps and camera alignment remain **UNKNOWN**. The audit's camera PASS means metadata spans only, explicitly recorded in `video_scope`.

## Statistics and checkpoint compatibility

State/action mean/std recomputed from every tabular row agree with published stats within upstream tolerance (1e-3 relative). This validates the published full-dataset stats, not their suitability for a held-out split.

Actual cached real_10 checkpoint: `ethanCSL/openarm_pringles_lerobot_real_smolvla_10_v00` at `8593bf2539eb663a536aac77c2ab6f33219a54ba`. Small config/processor files and normalizer tensors were inspected; no weights downloaded or full policy instantiated. Checkpoint normalizer dimensions and values match real_10; action width 16 and all cameras map correctly. Input config still declares state `[6]`; its normalizer is `[16]`. Current `prepare_state` pads actual state 16→32 without truncation. Strict serving validators may reject stale metadata. The CPU test executes that method without constructing the VLM and checks a legacy 32-column projection state-dict round trip. This is **not** a full legacy checkpoint loading test.

Verified training: base `lerobot/smolvla_base` (revision unpinned), seed 1000, batch 8, 20000 optimizer steps, augmentation enabled (brightness/contrast/saturation/hue/sharpness/affine), all 10 episodes, no eval split, dataset revision unpinned. Saved defaults: chunk/action steps 50, max state/action 32, state/action mean/std, vision frozen, state projection trainable. Preserve optimizer, schedule, trainable groups, image resize and task text in comparisons.

The claimed high-success 50-episode policy and insertion policy have **not been uniquely identified**. Cached similarly named checkpoints include `openarm_pringles_real_v00` (a different training dataset ID) and `openarm_plugbattery...` (different task name); neither is evidence for the requested runs. Exact training SHA, base revision, seeds/budgets/checkpoint selection and evaluation configuration for those runs remain blockers.

## Quality and coverage

Exact SHA256 comparison of state+action arrays maps real_10 episode i to real_50 episode i for i=0…9. This is a prefix subset, not an independent collection. It cannot be used as an independent held-out test set against that pool.

The unchanged handover adapter finds no severe flags in real_10 (one timing outlier and one action spike). It flags 14/50 pool episodes for review: **16,28,32,35,36,39,40,41,42,43,44,45,46,47**. Both-hold command overlap is 2–8 frames (median 3) in real_10, −2–12 (median 2) in the pool. These command/state proxies require video review; do not automatically discard them or infer physical grasp success. Handover-specific rules were deliberately not applied to battery insertion; insertion phase annotation is outstanding.

Inherited upstream annotations mark 50 starts as human verified but omit annotator identity. Image coordinate bounds: real_10 u=292.4–404.4, v=121.7–352.4; pool u=247.6–462.1, v=29.3–410.6 px. Upstream 48 px kernel effective counts are 3.01 and 4.53. Reference-point consistency was not revalidated; these are image diversity descriptors, not calibrated workspace coverage. No measured workspace boundary or matched evaluation start distribution exists. Robot start-joint variation is small (per-channel std in audit JSON); it does not establish object coverage.
