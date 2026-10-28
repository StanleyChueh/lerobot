# Low-data diagnosis

**The reported success gap has no identified causal explanation yet.** Coverage is a plausible contributor; absence of force sensing is not established as the cause. No training or robot evaluation was performed.

The brief's “around 10%” real_10 success conflicts with the [earlier OpenArm report](https://github.com/StanleyChueh/lerobot_openarm/blob/9abbe126128ce17f66c98c726688dac70b84a966/research/dataset_diagnostics/RESEARCH_READINESS_REPORT.md), which reports real_10 2/10 and sim_10 1/10. Those are distinct datasets and evaluations. Do not substitute the sim number. Raw trial logs, checkpoint hashes, denominators, starts and success definitions are needed to resolve this discrepancy. Even 1/10 has a roughly 2–40% Wilson 95% interval.

| Explanation | Evidence | Current assessment / discriminating test |
|---|---|---|
| Limited coverage | real_10 is the first ten demos; inherited image starts span a smaller range than the pool | Plausible; compare several frozen random and coverage-selected subsets at matched budgets and starts |
| Poor demos | Original ten have no severe handover flags; later pool contains 14 review flags | Original-ten quality alone does not explain gap; compare raw versus reviewed-clean subsets |
| Normalization/schema defect | Full-data stats and real_10 normalizer match; actual state 16→32 | No silent 6D truncation found; strict serving compatibility still needs full checkpoint test |
| Training confound | Only real_10 run uniquely verified; prior real_20 used batch 16 versus 8 at the same 20k steps | Cannot compare to the asserted 50-policy run without its exact artifacts |
| Execution mismatch | Same recorded schema, two-frame action lead proxy; no matching runtime logs | Unresolved: camera assignment, resize, normalizer, joint order/units, chunk refresh, stale images, reset and command filters |
| Missing contact feedback | No telemetry or contact labels in these datasets | Untested hypothesis, especially relevant to insertion; test after baseline parity |

## Frozen ten-episode comparisons

The following draws were generated with NumPy `default_rng(seed).choice(50,10,replace=False)`, before policy outcomes. They are comparisons of data composition; overlapping draws are not independent datasets.

| Set | Episode IDs in real_50 | Frames | Review flags | Median overlap frames | Image kernel effective n (48 px) |
|---|---|---:|---:|---:|---:|
| Original | 0–9 | 2575 | 0 | 3 | 3.01 |
| Seed 17 | 1,4,7,10,17,20,25,30,35,46 | 2467 | 2 | 2 | 3.79 |
| Seed 29 | 2,6,10,12,22,23,25,28,30,38 | 2507 | 1 | 2.5 | 5.57 |
| Seed 43 | 0,1,12,17,20,22,25,27,29,40 | 2541 | 1 | 2 | 5.32 |

A subset can have a higher kernel effective n than the pool because the statistic measures redundancy as well as count. It is neither a success predictor nor a metric workspace score. Equal episode count does not imply equal frame count or phase exposure.

Before training, separately freeze a video-reviewed clean pool; draw three random clean tens with the same algorithm and publish the changed IDs. Freeze one coverage-selected clean ten using calibrated object starts or consistently reviewed image points, without looking at evaluation outcomes. Compare to random clean tens to distinguish selection benefit from cleaning benefit. Retain the raw comparisons above as a secondary quality ablation.

Use training seeds 1000/2000/3000, identical pinned base weights, batch 8, 20000 updates (160000 sampled examples per run), optimizer/schedule, camera map, task text, augmentation and checkpoint-selection rule. Report per-episode and per-phase exposure; equal updates imply more reuse in smaller datasets. Use a secondary matched-epoch budget only as a separately labelled experiment. Fit all statistics on the selected training episodes. The default loader reads full `meta/stats.json`; selecting episodes or eval_split alone must not be assumed to recompute split-safe statistics.

Evaluation starts must be external to training collections, frozen and balanced by workspace region and insertion alignment. Use the same reset poses, object placement, lighting and randomized paired trial order per seed. Never reserve real_10 as the test set for real_50. If using held-out offline episodes, regenerate subset IDs from the remaining pool; do not silently reuse the frozen draws that intersect the holdout.

Required next evidence: exact 50-demo and insertion checkpoint/config/processor revisions; per-trial success logs; runtime observation/command timestamps and camera identities; reviewed insertion phases and recoveries; calibrated starts/evaluation regions; finalized telemetry contract and actual synchronized recordings. Until then: **NO-GO for claiming that data scarcity, execution mismatch, or force absence caused the observed success rate.**
