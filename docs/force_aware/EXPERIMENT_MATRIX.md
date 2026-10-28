# Research evaluation matrix

Primary: real battery insertion. Secondary: real ten-episode Pringles handover. This is a preregistered proposal; no training or execution has been run.

## Gates and arms

| Gate / arm | Observation and action setup | Comparison / decision |
|---|---|---|
| G0 baseline parity | A, original16 state, pinned processors and runtime logs | Resolve schema, checkpoint and evaluation identity before interpreting success |
| G1 low-data | Original ten, three random raw tens, three reviewed-clean tens, one coverage-selected clean ten, full50 | Separate sample count, selection and demo quality; all train at batch8 × 20k updates × three seeds |
| V vision-only ablation | Images+language; fixed neutral state input, train/eval consistently | Quantifies state contribution; distinct from A (vision+proprioception) |
| A baseline | Images+language+state16 | Main reference for B/C; identical action interface and controller |
| B raw telemetry | A+validated velocity/effort with validity and age | Test useful contact information without force-regression claims |
| C learned contact | A+frozen supervised causal contact latent; force only with verified Newton labels | Conditional on independent labelled data, split-safe encoder and calibration |
| D force-aware control | B or C plus separately validated feedback controller | Factorial controller off/on under identical sensing; owned by hardware team, not implemented here |
| Controls | B with zero/shuffled/delayed telemetry; capacity-matched unused/random latent | Distinguish real signal benefit from capacity, timing and dataset leakage |

Training seeds: 1000/2000/3000. Match base checkpoint revision, optimizer, LR schedule, sample count, precision, augmentation, decoder, camera assignment, task text, action horizon and trainable parameter groups. Keep total steps fixed and record frame/phase sampling. Where B/C add parameters report count and a matched-capacity A control. Compare controller arms at the same policy refresh rate; study a faster contact loop separately to expose latency as a mechanism. Select checkpoint at the fixed final step or an identical validation-only rule, never best physical test outcome.

Do not impose handover phase heuristics on insertion. Annotate insertion: approach, alignment, first contact, search/reorientation, seating, release, withdrawal, failure/recovery. Freeze phase definitions and annotate a blinded subset twice to quantify agreement. Split by collection session, object/holder and trajectory before windows, augmentation, encoder learning or normalization. All windows from an episode stay together. Screen duplicate trajectories across splits. Default full-pool metadata statistics must be replaced by training-only statistics in the eventual experiment configuration.

## Matched evaluation

Freeze a separate physical start manifest: object/holder pose and orientation, robot start joints, grasp offset, clearance/misalignment, region, lighting, randomization ID and reset tolerance. Hardware colleagues later perform execution; this repository only defines manifests and analyzes returned logs. Randomize paired arm order within start blocks. Use 30 matched starts per training seed for a pilot (90 outcomes/arm), with identical stop/time budgets; increase sample size using observed paired discordance and seed variance before making a powered claim. Starts may recur across seeds; account for that dependence.

Report seed-wise outcomes, mean and range, Wilson intervals for descriptive success, and paired risk differences with hierarchical bootstrap over training seed and start block. Three seeds are a pilot, not a reliable estimate of every source of variance. Predefine a meaningful success gain of 10 percentage points, with no increase in safety-event rate. A wide interval is inconclusive, not evidence of no effect. Keep a fresh confirmatory start set if experiment selection uses pilot outcomes.

| Metric | Operational definition |
|---|---|
| Task success | Independent binary completion within fixed timeout, no intervention; keep all attempted trials in denominator |
| Insertion completion | Verified seating depth/tolerance, electrical continuity where relevant, release and retention; define sensor/visual adjudication before trials |
| Contact detection | Event precision/recall/F1, AUPRC, false alarms/min free motion, onset delay and calibration against independent labels; threshold selected on validation only |
| Force estimation | Per-axis MAE/RMSE/bias in N, magnitude/direction and peak error, stratified by phase/load; only where calibrated labelled force exists, with frame/sign/reference point specified |
| Timing | Acquisition-to-feature and observation-to-action p50/p95/p99, estimator runtime, packet age/drop rate, missed deadlines and policy refresh frequency |
| Safety | Overcurrent/overtemperature, protective stop, collision, object damage/drop and operator intervention per attempt; peak contact loads only when measured |
| Learning | Validation action error, phase-weighted errors, sample reuse and run variance; offline loss is not physical success |

Define event matching windows and multi-contact treatment before contact scoring. Hardware owners specify absolute safe limits and stopping rules; offline effort cannot certify safety or substitute for measured end-effector force.

## Smallest falsification experiment

First, on independently labelled insertion/free-motion recordings, compare a simple effort/velocity contact detector to a state/velocity-only detector on held-out sessions with identical causal timing. If telemetry does not improve contact discrimination at matched false-alarm rate or arrives after useful correction time, the proposed contact pathway lacks its required signal; do not spend on VLA integration yet. This tests a necessary condition, not the whole hypothesis.

The smallest direct policy pilot uses A versus B on the same ten reviewed insertion demonstrations, three training seeds, the same controller and 30 paired starts/seed. Add shuffled-telemetry B as a diagnostic if B wins. Predeclare the 10-point effect target. If the upper confidence bound excludes that gain after sufficient power, reject the scoped hypothesis that validated actuator feedback materially helps this setup. If B improves contact scores but not insertion success, investigate correction authority and refresh latency; do not declare all force feedback ineffective. C and D are not prerequisites for this falsification pilot.

## Novelty assessment

[ForceVLA](https://arxiv.org/abs/2505.22159) already fuses measured force with visual-language features through a mixture of experts. [FM-VLA](https://arxiv.org/abs/2607.18231) already encodes force histories as compact memory for contact-dependent tasks. [NeuralActuator](https://arxiv.org/abs/2607.11734v2) already links actuator telemetry to learned force perception and behavior cloning. These precedents rule out a novelty claim based on appending force to SmolVLA.

A defensible research question is whether causally aligned, calibrated actuator-derived contact representations improve low-data insertion across sessions and contact regimes, after controlling for coverage, demonstration quality, model capacity and control rate. Evidence would need estimator generalization, uncertainty/failure analysis, causal signal ablations and matched policy evaluation. A broader systematic literature review is still needed before any novelty claim.

## GO / NO-GO

**GO:** offline source/data inspection, reviewing existing real demonstrations, freezing subsets/start manifests, synthetic adapter tests.

**NO-GO now:** causal attribution of the reported success gap; force-estimation accuracy claims; learned-force training without labels; production telemetry integration without the final schema; physical controller experiments without hardware-team validation. GPU training, deployment, robot connections and checkpoint uploads remain outside this work.
