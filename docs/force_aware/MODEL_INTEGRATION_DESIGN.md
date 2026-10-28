# Model integration design

**GO: isolated synthetic adapter and CPU checks. NO-GO: production integration until Claude Code supplies the finalized `lerobot_openarm/docs/force_aware/TELEMETRY_SCHEMA.md`.** That path is absent from the inspected OpenArm archive. No substitute document with that name is created here.

## NeuralActuator relevance

[NeuralActuator v2](https://arxiv.org/html/2607.11734v2), sections III-C/D, uses commanded/measured positions, velocity, tracking error and actuator telemetry (current, voltage, temperature; channels vary by platform). Nine frames comprise eight historical samples plus the current sample. Force/contact heads require synchronized external-force labels; a torque surrogate learns through differentiable pose rollouts. Sensor labels are transformed to the base frame; known-weight labels are only quasi-static references. The gate separates zero/nonzero labelled force; its 0.01 N training threshold is not an OpenArm safety threshold. Reported effort and torque-surrogate outputs are not ground-truth Cartesian force. OpenArm needs platform-specific data and calibration; SO-101/OMX weights cannot be assumed transferable.

The [reference inference code](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/main/NeuralActuator/infer_actuator.py) optionally standardizes with checkpoint statistics and clips to ±10. `_build_force_only` returns a causal step using history and current telemetry, emitting surrogate effort, force, gate and condition outputs before updating history. Save its feature order, statistics, sample period and weights together. Recorded-current rollouts do not predict currents for counterfactual actions. No reference model was run here.

The source URL above is mutable; its inspected file SHA256 is recorded in the validation notes. Paper facts are pinned to v2. Exact deployment package/version must be pinned before reuse.

## Existing SmolVLA path

`datasets/factory.py` resolves episode selection and temporal action indices, then supplies `dataset.meta.stats`; `dataset_metadata.py` loads published stats. `policies/factory.py` maps dataset features and constructs the policy/processor. `processor_smolvla.py` builds rename, normalization, task/token and device steps. Inspect actual serialized processor order rather than relying on prose order. `modeling_smolvla.py::prepare_state` takes the current state and pads to `max_state_dim`; `state_proj` consumes that width as a prefix token. Observation history defaults to one frame; action chunk and executed steps default to 50. A telemetry history needs its own causal encoder or explicit representation; adding observation columns alone does not make the network consume them.

Policy weights/config are serialized through `PreTrainedPolicy`; processor JSON and safetensor statistics are separate artifacts. A force-enabled checkpoint must record schema version/hash, joint names/units, calibration, clock transform, sample/window rate, missingness policy, feature layout, train split digest/statistics, encoder revision and action dimensionality. Validate all at load time. Do not infer compatibility from tensor width alone.

| Option | Features / model change | Data gate | Compatibility |
|---|---|---|---|
| A | Existing images+language+16D state; unchanged action16 and pad32 | Existing real demos | Preserve baseline config/processors/weights and commands |
| B | A plus validated joint velocity16 and actuator effort16 | Verified telemetry units, signs, calibration and causal alignment | Raw state is 48D before validity/age; exceeds default32. Explicit new projection/config or a separate telemetry encoder is required |
| C | A plus frozen learned contact representation, optionally calibrated force and uncertainty | Independently verified synchronized contact labels; Newton regression only with measured force/frame labels | New versioned encoder/token adapter; cannot load legacy weights into a widened projection implicitly |

For B, prefer a separate telemetry encoder/token path with masks and ages so the 16D state normalizer and action head remain interpretable. Initialize added conditioning with zero influence and verify A-equivalence before fine-tuning. A simple 48D concatenation is a useful capacity-controlled ablation, not the production default. If widening the state projection, explicitly migrate its first 32 columns, initialize extras, and create a new checkpoint type/version; never silently reshape old weights. A 32D “state+torque” shortcut drops velocity and differs from the requested B.

For C, start with contact classification, calibration and a compact causal latent. Train the encoder only on training sessions, freeze it for the first VLA experiment; compare against equally sized random/frozen and raw-feature adapters. Include free motion, static loads, acceleration, thermal drift, different grasp stiffness and contact directions. Use validity masks and session/object/start holdouts. Labels derived from the same effort threshold used as input are circular validation. Quasi-static weights do not validate insertion force. No valid labels means **C remains a design only**.

## Provisional adapter boundary

[provisional.py](../../tools/force_aware/provisional.py) is outside `src/lerobot`, unregistered and unreachable from training. Its contract accepts exactly 16 unique named joints and explicit per-channel position/velocity/effort units; it rejects unit mismatches instead of guessing conversions. The synthetic fixture's all-radian grippers and N·m effort are intentionally invented and **must not become OpenArm defaults**.

A packet carries a globally qualified episode ID and acquisition/availability times in one provisional clock. Alignment selects the newest sample acquired AND available by observation time. Actions have a separate target timestamp and bounded delay. No future interpolation or cross-episode carry is permitted; duplicate acquisition timestamps fail. Stale/missing samples produce zeros plus invalid status and age; normalization refuses invalid rows. The normalizer fits exactly named training episodes and checks disjoint test IDs. Callers must exclude invalid rows and also hold out validation sessions; episode-ID checking alone cannot detect duplicated trajectories under different IDs.

The final schema must settle hardware joint IDs/gripper semantics, raw versus calibrated effort, saturation flags, timestamps/clock drift, arrival latency, sequence numbers, per-channel freshness, calibration revisions and optional label frames. Then replace the fixture decoder behind the same conceptual validate→align→normalize→feature interface. Require recorded schema fixtures and cross-repository contract tests before connecting training. A production missingness strategy (masking, dropping windows or explicit fallback) must be trained and validated, not inferred from this zero payload.

## Test scope

Six synthetic CPU tests cover joint permutation/unit rejection, action/observation/telemetry causality, delayed availability, stale/missing and duplicate samples, episode isolation, train-only normalization, feature-width rejection, unchanged baseline arrays, actual SmolVLA state padding and strict legacy projection state-dict restoration. Existing SmolVLA processor tests also run. Full VLM checkpoint loading, GPU-only skipped cases, widened checkpoint migration, production telemetry decoding and physical force accuracy are **not verified**. There is no synthetic force-accuracy claim.
