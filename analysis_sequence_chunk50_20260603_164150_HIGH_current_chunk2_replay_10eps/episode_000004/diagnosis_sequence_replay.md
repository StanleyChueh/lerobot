# Sequence-aware rest-to-grasp replay diagnosis

## What this test does

This experiment resets the policy once at the start of each replay, feeds a saved rest-to-grasp observation sequence, does not reset between intermediate observations, and records the predicted action at the configured decision step.

This is stronger than a single-snapshot determinism test because it includes policy/preprocessor/action-queue state evolution across the observation history.

## Configuration

```
high = /tmp/chunk2_replay_high_episode_000004
policy_path = ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2
dataset_repo_id = ethanCSL/eval_high_intervention_rollout_debug
intervention_name = high_transport
alpha = 6.0
sequence_chunks = 2
decision_index = -1
repeat_exact_sequence = 10
repeat_per_episode_sequence = 10
```

Exact sequence:
```
 episode_idx  seq_pos  chunk_idx                                                                                        source_path
           4        0          2 /tmp/chunk2_replay_high_episode_000004/episode_000004/debug_chunk_rawid_2_002_observation_frame.pt
```

## Exact same sequence repeated

Decision-step action repeat summary:
```
     action_key       mean  std        min        max  range  max_abs_deviation_from_mean
action_values.0  -8.171820  0.0  -8.171820  -8.171820    0.0                          0.0
action_values.1  69.608879  0.0  69.608879  69.608879    0.0                          0.0
action_values.2 -53.914471  0.0 -53.914471 -53.914471    0.0                          0.0
action_values.3 -87.774353  0.0 -87.774353 -87.774353    0.0                          0.0
action_values.4  -3.520399  0.0  -3.520399  -3.520399    0.0                          0.0
action_values.5  14.656940  0.0  14.656940  14.656940    0.0                          0.0
```

- Max decision-step action std: `0.00000000`
- Max decision-step action range: `0.00000000`
- FK(predicted action) EEF-z std: `0.00000000`
- FK(predicted action) EEF-z range: `0.00000000`
- Interpretation: same saved observation sequence produced numerically identical decision-step joint actions and identical FK-implied commanded EEF height.

FK note:
- `predicted_action_eef_z` is the commanded EEF height implied by the predicted joint target using forward kinematics.
- It is not the measured physical robot EEF height after motor dynamics, grasp contact, or object interaction.

Saved plots:
- `plots/sequence_exact_observation_montage.png`
- `plots/sequence_exact_decision_action_repeat_plot.png`
- `plots/sequence_exact_decision_all_action_dims_repeat_plot.png`
- `plots/sequence_exact_decision_predicted_fk_eef_z_repeat_plot.png`
- `plots/sequence_step_action_values_trial0.png`
- `plots/sequence_step_predicted_fk_eef_z_trial0.png`

## Predicted action chunk repeatability

- Drained predicted/queued action chunk steps: `50`
- Mean max FK EEF-z over predicted action chunk: `0.29700722`
- Range of max FK EEF-z over predicted action chunk across repeats: `0.00000000`
- Interpretation: the max FK-implied EEF height inside the predicted action chunk is identical across repeated replays of the same observation sequence.

Saved predicted-chunk files:
- `sequence_exact_predicted_action_chunk_trials.csv`
- `sequence_exact_predicted_action_chunk_per_trial_summary.csv`
- `sequence_exact_predicted_action_chunk_repeat_consistency.csv`
- `sequence_exact_predicted_action_chunk_max_fk_eef_z_summary.csv`

Saved predicted-chunk plots:
- `plots/sequence_exact_predicted_action_chunk_focus_dims_trial0.png`
- `plots/sequence_exact_predicted_action_chunk_fk_eef_z_trial0.png`
- `plots/sequence_exact_predicted_action_chunk_max_fk_eef_z_repeat_plot.png`
- `plots/sequence_exact_predicted_action_chunk_repeat_range_heatmap.png`

Important limitation:
- The chunk is obtained by feeding the sequence up to the decision observation, then repeatedly calling `predict_action()` with the decision observation to drain the policy/action queue.
- This is a good offline proxy for an action-chunking policy, but the most definitive physical evidence still requires real-robot fixed-action replay.

## Diverse sequence replay across episodes

Per-episode repeated-sequence stability:
```
 episode_idx  decision_chunk_idx  num_repeats  predicted_action_eef_z  predicted_action_eef_z_repeat_std  predicted_action_eef_z_repeat_range  lift_eef_z  action_values.0_repeat_std  action_values.1_repeat_std  action_values.2_repeat_std  action_values.3_repeat_std  action_values.4_repeat_std  action_values.5_repeat_std  predicted_action_eef_z_repeat_std
           4                   2           10                0.294166                       5.551115e-17                                  0.0    0.099656                         0.0                         0.0                         0.0                         0.0                         0.0                         0.0                       5.551115e-17
```

- Max within-episode sequence repeat std: `0.00000000`

## Professor-facing interpretation

If the exact-sequence max range is near zero, the supported claim is:

> For the same saved rest-to-grasp observation history and the same steering configuration, the policy produced the same lift-decision joint action and the same FK-implied commanded EEF height under this offline replay setup.

If normal real-world rollouts still branch into different lift heights, the likely cause is not random inference from the same history, but differences in the actual pre-lift observation/state/contact/grasp history or dataset-induced multimodality.
