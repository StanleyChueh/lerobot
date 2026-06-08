# Sequence-aware rest-to-grasp replay diagnosis

## What this test does

This experiment resets the policy once at the start of each replay, feeds a saved rest-to-grasp observation sequence, does not reset between intermediate observations, and records the predicted action at the configured decision step.

This is stronger than a single-snapshot determinism test because it includes policy/preprocessor/action-queue state evolution across the observation history.

## Configuration

```
high = debug_runs/20260603_164150_HIGH_current
policy_path = ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2
dataset_repo_id = ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2
intervention_name = high_transport
alpha = 6.0
sequence_chunks = 0,1
decision_index = -1
repeat_exact_sequence = 10
repeat_per_episode_sequence = 1
```

Exact sequence:
```
 episode_idx  seq_pos  chunk_idx                                                                                         source_path
           8        0          0 debug_runs/20260603_164150_HIGH_current/episode_000008/debug_chunk_rawid_0_000_observation_frame.pt
           8        1          1 debug_runs/20260603_164150_HIGH_current/episode_000008/debug_chunk_rawid_1_001_observation_frame.pt
```

## Exact same sequence repeated

Decision-step action repeat summary:
```
     action_key       mean  std        min        max  range  max_abs_deviation_from_mean
action_values.0  -9.178151  0.0  -9.178151  -9.178151    0.0                          0.0
action_values.1  83.822914  0.0  83.822914  83.822914    0.0                          0.0
action_values.2   0.473452  0.0   0.473452   0.473452    0.0                          0.0
action_values.3 -89.774529  0.0 -89.774529 -89.774529    0.0                          0.0
action_values.4  -4.776394  0.0  -4.776394  -4.776394    0.0                          0.0
action_values.5  59.766430  0.0  59.766430  59.766430    0.0                          0.0
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
- Mean max FK EEF-z over predicted action chunk: `0.26914818`
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

No all-episode sequence results were produced.

## Professor-facing interpretation

If the exact-sequence max range is near zero, the supported claim is:

> For the same saved rest-to-grasp observation history and the same steering configuration, the policy produced the same lift-decision joint action and the same FK-implied commanded EEF height under this offline replay setup.

If normal real-world rollouts still branch into different lift heights, the likely cause is not random inference from the same history, but differences in the actual pre-lift observation/state/contact/grasp history or dataset-induced multimodality.
