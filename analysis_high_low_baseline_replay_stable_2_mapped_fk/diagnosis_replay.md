# Replay lift action diagnostics

## Configuration

```
high_root = debug_runs/20260605_155251_Stable_2
policy_path = ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2
dataset_repo_id = ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2
intervention_name = high_transport
alpha = 6.0
reach_chunk = 1
lift_chunk = 2
repeat_exact = 10
repeat_per_observation = 10
```

## Test 1: exact same observation repeated

If the same saved reach observation produces different actions across repeats, the policy/intervention path is nondeterministic or stateful.

Top varying action dimensions:
```
     action_key       mean  std        min        max  range  max_abs_deviation_from_mean
action_values.0 -13.494370  0.0 -13.494370 -13.494370    0.0                          0.0
action_values.1  15.654602  0.0  15.654602  15.654602    0.0                          0.0
action_values.2 -53.632915  0.0 -53.632915 -53.632915    0.0                          0.0
action_values.3 -86.909424  0.0 -86.909424 -86.909424    0.0                          0.0
action_values.4 -12.372672  0.0 -12.372672 -12.372672    0.0                          0.0
action_values.5  29.067268  0.0  29.067268  29.067268    0.0                          0.0
```

- Test 1 max action std: `0.00000000`
- Test 1 max action range: `0.00000000`
- Interpretation: exact same observation gives numerically identical action. Policy replay is deterministic.

Saved Test 1 visualization files:
- `plots/test1_exact_same_observation_action_repeat_plot.png`
- `plots/test1_exact_observation_montage.png`

How to read Test 1:
- `plots/test1_exact_observation_montage.png` shows the exact RGB observation used for repeated inference.
- `plots/test1_exact_same_observation_action_repeat_plot.png` shows predicted action values across repeated inference calls.
- If the montage is fixed and the repeated-action plot shows flat horizontal lines, the same saved observation is producing the same predicted action across repeats.
- This means the replay path is deterministic for the same input under the same intervention setup.

## High EEF mode split

```
 episode_idx      eef_mode  reach_eef_z  lift_eef_z  delta_eef_z_lift_minus_reach
           0  low_eef_mode     0.029141    0.010312                     -0.018829
           1  low_eef_mode     0.030337    0.007449                     -0.022888
           2 high_eef_mode     0.030275    0.025043                     -0.005231
           3  low_eef_mode     0.030779    0.007852                     -0.022927
           4 high_eef_mode     0.030128    0.010595                     -0.019533
           5 high_eef_mode     0.028273    0.016131                     -0.012142
           6  low_eef_mode     0.030145    0.009852                     -0.020293
           7  low_eef_mode     0.029720    0.010484                     -0.019237
           8 high_eef_mode     0.029067    0.050013                      0.020946
           9 high_eef_mode     0.029228    0.029202                     -0.000026
```

## Test 2: high-mode vs low-mode replay from chunk-1 observations

This compares predicted actions from the reach/about-to-grasp observation inside high intervention only.

Most separating action dimensions:
```
     action_key  high_mode_mean  low_mode_mean  high_minus_low  abs_high_minus_low  effect_z  high_mode_std  low_mode_std
action_values.1       15.759100      16.816592       -1.057493            1.057493  2.086515       0.498564      0.514948
action_values.0      -13.445453     -13.786065        0.340612            0.340612  1.466511       0.272886      0.182818
action_values.5       34.237387      30.073423        4.163964            4.163964  1.186509       3.753680      3.246848
action_values.3      -86.947629     -86.873654       -0.073975            0.073975  0.824847       0.085802      0.093403
action_values.2      -52.505586     -52.115595       -0.389991            0.389991  0.527126       0.937229      0.465121
action_values.4      -12.442408     -12.494194        0.051786            0.051786  0.191586       0.198565      0.326650
```

## Test 3: chunk-1 predicted action vs actual chunk-2 EEF height

Top action dimensions correlated with actual lift height:
```
                      target      action_key  pearson_corr  abs_corr
delta_eef_z_lift_minus_reach action_values.1     -0.812882  0.812882
delta_eef_z_lift_minus_reach action_values.3     -0.602997  0.602997
delta_eef_z_lift_minus_reach action_values.2     -0.310002  0.310002
delta_eef_z_lift_minus_reach action_values.0      0.229256  0.229256
delta_eef_z_lift_minus_reach action_values.4     -0.135186  0.135186
delta_eef_z_lift_minus_reach action_values.5      0.116291  0.116291
                  lift_eef_z action_values.1     -0.804352  0.804352
                  lift_eef_z action_values.3     -0.589199  0.589199
                  lift_eef_z action_values.2     -0.347270  0.347270
                  lift_eef_z action_values.0      0.238098  0.238098
                  lift_eef_z action_values.4     -0.128254  0.128254
                  lift_eef_z action_values.5      0.087287  0.087287
```

## Practical interpretation

Use these rules:

1. If Test 1 has near-zero variance, exact same input is stable.
2. If Test 2 separates high-EEF and low-EEF modes, the decisive difference is already present in the chunk-1 observation/state.
3. If Test 3 shows strong correlation between one predicted action dimension and chunk-2 EEF height, that dimension is a likely lift-amplitude control channel.
4. If Test 1 is stable but Test 2 differs, the issue is input sensitivity, not random policy output.
5. If Test 2 does not differ but Test 3 lift heights differ, the issue is likely robot execution/contact/grasp dynamics after the action is issued.

Focus action dimensions plotted: `[1, 2, 5]`
