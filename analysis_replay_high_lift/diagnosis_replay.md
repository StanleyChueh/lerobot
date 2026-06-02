# Replay lift action diagnostics

## Configuration

```
high_root = debug_runs/20260529_112158_high
policy_path = ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2
dataset_repo_id = ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2
intervention_name = high_transport
alpha = 6.0
reach_chunk = 1
lift_chunk = 2
repeat_exact = 100
repeat_per_observation = 10
```

## Test 1: exact same observation repeated

If the same saved reach observation produces different actions across repeats, the policy/intervention path is nondeterministic or stateful.

Top varying action dimensions:
```
     action_key       mean  std        min        max  range  max_abs_deviation_from_mean
action_values.0  -8.163359  0.0  -8.163359  -8.163359    0.0                          0.0
action_values.1  31.876858  0.0  31.876858  31.876858    0.0                          0.0
action_values.2 -22.726837  0.0 -22.726837 -22.726837    0.0                          0.0
action_values.3 -86.645744  0.0 -86.645744 -86.645744    0.0                          0.0
action_values.4  -5.220163  0.0  -5.220163  -5.220163    0.0                          0.0
action_values.5  32.337440  0.0  32.337440  32.337440    0.0                          0.0
```

- Test 1 max action std: `0.00000000`
- Test 1 max action range: `0.00000000`
- Interpretation: exact same observation gives numerically identical action. Policy replay is deterministic.

## High EEF mode split

```
 episode_idx      eef_mode  reach_eef_z  lift_eef_z  delta_eef_z_lift_minus_reach
           0  low_eef_mode     0.009717    0.090976                      0.081260
           1 high_eef_mode     0.007196    0.104742                      0.097546
           2 high_eef_mode     0.006140    0.104576                      0.098436
           3  low_eef_mode     0.005951    0.071588                      0.065637
           4  low_eef_mode     0.008331    0.078873                      0.070543
           5 high_eef_mode     0.006372    0.110050                      0.103679
           6 high_eef_mode     0.007822    0.109366                      0.101543
           7  low_eef_mode     0.008607    0.102518                      0.093911
           8  low_eef_mode     0.006197    0.075996                      0.069799
           9 high_eef_mode     0.006700    0.108277                      0.101577
```

## Test 2: high-mode vs low-mode replay from chunk-1 observations

This compares predicted actions from the reach/about-to-grasp observation inside high intervention only.

Most separating action dimensions:
```
     action_key  high_mode_mean  low_mode_mean  high_minus_low  abs_high_minus_low  effect_z  high_mode_std  low_mode_std
action_values.3      -86.681825     -86.786336        0.104510            0.104510  2.639730       0.031121      0.046545
action_values.2      -22.132223     -22.615932        0.483709            0.483709  0.457103       0.738707      1.301502
action_values.1       32.493340      32.763788       -0.270448            0.270448  0.451539       0.621696      0.575297
action_values.5       34.378284      35.703121       -1.324836            1.324836  0.433480       1.761434      3.947031
action_values.0       -8.783778      -8.586438       -0.197340            0.197340  0.423031       0.341883      0.564217
action_values.4       -6.244497      -6.192398       -0.052099            0.052099  0.106840       0.523470      0.448956
```

## Test 3: chunk-1 predicted action vs actual chunk-2 EEF height

Top action dimensions correlated with actual lift height:
```
                      target      action_key  pearson_corr  abs_corr
delta_eef_z_lift_minus_reach action_values.3      0.604202  0.604202
delta_eef_z_lift_minus_reach action_values.1     -0.307898  0.307898
delta_eef_z_lift_minus_reach action_values.4     -0.174519  0.174519
delta_eef_z_lift_minus_reach action_values.0     -0.154778  0.154778
delta_eef_z_lift_minus_reach action_values.5     -0.105707  0.105707
delta_eef_z_lift_minus_reach action_values.2     -0.103666  0.103666
                  lift_eef_z action_values.3      0.553233  0.553233
                  lift_eef_z action_values.1     -0.339573  0.339573
                  lift_eef_z action_values.2     -0.179899  0.179899
                  lift_eef_z action_values.4     -0.142304  0.142304
                  lift_eef_z action_values.0     -0.090743  0.090743
                  lift_eef_z action_values.5     -0.075634  0.075634
```

## Practical interpretation

Use these rules:

1. If Test 1 has near-zero variance, exact same input is stable.
2. If Test 2 separates high-EEF and low-EEF modes, the decisive difference is already present in the chunk-1 observation/state.
3. If Test 3 shows strong correlation between one predicted action dimension and chunk-2 EEF height, that dimension is a likely lift-amplitude control channel.
4. If Test 1 is stable but Test 2 differs, the issue is input sensitivity, not random policy output.
5. If Test 2 does not differ but Test 3 lift heights differ, the issue is likely robot execution/contact/grasp dynamics after the action is issued.

Focus action dimensions plotted: `[1, 2, 5]`
