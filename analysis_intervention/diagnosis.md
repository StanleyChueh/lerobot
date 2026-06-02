# Intervention Stability Diagnosis

## Scope

This report compares baseline, high intervention, and low intervention debug rollouts.
It localizes unstable chunks and action dimensions. It does not prove physical causality unless action dimensions are mapped to robot joints or EEF-Z.

## Data loaded

- Total chunks: 151
- Conditions: baseline, high, low

| condition   |   count |   mean |   min |   max |
|:------------|--------:|-------:|------:|------:|
| baseline    |      10 |    5.1 |     5 |     6 |
| high        |      10 |    5   |     4 |     6 |
| low         |      10 |    5   |     5 |     5 |

## Focus / lift chunks

- Focus chunks used for episode-level score: `[0, 3, 2]`
- Height candidate dimensions: `['action_values.2', 'action_values.4', 'action_values.1', 'action_values.0', 'action_values.5']`

## Episode-level high-like / low-like behavior

| condition   |   num_episodes |   mean_score |   std_score |   min_score |   max_score |   range_score |   num_high_like |   num_low_like |   num_surprising |
|:------------|---------------:|-------------:|------------:|------------:|------------:|--------------:|----------------:|---------------:|-----------------:|
| baseline    |             10 |    -1.58778  |   0.413875  |    -1.85802 |   -0.371622 |      1.4864   |               1 |              9 |                0 |
| high        |             10 |    -0.779927 |   0.371522  |    -1.2888  |   -0.313291 |      0.975506 |               9 |              1 |                0 |
| low         |             10 |    -1.86482  |   0.0615385 |    -1.97325 |   -1.78001  |      0.193238 |               0 |             10 |                0 |

No high/low episodes were classified as clearly opposite-direction under the high-minus-low action prototype score.

## Strongest high-vs-low action dimensions

|   save_idx |   action_dim | action_key      |   baseline_mean |   high_mean |   low_mean |   high_low_delta |   high_low_effect_z |   high_std |   low_std |
|-----------:|-------------:|:----------------|----------------:|------------:|-----------:|-----------------:|--------------------:|-----------:|----------:|
|          0 |            2 | action_values.2 |         9.68016 |    -1.84235 |   13.8922  |        -15.7345  |            15.8656  |   0.846723 |  1.1181   |
|          3 |            4 | action_values.4 |        12.4791  |     8.08301 |   15.4512  |         -7.36822 |            10.1539  |   0.821692 |  0.614792 |
|          0 |            1 | action_values.1 |       100.76    |    92.2892  |   99.4942  |         -7.20506 |             6.98091 |   0.47879  |  1.37886  |
|          3 |            0 | action_values.0 |        13.1663  |    10.2067  |   13.6157  |         -3.40899 |             6.75552 |   0.568953 |  0.43079  |
|          2 |            2 | action_values.2 |       -45.7725  |   -53.7419  |  -42.3555  |        -11.3864  |             5.66594 |   2.17992  |  1.8235   |
|          2 |            5 | action_values.5 |         3.39223 |     4.12643 |    2.81881 |          1.30763 |             5.3903  |   0.323632 |  0.113845 |
|          3 |            1 | action_values.1 |        25.7743  |    35.4863  |   21.1577  |         14.3286  |             4.61745 |   4.2418   |  1.12523  |
|          0 |            3 | action_values.3 |       -87.4128  |   -89.1089  |  -85.8457  |         -3.26326 |             4.35603 |   0.728949 |  0.768793 |
|          2 |            4 | action_values.4 |        -5.32992 |    -2.50669 |   -6.0214  |          3.51471 |             3.28713 |   1.38481  |  0.607312 |
|          1 |            3 | action_values.3 |       -78.8184  |   -71.1885  |  -81.9267  |         10.7381  |             2.85465 |   0.490637 |  5.29708  |
|          1 |            0 | action_values.0 |       -10.3968  |    -8.91981 |  -10.5588  |          1.63903 |             2.58804 |   0.512279 |  0.73466  |
|          4 |            5 | action_values.5 |        47.5689  |    47.7072  |   49.0185  |         -1.31135 |             2.28094 |   0.777164 |  0.238902 |
|          4 |            3 | action_values.3 |       -88.7753  |   -93.2531  |  -85.75    |         -7.50302 |             2.26663 |   4.58448  |  0.947428 |
|          2 |            0 | action_values.0 |        -5.75866 |    -2.94206 |   -7.70603 |          4.76397 |             2.20172 |   2.85964  |  1.08908  |
|          4 |            2 | action_values.2 |         7.2593  |   -28.4558  |   17.2784  |        -45.7343  |             2.10723 |  30.6879   |  0.581128 |
|          1 |            2 | action_values.2 |       -18.8051  |   -14.0719  |  -20.3135  |          6.24157 |             2.08414 |   0.663261 |  4.18302  |
|          4 |            1 | action_values.1 |        93.2007  |    57.5005  |  100.529   |        -43.0287  |             2.07683 |  29.2971   |  0.435374 |
|          1 |            4 | action_values.4 |        -7.9303  |    -7.09937 |   -8.17273 |          1.07336 |             2.01715 |   0.481977 |  0.577917 |
|          4 |            0 | action_values.0 |        -4.50703 |     3.95691 |   -6.24943 |         10.2063  |             1.79363 |   8.03198  |  0.496915 |
|          3 |            3 | action_values.3 |       -92.9686  |   -90.3846  |  -95.2368  |          4.85222 |             1.7504  |   3.6748   |  1.36543  |

## Most unstable action dimensions inside each condition

| condition   |   save_idx |   action_dim | action_key      |      mean |      std |   range |   count |
|:------------|-----------:|-------------:|:----------------|----------:|---------:|--------:|--------:|
| high        |          4 |            2 | action_values.2 | -28.4558  | 30.6879  | 64.1833 |       9 |
| high        |          4 |            1 | action_values.1 |  57.5005  | 29.2971  | 60.4443 |       9 |
| baseline    |          4 |            1 | action_values.1 |  93.2007  | 23.8081  | 76.3071 |      10 |
| high        |          3 |            5 | action_values.5 |  21.1575  | 21.7076  | 43.2663 |      10 |
| baseline    |          4 |            2 | action_values.2 |   7.2593  | 20.8835  | 68.6702 |      10 |
| baseline    |          3 |            5 | action_values.5 |  35.4518  | 12.8765  | 41.9442 |      10 |
| baseline    |          3 |            1 | action_values.1 |  25.7743  | 12.0127  | 40.5162 |      10 |
| low         |          3 |            5 | action_values.5 |  25.3408  | 10.9692  | 28.6415 |      10 |
| high        |          2 |            3 | action_values.3 | -87.9059  |  8.11639 | 19.5153 |      10 |
| high        |          4 |            0 | action_values.0 |   3.95691 |  8.03198 | 17.0163 |       9 |
| high        |          2 |            1 | action_values.1 |  54.9199  |  7.73869 | 20.8165 |      10 |
| baseline    |          1 |            3 | action_values.3 | -78.8184  |  7.15227 | 20.5401 |      10 |
| baseline    |          2 |            3 | action_values.3 | -93.5472  |  7.03002 | 23.605  |      10 |
| baseline    |          1 |            2 | action_values.2 | -18.8051  |  6.61394 | 19.9239 |      10 |
| baseline    |          4 |            0 | action_values.0 |  -4.50703 |  6.61343 | 21.914  |      10 |
| baseline    |          2 |            2 | action_values.2 | -45.7725  |  6.51288 | 22.9219 |      10 |
| high        |          4 |            4 | action_values.4 |   3.14558 |  6.38702 | 14.2087 |       9 |
| baseline    |          4 |            4 | action_values.4 |  -2.83241 |  5.34118 | 17.7613 |      10 |
| low         |          1 |            3 | action_values.3 | -81.9267  |  5.29708 | 14.6563 |      10 |
| baseline    |          3 |            3 | action_values.3 | -92.9686  |  4.97865 | 17.4299 |      10 |

## Similar observation, different action evidence

Rows below are high-priority cases where observation distance is low but action distance is high.

| condition   | episode        |   save_idx | nearest_condition   | nearest_episode   |   nearest_save_idx |   obs_distance |   action_distance |   mismatch_score |
|:------------|:---------------|-----------:|:--------------------|:------------------|-------------------:|---------------:|------------------:|-----------------:|
| low         | episode_000002 |          0 | high                | episode_000003    |                  0 |        12.4172 |          0.785179 |         0.741152 |
| baseline    | episode_000000 |          0 | low                 | episode_000007    |                  0 |        17.3229 |          1.02361  |         0.667734 |
| high        | episode_000006 |          2 | high                | episode_000009    |                  2 |        12.405  |          0.44333  |         0.661057 |
| high        | episode_000009 |          2 | high                | episode_000006    |                  2 |        12.405  |          0.44333  |         0.661057 |
| baseline    | episode_000001 |          0 | high                | episode_000003    |                  0 |        16.2488 |          0.564936 |         0.607868 |
| low         | episode_000008 |          0 | baseline            | episode_000009    |                  0 |        13.4427 |          0.431746 |         0.603658 |
| baseline    | episode_000009 |          0 | baseline            | episode_000006    |                  0 |        11.6443 |          0.350943 |         0.586937 |
| baseline    | episode_000006 |          0 | baseline            | episode_000009    |                  0 |        11.6443 |          0.350943 |         0.586937 |
| high        | episode_000002 |          0 | baseline            | episode_000003    |                  0 |        18.13   |          0.57798  |         0.573703 |
| low         | episode_000001 |          0 | low                 | episode_000002    |                  0 |        12.5595 |          0.309045 |         0.512434 |
| baseline    | episode_000002 |          4 | baseline            | episode_000005    |                  4 |        14.3631 |          0.285823 |         0.44577  |
| high        | episode_000004 |          3 | high                | episode_000000    |                  3 |        20.9847 |          0.443336 |         0.440606 |
| high        | episode_000000 |          3 | high                | episode_000004    |                  3 |        20.9847 |          0.443336 |         0.440606 |
| baseline    | episode_000002 |          2 | baseline            | episode_000004    |                  2 |        18.1545 |          0.322553 |         0.425124 |
| baseline    | episode_000004 |          2 | baseline            | episode_000002    |                  2 |        18.1545 |          0.322553 |         0.425124 |
| low         | episode_000006 |          3 | low                 | episode_000005    |                  3 |        17.0849 |          0.286074 |         0.403107 |
| low         | episode_000005 |          3 | low                 | episode_000006    |                  3 |        17.0849 |          0.286074 |         0.403107 |
| low         | episode_000009 |          0 | high                | episode_000008    |                  0 |        22.6976 |          0.730055 |         0.401693 |
| low         | episode_000005 |          4 | low                 | episode_000003    |                  4 |        17.5896 |          0.297159 |         0.401386 |
| low         | episode_000003 |          0 | high                | episode_000001    |                  0 |        23.3021 |          0.867313 |         0.390772 |

## Height-dimension / candidate-dimension episode score

| condition   | episode        |   episode_idx |   mean_height_action |   sum_height_action |   mean_abs_height_action |   num_chunks |
|:------------|:---------------|--------------:|---------------------:|--------------------:|-------------------------:|-------------:|
| baseline    | episode_000000 |             0 |             11.4169  |             171.253 |                  26.579  |            3 |
| baseline    | episode_000001 |             1 |             10.3133  |             154.7   |                  28.5255 |            3 |
| baseline    | episode_000002 |             2 |             10.9115  |             163.673 |                  28.8106 |            3 |
| baseline    | episode_000003 |             3 |              9.99467 |             149.92  |                  28.6452 |            3 |
| baseline    | episode_000004 |             4 |             11.8339  |             177.508 |                  28.8178 |            3 |
| baseline    | episode_000005 |             5 |             10.8764  |             163.147 |                  28.5844 |            3 |
| baseline    | episode_000006 |             6 |              9.97794 |             149.669 |                  28.3859 |            3 |
| baseline    | episode_000007 |             7 |             12.071   |             181.065 |                  28.5234 |            3 |
| baseline    | episode_000008 |             8 |             10.9516  |             164.274 |                  28.7966 |            3 |
| baseline    | episode_000009 |             9 |              9.81692 |             147.254 |                  28.4301 |            3 |
| high        | episode_000000 |             0 |             11.2857  |             169.286 |                  27.5252 |            3 |
| high        | episode_000001 |             1 |              8.60071 |             129.011 |                  26.9114 |            3 |
| high        | episode_000002 |             2 |              8.36862 |             125.529 |                  26.9243 |            3 |
| high        | episode_000003 |             3 |             10.9298  |             163.947 |                  27.5331 |            3 |
| high        | episode_000004 |             4 |             10.3271  |             154.907 |                  27.5648 |            3 |
| high        | episode_000005 |             5 |              9.06615 |             135.992 |                  27.0984 |            3 |
| high        | episode_000006 |             6 |              8.41003 |             126.15  |                  27.2613 |            3 |
| high        | episode_000007 |             7 |              7.82096 |             117.314 |                  26.5815 |            3 |
| high        | episode_000008 |             8 |             10.6228  |             159.343 |                  27.6364 |            3 |
| high        | episode_000009 |             9 |              8.65969 |             129.895 |                  26.9071 |            3 |
| low         | episode_000000 |             0 |             10.9349  |             164.024 |                  26.8354 |            3 |
| low         | episode_000001 |             1 |             11.5986  |             173.979 |                  28.6464 |            3 |
| low         | episode_000002 |             2 |              9.55189 |             143.278 |                  27.1608 |            3 |
| low         | episode_000003 |             3 |             11.6226  |             174.339 |                  28.7774 |            3 |
| low         | episode_000004 |             4 |             11.2439  |             168.658 |                  28.205  |            3 |
| low         | episode_000005 |             5 |             10.3471  |             155.207 |                  27.541  |            3 |
| low         | episode_000006 |             6 |             10.0155  |             150.233 |                  27.4836 |            3 |
| low         | episode_000007 |             7 |              9.9259  |             148.888 |                  27.3594 |            3 |
| low         | episode_000008 |             8 |              9.54783 |             143.217 |                  27.1043 |            3 |
| low         | episode_000009 |             9 |             10.3941  |             155.911 |                  27.1378 |            3 |

## How to use the outputs

1. Check `episode_scores.csv`: high episodes marked `HIGH_RUN_LOOKS_LOW_LIKE` and low episodes marked `LOW_RUN_LOOKS_HIGH_LIKE` are the opposite-direction failures.
2. Check `projection_score_by_chunk.png`: this shows at which chunk index high/low separation collapses or overlaps.
3. Check `top_unstable_action_dims.csv`: this ranks the action dimensions with high variance across rollouts.
4. Check `top_high_low_separation.csv`: this ranks action dimensions that define high vs low behavior.
5. If the same dimensions appear in both files, intervention is affecting the intended channel but inconsistently.
6. Check `similar_obs_different_action_pairs.csv` and copied images in `inspect_top_mismatch_images/`: low observation distance + high action distance means similar state caused different action.
7. If mismatch pairs have high observation distance, the instability likely starts from physical rollout divergence before that chunk.
8. If you know height-related action dimensions, rerun with `--height-dims` and inspect `height_episode_scores.csv`.
