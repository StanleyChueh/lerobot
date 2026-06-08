# Random-placement dataset robustness diagnosis

Dataset: `/home/bruce/.cache/huggingface/lerobot/ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2`
Camera key: `observation.images.top`
ROI: `(250, 120, 570, 340)`

## Method

The script uses traditional computer vision, not a VLM:

```
top-camera frame -> ROI crop -> HSV red threshold -> contour -> cube center
```

It then performs two analyses:

```
1. Fixed-grid analysis: human-readable coverage map.
2. Data-driven auto-region analysis: regions inferred from the dataset cube positions.
```

## Fixed-grid health summary

```json
{
  "total_episodes": 61,
  "total_grid_cells": 20,
  "nonempty_grid_cells": 12,
  "empty_grid_cells": 8,
  "low_count_grid_cells": 15,
  "curate_or_rerecord_grid_cells": 12,
  "ok_grid_cells": 0,
  "num_outlier_episode_flags": 22,
  "coverage_score_0_1": 0.6,
  "count_balance_score_0_1": 0.25,
  "consistency_score_0_1": 0.4,
  "outlier_score_0_1": 0.639344262295082,
  "dataset_health_score_0_100": 44.643442622950815,
  "note": "Health score is a heuristic. Use grid_cell_advice.csv and outlier_episodes.csv for actual recording decisions."
}
```

## Data-driven auto-region selection

```json
{
  "selected_k": 5,
  "fixed_k_requested": 0,
  "min_k": 3,
  "max_k": 12,
  "target_episodes_per_region": 8,
  "min_episodes_per_auto_region": 6,
  "selection_note": "selected_k is chosen by silhouette score with penalties for low-count regions unless --auto-region-k is set."
}
```

Candidate region-count scores:

```
 k    score  silhouette  min_cluster_count  max_cluster_count  low_count_fraction  mean_cluster_count  target_count_penalty      inertia
 5 0.388584    0.430584                  8                 16            0.000000           12.200000              0.525000 38505.175486
 6 0.366852    0.446852                  2                 16            0.166667           10.166667              0.270833 31315.838843
 8 0.349203    0.440453                  2                 16            0.250000            7.625000              0.046875 22728.024325
 7 0.326057    0.433200                  2                 15            0.285714            8.714286              0.089286 26527.279400
 4 0.322736    0.395236                 12                 18            0.000000           15.250000              0.906250 53871.528944
 3 0.298911    0.422244                 18                 23            0.000000           20.333333              1.541667 69865.804231
10 0.290236    0.449236                  3                 13            0.400000            6.100000              0.237500 15636.099242
11 0.234989    0.450444                  2                 13            0.545455            5.545455              0.306818 13274.285718
 9 0.219451    0.387229                  4                 14            0.444444            6.777778              0.152778 21026.946913
12 0.153116    0.415616                  1                 12            0.666667            5.083333              0.364583 12597.691837
```

## Data-driven auto-region health summary

```json
{
  "selected_auto_regions": 5,
  "total_auto_regions": 5,
  "ok_auto_regions": 0,
  "low_count_auto_regions": 0,
  "curate_or_rerecord_auto_regions": 5,
  "num_auto_region_outlier_flags": 36,
  "count_score_0_1": 1.0,
  "consistency_score_0_1": 0.0,
  "outlier_score_0_1": 0.4098360655737705,
  "auto_region_health_score_0_100": 41.147540983606554,
  "note": "Auto-regions are data-driven diagnostic regions, not ground-truth block positions. Use them with fixed-grid and RGB overlays."
}
```

## Data-driven auto-region advice

```
auto_region_id             status                                                         flags  needed_demos  num_episodes  num_high  num_low  high_low_balance_ratio  eef_z_max_range  prelift_wrist_iqr  prelift_elbow_iqr                                   episode_ids                                                                                                                                                                                                                                                                                                advice
     region_00 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0            16         9        7                0.777778         0.086756          31.322082          28.013911 6,8,11,13,14,21,25,28,29,32,41,42,48,49,53,59 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
     region_01 CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE             0             8         5        3                0.600000         0.062843          14.491269           4.733384                         3,9,15,19,20,39,40,47                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos
     region_02 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0            15         7        8                0.875000         0.073961          29.432870          26.643951      0,1,4,5,12,22,26,30,33,52,55,56,57,58,60 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
     region_03 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0            13         6        7                0.857143         0.078813          22.633910          25.418013         7,10,17,23,24,27,31,34,43,44,46,50,54 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
     region_04 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0             9         3        6                0.500000         0.063001          22.392267          17.865673                     2,16,18,35,36,37,38,45,51 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
```

## Fixed-grid advice

```
     grid_id  grid_x  grid_y                 status                                                         flags  num_episodes  num_high  num_low  high_low_balance_ratio  eef_z_max_range  prelift_wrist_iqr  prelift_elbow_iqr                  episode_ids                                                                                                                                                                                                                                                                                                   advice
cell_x00_y00       0       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                              record demos in this spatial cell if it is inside the intended workspace
cell_x01_y00       1       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                              record demos in this spatial cell if it is inside the intended workspace
cell_x02_y00       2       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                              record demos in this spatial cell if it is inside the intended workspace
cell_x03_y00       3       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                              record demos in this spatial cell if it is inside the intended workspace
cell_x04_y00       4       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                              record demos in this spatial cell if it is inside the intended workspace
cell_x00_y01       0       1     CURATE_OR_RERECORD                                  LOW_COUNT,MISSING_OR_LOW_LOW             1         1        0                0.000000         0.000000           0.000000           0.000000                           15                                                                                                                                                                                                           resume recording: add at least 4 demos in this cell; add at least 1 low-EEF demos in this cell
cell_x01_y01       1       1     CURATE_OR_RERECORD                                 LOW_COUNT,MISSING_OR_LOW_HIGH             2         0        2                0.000000         0.010782           3.932193           8.658845                        39,40                                                                                                                                                                                                          resume recording: add at least 3 demos in this cell; add at least 1 high-EEF demos in this cell
cell_x02_y01       2       1     CURATE_OR_RERECORD                                  LOW_COUNT,MISSING_OR_LOW_LOW             2         2        0                0.000000         0.016485           0.333175           1.594773                        12,19                                                                                                                                                                                                           resume recording: add at least 3 demos in this cell; add at least 1 low-EEF demos in this cell
cell_x03_y01       3       1     CURATE_OR_RERECORD                         LOW_COUNT,HIGH_EEF_STD,HIGH_EEF_RANGE             3         2        1                0.500000         0.050747           1.775711           2.121681                     16,18,38                                                                                                            resume recording: add at least 2 demos in this cell; inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos
cell_x04_y01       4       1 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                              record demos in this spatial cell if it is inside the intended workspace
cell_x00_y02       0       2 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                              record demos in this spatial cell if it is inside the intended workspace
cell_x01_y02       1       2     CURATE_OR_RERECORD                  HIGH_EEF_STD,HIGH_EEF_RANGE,ELBOW_MULTIMODAL            10         6        4                0.666667         0.063023          13.647349          19.462203   3,6,9,13,20,29,32,47,49,59                                                                                 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x02_y02       2       2     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL            10         4        6                0.666667         0.068891          31.023688          27.137488   0,1,4,26,30,52,55,56,58,60 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x03_y02       3       2     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL            10         2        8                0.250000         0.068208          23.724967          16.753899 2,27,31,35,36,37,46,50,51,57 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x04_y02       4       2     CURATE_OR_RERECORD                                 LOW_COUNT,MISSING_OR_LOW_HIGH             1         0        1                0.000000         0.000000           0.000000           0.000000                           45                                                                                                                                                                                                          resume recording: add at least 4 demos in this cell; add at least 1 high-EEF demos in this cell
cell_x00_y03       0       3     CURATE_OR_RERECORD                                  LOW_COUNT,MISSING_OR_LOW_LOW             1         1        0                0.000000         0.000000           0.000000           0.000000                           14                                                                                                                                                                                                           resume recording: add at least 4 demos in this cell; add at least 1 low-EEF demos in this cell
cell_x01_y03       1       3     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             9         5        4                0.800000         0.082377          32.343573          28.944780    8,11,21,25,28,41,42,48,53 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x02_y03       2       3     CURATE_OR_RERECORD                         LOW_COUNT,HIGH_EEF_STD,HIGH_EEF_RANGE             4         3        1                0.333333         0.070527           7.490939          10.446818                   5,17,22,33                                                                                                            resume recording: add at least 1 demos in this cell; inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos
cell_x03_y03       3       3     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             8         4        4                1.000000         0.072847          17.832168          21.611985       7,10,23,24,34,43,44,54 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x04_y03       4       3 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                              record demos in this spatial cell if it is inside the intended workspace
```

## Data-driven auto-region metrics

```
auto_region_id  num_episodes  num_high  num_low  high_low_balance_ratio                                   episode_ids  center_cx  center_cy  eef_z_max_mean  eef_z_max_std  eef_z_max_range  prelift_wrist_mean  prelift_wrist_iqr  prelift_elbow_mean  prelift_elbow_iqr
     region_00            16         9        7                0.777778 6,8,11,13,14,21,25,28,29,32,41,42,48,49,53,59 346.157011 289.933370        0.104174       0.026599         0.086756          -78.901530          31.322082          -20.462449          28.013911
     region_01             8         5        3                0.600000                         3,9,15,19,20,39,40,47 356.749819 234.674685        0.107895       0.026504         0.062843          -78.334493          14.491269          -33.996944           4.733384
     region_02            15         7        8                0.875000      0,1,4,5,12,22,26,30,33,52,55,56,57,58,60 417.785222 260.725489        0.096591       0.025526         0.073961          -82.163317          29.432870          -34.655988          26.643951
     region_03            13         6        7                0.857143         7,10,17,23,24,27,31,34,43,44,46,50,54 468.669941 296.789607        0.096727       0.029161         0.078813          -80.229758          22.633910          -23.741637          25.418013
     region_04             9         3        6                0.500000                     2,16,18,35,36,37,38,45,51 489.194011 242.697884        0.088354       0.024696         0.063001          -85.224781          22.392267          -37.942017          17.865673
```

## Fixed-grid metrics

```
     grid_id  num_episodes  num_high  num_low  high_low_balance_ratio                  episode_ids  eef_z_max_mean  eef_z_max_std  eef_z_max_range  prelift_wrist_mean  prelift_wrist_iqr  prelift_elbow_mean  prelift_elbow_iqr
cell_x00_y00             0         0        0                     NaN                                          NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x01_y00             0         0        0                     NaN                                          NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x02_y00             0         0        0                     NaN                                          NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x03_y00             0         0        0                     NaN                                          NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x04_y00             0         0        0                     NaN                                          NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x00_y01             1         1        0                0.000000                           15        0.133169       0.000000         0.000000          -73.529087           0.000000          -27.687228           0.000000
cell_x01_y01             2         0        2                0.000000                        39,40        0.075716       0.005391         0.010782          -92.282064           3.932193          -50.642827           8.658845
cell_x02_y01             2         2        0                0.000000                        12,19        0.119383       0.008243         0.016485          -71.204189           0.333175          -31.003232           1.594773
cell_x03_y01             3         2        1                0.500000                     16,18,38        0.101796       0.023109         0.050747          -75.230355           1.775711          -30.260878           2.121681
cell_x04_y01             0         0        0                     NaN                                          NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x00_y02             0         0        0                     NaN                                          NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x01_y02            10         6        4                0.666667   3,6,9,13,20,29,32,47,49,59        0.103481       0.025269         0.063023          -75.971150          13.647349          -23.378530          19.462203
cell_x02_y02            10         4        6                0.666667   0,1,4,26,30,52,55,56,58,60        0.095873       0.025470         0.068891          -82.466225          31.023688          -35.954054          27.137488
cell_x03_y02            10         2        8                0.250000 2,27,31,35,36,37,46,50,51,57        0.079869       0.022061         0.068208          -87.681324          23.724967          -36.738794          16.753899
cell_x04_y02             1         0        1                0.000000                           45        0.073518       0.000000         0.000000          -99.699776           0.000000          -47.231979           0.000000
cell_x00_y03             1         1        0                0.000000                           14        0.123418       0.000000         0.000000          -70.270568           0.000000          -14.577772           0.000000
cell_x01_y03             9         5        4                0.800000    8,11,21,25,28,41,42,48,53        0.106611       0.028221         0.082377          -81.054199          32.343573          -21.403369          28.944780
cell_x02_y03             4         3        1                0.333333                   5,17,22,33        0.109220       0.028660         0.070527          -78.398565           7.490939          -24.726008          10.446818
cell_x03_y03             8         4        4                1.000000       7,10,23,24,34,43,44,54        0.102144       0.027782         0.072847          -78.891004          17.832168          -21.689265          21.611985
cell_x04_y03             0         0        0                     NaN                                          NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
```

## Auto-region outlier episodes

```
 episode_idx auto_region_id                flag                     metric       value   robust_z  eef_z_max  prelift_wrist  prelift_elbow  peak_episode_step
          32      region_00       ELBOW_OUTLIER action.elbow_flex.pos.mean  -41.070676  -3.261785   0.066010     -97.744666     -41.070676                170
          32      region_00       WRIST_OUTLIER action.wrist_flex.pos.mean  -97.744666  -3.881180   0.066010     -97.744666     -41.070676                170
          41      region_00       ELBOW_OUTLIER action.elbow_flex.pos.mean  -36.630603  -2.745391   0.082142     -97.261378     -36.630603                187
          41      region_00 PEAK_TIMING_OUTLIER          peak_episode_step  187.000000   3.943231   0.082142     -97.261378     -36.630603                187
          41      region_00       WRIST_OUTLIER action.wrist_flex.pos.mean  -97.261378  -3.813832   0.082142     -97.261378     -36.630603                187
          42      region_00       ELBOW_OUTLIER action.elbow_flex.pos.mean  -37.024027  -2.791147   0.080305     -98.879653     -37.024027                186
          42      region_00 PEAK_TIMING_OUTLIER          peak_episode_step  186.000000   3.839462   0.080305     -98.879653     -37.024027                186
          42      region_00       WRIST_OUTLIER action.wrist_flex.pos.mean  -98.879653  -4.039346   0.080305     -98.879653     -37.024027                186
          48      region_00       ELBOW_OUTLIER action.elbow_flex.pos.mean  -36.103696  -2.684110   0.076077     -98.535497     -36.103696                175
          48      region_00 PEAK_TIMING_OUTLIER          peak_episode_step  175.000000   2.698000   0.076077     -98.535497     -36.103696                175
          48      region_00       WRIST_OUTLIER action.wrist_flex.pos.mean  -98.535497  -3.991386   0.076077     -98.535497     -36.103696                175
          53      region_00       ELBOW_OUTLIER action.elbow_flex.pos.mean  -36.679781  -2.751110   0.070389     -98.271885     -36.679781                188
          53      region_00 PEAK_TIMING_OUTLIER          peak_episode_step  188.000000   4.047000   0.070389     -98.271885     -36.679781                188
          53      region_00       WRIST_OUTLIER action.wrist_flex.pos.mean  -98.271885  -3.954651   0.070389     -98.271885     -36.679781                188
          59      region_00       ELBOW_OUTLIER action.elbow_flex.pos.mean  -39.461852  -3.074674   0.078627     -98.813753     -39.461852                171
          59      region_00       WRIST_OUTLIER action.wrist_flex.pos.mean  -98.813753  -4.030162   0.078627     -98.813753     -39.461852                171
          20      region_01       ELBOW_OUTLIER action.elbow_flex.pos.mean  -23.788113   2.702796   0.129033     -71.193205     -23.788113                141
          39      region_01    EEF_PEAK_OUTLIER                  eef_z_max    0.081107  -4.748281   0.081107     -96.214256     -59.301671                207
          39      region_01       ELBOW_OUTLIER action.elbow_flex.pos.mean  -59.301671 -13.456436   0.081107     -96.214256     -59.301671                207
          39      region_01 PEAK_TIMING_OUTLIER          peak_episode_step  207.000000   3.124000   0.081107     -96.214256     -59.301671                207
          39      region_01       WRIST_OUTLIER action.wrist_flex.pos.mean  -96.214256  -8.215945   0.081107     -96.214256     -59.301671                207
          40      region_01    EEF_PEAK_OUTLIER                  eef_z_max    0.070326  -5.917762   0.070326     -88.349871     -41.983982                197
          40      region_01       ELBOW_OUTLIER action.elbow_flex.pos.mean  -41.983982  -5.576613   0.070326     -88.349871     -41.983982                197
          40      region_01       WRIST_OUTLIER action.wrist_flex.pos.mean  -88.349871  -5.487459   0.070326     -88.349871     -41.983982                197
          47      region_01    EEF_PEAK_OUTLIER                  eef_z_max    0.070862  -5.859592   0.070862     -84.673966     -30.651960                161
          47      region_01       WRIST_OUTLIER action.wrist_flex.pos.mean  -84.673966  -4.212133   0.070862     -84.673966     -30.651960                161
          10      region_03    EEF_PEAK_OUTLIER                  eef_z_max    0.142607   2.651239   0.142607     -83.480393     -38.569622                172
          43      region_03       WRIST_OUTLIER action.wrist_flex.pos.mean -100.000000  -2.766076   0.074244    -100.000000     -30.384993                198
          46      region_03       WRIST_OUTLIER action.wrist_flex.pos.mean  -97.905760  -2.542286   0.066540     -97.905760     -37.213714                188
           2      region_04    EEF_PEAK_OUTLIER                  eef_z_max    0.132002   9.097987   0.132002     -73.089738     -42.180694                178
          16      region_04    EEF_PEAK_OUTLIER                  eef_z_max    0.119929   7.219771   0.119929     -77.051222     -32.829844                157
          18      region_04    EEF_PEAK_OUTLIER                  eef_z_max    0.116276   6.651572   0.116276     -73.499800     -29.366306                149
          35      region_04       WRIST_OUTLIER action.wrist_flex.pos.mean  -95.379490  -3.120653   0.075228     -95.379490     -38.829563                205
          37      region_04       WRIST_OUTLIER action.wrist_flex.pos.mean -100.000000  -3.907362   0.069354    -100.000000     -48.208515                201
          45      region_04       WRIST_OUTLIER action.wrist_flex.pos.mean  -99.699776  -3.856245   0.073518     -99.699776     -47.231979                186
          51      region_04       WRIST_OUTLIER action.wrist_flex.pos.mean  -97.532310  -3.487202   0.070696     -97.532310     -49.297457                218
```

## Fixed-grid outlier episodes

```
 episode_idx      grid_id                flag                     metric       value  robust_z  eef_z_max  prelift_wrist  prelift_elbow  peak_episode_step
          32 cell_x01_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.066010 -3.121865   0.066010     -97.744666     -41.070676                170
          32 cell_x01_y02       WRIST_OUTLIER action.wrist_flex.pos.mean  -97.744666 -5.078952   0.066010     -97.744666     -41.070676                170
          47 cell_x01_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.070862 -2.829265   0.070862     -84.673966     -30.651960                161
          47 cell_x01_y02       WRIST_OUTLIER action.wrist_flex.pos.mean  -84.673966 -2.634317   0.070862     -84.673966     -30.651960                161
          59 cell_x01_y02       WRIST_OUTLIER action.wrist_flex.pos.mean  -98.813753 -5.278905   0.078627     -98.813753     -39.461852                171
          41 cell_x01_y03 PEAK_TIMING_OUTLIER          peak_episode_step  187.000000  3.119562   0.082142     -97.261378     -36.630603                187
          41 cell_x01_y03       WRIST_OUTLIER action.wrist_flex.pos.mean  -97.261378 -2.778536   0.082142     -97.261378     -36.630603                187
          42 cell_x01_y03 PEAK_TIMING_OUTLIER          peak_episode_step  186.000000  3.035250   0.080305     -98.879653     -37.024027                186
          42 cell_x01_y03       WRIST_OUTLIER action.wrist_flex.pos.mean  -98.879653 -2.949678   0.080305     -98.879653     -37.024027                186
          48 cell_x01_y03       WRIST_OUTLIER action.wrist_flex.pos.mean  -98.535497 -2.913282   0.076077     -98.535497     -36.103696                175
          53 cell_x01_y03 PEAK_TIMING_OUTLIER          peak_episode_step  188.000000  3.203875   0.070389     -98.271885     -36.679781                188
          53 cell_x01_y03       WRIST_OUTLIER action.wrist_flex.pos.mean  -98.271885 -2.885403   0.070389     -98.271885     -36.679781                188
           0 cell_x02_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.129495  2.829268   0.129495     -65.086223     -33.700997                168
           1 cell_x02_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.136345  3.231497   0.136345     -65.899023     -18.301250                140
          33 cell_x02_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.062384 -3.340745   0.062384     -93.241314     -38.948995                187
          33 cell_x02_y03 PEAK_TIMING_OUTLIER          peak_episode_step  187.000000  3.683808   0.062384     -93.241314     -38.948995                187
          33 cell_x02_y03       WRIST_OUTLIER action.wrist_flex.pos.mean  -93.241314 -3.017566   0.062384     -93.241314     -38.948995                187
           2 cell_x03_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.132002  8.192497   0.132002     -73.089738     -42.180694                178
          27 cell_x03_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.113072  5.690120   0.113072     -73.573024     -27.525643                151
          50 cell_x03_y02       WRIST_OUTLIER action.wrist_flex.pos.mean  -70.534178  2.796329   0.063794     -70.534178     -11.795701                129
          43 cell_x03_y03       WRIST_OUTLIER action.wrist_flex.pos.mean -100.000000 -3.206949   0.074244    -100.000000     -30.384993                198
          44 cell_x03_y03       WRIST_OUTLIER action.wrist_flex.pos.mean  -96.558415 -2.794169   0.079778     -96.558415     -38.597725                193
```

## How to act on this report

- Use `auto_region_recording_gap_overlay.png` to see dataset-driven regions directly on the RGB image.
- Use `grid_recording_gap_overlay.png` for a fixed, human-readable workspace map.
- `LOW_COUNT` means resume recording in that region.
- `HIGH_EEF_RANGE`, `WRIST_MULTIMODAL`, or `ELBOW_MULTIMODAL` means curate/re-record; adding random demos alone may worsen multimodality.
- Data-driven auto-regions are diagnostic partitions, not physical ground-truth block positions.
