# Random-placement dataset robustness diagnosis

Dataset: `/home/bruce/.cache/huggingface/lerobot/ethanCSL/svla_koch_pick_n_place_vla_steering_height_test3_similar_position`
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
  "total_episodes": 90,
  "total_grid_cells": 20,
  "nonempty_grid_cells": 7,
  "empty_grid_cells": 13,
  "low_count_grid_cells": 16,
  "curate_or_rerecord_grid_cells": 4,
  "ok_grid_cells": 1,
  "num_outlier_episode_flags": 10,
  "coverage_score_0_1": 0.35,
  "count_balance_score_0_1": 0.19999999999999996,
  "consistency_score_0_1": 0.8,
  "outlier_score_0_1": 0.8888888888888888,
  "dataset_health_score_0_100": 52.388888888888886,
  "note": "Health score is a heuristic. Use grid_cell_advice.csv and outlier_episodes.csv for actual recording decisions."
}
```

## Data-driven auto-region selection

```json
{
  "selected_k": 7,
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
 7 0.333835    0.382407                  6                 18            0.000000           12.857143              0.607143 12174.181024
 9 0.329409    0.349409                  6                 18            0.000000           10.000000              0.250000  9940.415298
 8 0.329002    0.361502                  6                 19            0.000000           11.250000              0.406250 10563.860881
10 0.318327    0.363327                  3                 15            0.100000            9.000000              0.125000  8448.881152
 6 0.313976    0.383976                 10                 22            0.000000           15.000000              0.875000 14255.901116
 5 0.272764    0.372764                 11                 25            0.000000           18.000000              1.250000 18476.738704
12 0.266691    0.359191                  3                 13            0.250000            7.500000              0.062500  6992.458974
11 0.254484    0.351757                  3                 13            0.272727            8.181818              0.022727  7789.675057
 4 0.236166    0.381166                 17                 28            0.000000           22.500000              1.812500 23630.680989
 3 0.154290    0.374290                 21                 36            0.000000           30.000000              2.750000 32147.318226
```

## Data-driven auto-region health summary

```json
{
  "selected_auto_regions": 7,
  "total_auto_regions": 7,
  "ok_auto_regions": 1,
  "low_count_auto_regions": 0,
  "curate_or_rerecord_auto_regions": 6,
  "num_auto_region_outlier_flags": 29,
  "count_score_0_1": 1.0,
  "consistency_score_0_1": 0.1428571428571429,
  "outlier_score_0_1": 0.6777777777777778,
  "auto_region_health_score_0_100": 52.3095238095238,
  "note": "Auto-regions are data-driven diagnostic regions, not ground-truth block positions. Use them with fixed-grid and RGB overlays."
}
```

## Data-driven auto-region advice

```
auto_region_id             status                                                         flags  needed_demos  num_episodes  eef_z_max_range  prelift_wrist_iqr  prelift_elbow_iqr                                        episode_ids                                                                                                                                                                                                                                                                                                advice
     region_00 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0            16         0.123971          29.465822          28.468807      2,6,12,16,26,32,37,44,50,54,59,63,67,68,72,77 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
     region_01 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0            12         0.118383          20.340128          18.619151                 9,13,15,17,21,27,40,45,56,66,78,88 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
     region_02 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0            18         0.147316          20.836232          23.779331 4,5,7,11,22,25,29,30,31,34,39,43,49,57,60,69,76,85 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
     region_03 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0            17         0.135524          20.766666          32.738514     0,1,3,8,14,19,20,23,28,35,46,51,53,58,71,80,86 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
     region_04 CURATE_OR_RERECORD                  HIGH_EEF_STD,HIGH_EEF_RANGE,ELBOW_MULTIMODAL             0            12         0.126876           4.296489          22.713222                10,18,24,33,38,41,42,48,52,55,79,82                                                                                 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift elbow style varies too much; re-record with consistent elbow strategy
     region_05 CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE             0             9         0.076057          13.370923          11.894056                         36,47,64,65,74,75,83,84,89                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos
     region_06                 OK                                                                           0             6         0.024151           2.262658           8.620205                                  61,62,70,73,81,87                                                                                                                                                                                                                                                                                region appears healthy
```

## Fixed-grid advice

```
     grid_id  grid_x  grid_y                 status                                                         flags  num_episodes  eef_z_max_range  prelift_wrist_iqr  prelift_elbow_iqr                                                                                                                                                             episode_ids                                                                                                                                                                                                                                                                                                   advice
cell_x00_y00       0       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x01_y00       1       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x02_y00       2       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x03_y00       3       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x04_y00       4       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x00_y01       0       1 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x01_y01       1       1 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x02_y01       2       1 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x03_y01       3       1       RESUME_RECORDING                                                     LOW_COUNT             1         0.000000           0.000000           0.000000                                                                                                                                                                      70                                                                                                                                                                                                                                                      resume recording: add at least 4 demos in this cell
cell_x04_y01       4       1 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x00_y02       0       2 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x01_y02       1       2     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             9         0.120195          25.431114          22.432205                                                                                                                                               2,16,26,44,59,63,67,68,77 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x02_y02       2       2     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL            59         0.148609          29.147292          27.729380 0,1,3,4,5,6,7,8,9,10,11,12,13,14,17,18,19,20,21,22,23,24,25,28,29,30,31,32,33,34,37,38,39,40,41,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,60,69,72,76,79,82,85,89 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x03_y02       3       2                     OK                                                                           7         0.024151           2.658075           2.272727                                                                                                                                                    61,62,64,73,74,81,87                                                                                                                                                                                                                                                                                     cell appears healthy
cell_x04_y02       4       2 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x00_y03       0       3 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
cell_x01_y03       1       3     CURATE_OR_RERECORD                                    LOW_COUNT,ELBOW_MULTIMODAL             2         0.029402          12.605718          20.219193                                                                                                                                                                   15,78                                                                                                                                                                      resume recording: add at least 3 demos in this cell; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x02_y03       2       3     CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE             8         0.135219          10.108738          13.961290                                                                                                                                                 27,35,36,66,71,80,86,88                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos
cell_x03_y03       3       3       RESUME_RECORDING                                                     LOW_COUNT             4         0.004021           9.925675          12.896936                                                                                                                                                             65,75,83,84                                                                                                                                                                                                                                                      resume recording: add at least 1 demos in this cell
cell_x04_y03       4       3 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0              NaN                NaN                NaN                                                                                                                                                                                                                                                                                                                                                                                                         record demos in this spatial cell if it is inside the intended workspace
```

## Data-driven auto-region metrics

```
auto_region_id  num_episodes                                        episode_ids  center_cx  center_cy  eef_z_max_mean  eef_z_max_std  eef_z_max_range  prelift_wrist_mean  prelift_wrist_iqr  prelift_elbow_mean  prelift_elbow_iqr
     region_00            16      2,6,12,16,26,32,37,44,50,54,59,63,67,68,72,77 376.490582 255.864495        0.123701       0.045730         0.123971          -82.334877          29.465822          -39.186104          28.468807
     region_01            12                 9,13,15,17,21,27,40,45,56,66,78,88 385.993182 279.948405        0.103009       0.042270         0.118383          -88.529882          20.340128          -45.645988          18.619151
     region_02            18 4,5,7,11,22,25,29,30,31,34,39,43,49,57,60,69,76,85 409.437048 249.185259        0.121151       0.052023         0.147316          -87.225456          20.836232          -42.334083          23.779331
     region_03            17     0,1,3,8,14,19,20,23,28,35,46,51,53,58,71,80,86 410.609709 279.900620        0.104906       0.046046         0.135524          -87.319118          20.766666          -41.789750          32.738514
     region_04            12                10,18,24,33,38,41,42,48,52,55,79,82 430.237448 256.809364        0.130693       0.046717         0.126876          -75.102211           4.296489          -34.388319          22.713222
     region_05             9                         36,47,64,65,74,75,83,84,89 448.726069 287.487793        0.117170       0.028939         0.076057          -78.654853          13.370923          -28.230528          11.894056
     region_06             6                                  61,62,70,73,81,87 455.395083 253.461289        0.105165       0.007883         0.024151          -79.186956           2.262658          -29.738654           8.620205
```

## Fixed-grid metrics

```
     grid_id  num_episodes                                                                                                                                                             episode_ids  eef_z_max_mean  eef_z_max_std  eef_z_max_range  prelift_wrist_mean  prelift_wrist_iqr  prelift_elbow_mean  prelift_elbow_iqr
cell_x00_y00             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x01_y00             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x02_y00             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x03_y00             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x04_y00             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x00_y01             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x01_y01             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x02_y01             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x03_y01             1                                                                                                                                                                      70        0.107219       0.000000         0.000000          -79.130817           0.000000          -33.427006           0.000000
cell_x04_y01             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x00_y02             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x01_y02             9                                                                                                                                               2,16,26,44,59,63,67,68,77        0.116244       0.042785         0.120195          -85.284988          25.431114          -41.130002          22.432205
cell_x02_y02            59 0,1,3,4,5,6,7,8,9,10,11,12,13,14,17,18,19,20,21,22,23,24,25,28,29,30,31,32,33,34,37,38,39,40,41,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,60,69,72,76,79,82,85,89        0.118638       0.050007         0.148609          -83.820084          29.147292          -41.826803          27.729380
cell_x03_y02             7                                                                                                                                                    61,62,64,73,74,81,87        0.103365       0.007740         0.024151          -78.433085           2.658075          -27.872900           2.272727
cell_x04_y02             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x00_y03             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x01_y03             2                                                                                                                                                                   15,78        0.082410       0.014701         0.029402          -87.394282          12.605718          -38.155121          20.219193
cell_x02_y03             8                                                                                                                                                 27,35,36,66,71,80,86,88        0.121957       0.040337         0.135219          -83.002600          10.108738          -29.113391          13.961290
cell_x03_y03             4                                                                                                                                                             65,75,83,84        0.104048       0.001543         0.004021          -84.630030           9.925675          -29.157299          12.896936
cell_x04_y03             0                                                                                                                                                                                     NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
```

## Auto-region outlier episodes

```
 episode_idx auto_region_id                flag                     metric      value  robust_z  eef_z_max  prelift_wrist  prelift_elbow  peak_episode_step
          40      region_01    EEF_PEAK_OUTLIER                  eef_z_max   0.173596  2.650483   0.173596     -81.041996     -49.768160                185
          45      region_01    EEF_PEAK_OUTLIER                  eef_z_max   0.179192  2.824159   0.179192     -69.560284     -51.391036                177
          45      region_01       WRIST_OUTLIER action.wrist_flex.pos.mean -69.560284  3.089119   0.179192     -69.560284     -51.391036                177
          56      region_01       WRIST_OUTLIER action.wrist_flex.pos.mean -62.589242  3.951031   0.159098     -62.589242     -39.089504                168
          78      region_01       ELBOW_OUTLIER action.elbow_flex.pos.mean -17.935928  3.098432   0.097111     -74.788563     -17.935928                148
          35      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.196028  6.856848   0.196028     -84.132099     -44.119713                167
          46      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.168475  5.325663   0.168475     -80.946802     -55.472812                172
          51      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.162652  5.002069   0.162652     -64.551679     -16.594070                146
          51      region_03       WRIST_OUTLIER action.wrist_flex.pos.mean -64.551679  3.582677   0.162652     -64.551679     -16.594070                146
          53      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.159558  4.830122   0.159558     -70.190018     -36.145848                164
          53      region_03       WRIST_OUTLIER action.wrist_flex.pos.mean -70.190018  2.905539   0.159558     -70.190018     -36.145848                164
          58      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.165740  5.173706   0.165740     -68.242228     -23.373613                153
          58      region_03       WRIST_OUTLIER action.wrist_flex.pos.mean -68.242228  3.139459   0.165740     -68.242228     -23.373613                153
          71      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.118008  2.521135   0.118008     -69.413830     -13.123507                158
          71      region_03       WRIST_OUTLIER action.wrist_flex.pos.mean -69.413830  2.998755   0.118008     -69.413830     -13.123507                158
          10      region_04    EEF_PEAK_OUTLIER                  eef_z_max   0.062266 -2.773956   0.062266     -73.404609     -20.956864                138
          18      region_04    EEF_PEAK_OUTLIER                  eef_z_max   0.058064 -2.896993   0.058064     -68.740160     -15.484052                147
          24      region_04    EEF_PEAK_OUTLIER                  eef_z_max   0.068580 -2.589122   0.068580     -99.231133     -60.699733                225
          24      region_04       ELBOW_OUTLIER action.elbow_flex.pos.mean -60.699733 -3.444527   0.068580     -99.231133     -60.699733                225
          24      region_04 PEAK_TIMING_OUTLIER          peak_episode_step 225.000000  4.271833   0.068580     -99.231133     -60.699733                225
          24      region_04       WRIST_OUTLIER action.wrist_flex.pos.mean -99.231133 -5.646165   0.068580     -99.231133     -60.699733                225
          41      region_04       ELBOW_OUTLIER action.elbow_flex.pos.mean -56.913025 -3.040127   0.184939     -79.357814     -56.913025                177
          52      region_04       ELBOW_OUTLIER action.elbow_flex.pos.mean -54.046649 -2.734013   0.162973     -73.448542     -54.046649                174
          36      region_05    EEF_PEAK_OUTLIER                  eef_z_max   0.172623 13.896723   0.172623     -75.418298     -25.052691                165
          47      region_05    EEF_PEAK_OUTLIER                  eef_z_max   0.169367 13.244065   0.169367     -68.989126     -48.306871                175
          61      region_06       ELBOW_OUTLIER action.elbow_flex.pos.mean -50.941408 -3.526403   0.119281     -91.161718     -50.941408                179
          61      region_06       WRIST_OUTLIER action.wrist_flex.pos.mean -91.161718 -6.131820   0.119281     -91.161718     -50.941408                179
          62      region_06 PEAK_TIMING_OUTLIER          peak_episode_step 240.000000  3.547370   0.102448     -76.524000     -19.931151                240
          73      region_06       WRIST_OUTLIER action.wrist_flex.pos.mean -70.746530  3.924365   0.095131     -70.746530     -22.312772                164
```

## Fixed-grid outlier episodes

```
 episode_idx      grid_id                flag                     metric      value   robust_z  eef_z_max  prelift_wrist  prelift_elbow  peak_episode_step
          35 cell_x02_y03    EEF_PEAK_OUTLIER                  eef_z_max   0.196028   3.033373   0.196028     -84.132099     -44.119713                167
          86 cell_x02_y03 PEAK_TIMING_OUTLIER          peak_episode_step 193.000000   4.047000   0.098331     -80.038809     -24.041028                193
          61 cell_x03_y02       ELBOW_OUTLIER action.elbow_flex.pos.mean -50.941408 -13.035603   0.119281     -91.161718     -50.941408                179
          61 cell_x03_y02       WRIST_OUTLIER action.wrist_flex.pos.mean -91.161718  -5.821024   0.119281     -91.161718     -50.941408                179
          62 cell_x03_y02       ELBOW_OUTLIER action.elbow_flex.pos.mean -19.931151   2.634101   0.102448     -76.524000     -19.931151                240
          62 cell_x03_y02 PEAK_TIMING_OUTLIER          peak_episode_step 240.000000   2.742967   0.102448     -76.524000     -19.931151                240
          73 cell_x03_y02       WRIST_OUTLIER action.wrist_flex.pos.mean -70.746530   2.765757   0.095131     -70.746530     -22.312772                164
          75 cell_x03_y03       ELBOW_OUTLIER action.elbow_flex.pos.mean  -9.421105   3.113182   0.102624     -69.640831      -9.421105                139
          75 cell_x03_y03       WRIST_OUTLIER action.wrist_flex.pos.mean -69.640831   3.330137   0.102624     -69.640831      -9.421105                139
          84 cell_x03_y03    EEF_PEAK_OUTLIER                  eef_z_max   0.106645   4.216115   0.106645     -85.296379     -29.977518                160
```

## How to act on this report

- Use `auto_region_recording_gap_overlay.png` to see dataset-driven regions directly on the RGB image.
- Use `grid_recording_gap_overlay.png` for a fixed, human-readable workspace map.
- `LOW_COUNT` means resume recording in that region.
- `HIGH_EEF_RANGE`, `WRIST_MULTIMODAL`, or `ELBOW_MULTIMODAL` means curate/re-record; adding random demos alone may worsen multimodality.
- Data-driven auto-regions are diagnostic partitions, not physical ground-truth block positions.
