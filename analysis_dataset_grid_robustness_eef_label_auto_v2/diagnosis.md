# Random-placement dataset robustness diagnosis

Dataset: `/home/bruce/.cache/huggingface/lerobot/ethanCSL/svla_koch_pick_n_place_vla_steering_height_test4_fix_dataset`
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
  "total_episodes": 100,
  "total_grid_cells": 20,
  "nonempty_grid_cells": 14,
  "empty_grid_cells": 6,
  "low_count_grid_cells": 11,
  "curate_or_rerecord_grid_cells": 14,
  "ok_grid_cells": 0,
  "num_outlier_episode_flags": 46,
  "coverage_score_0_1": 0.7,
  "count_balance_score_0_1": 0.44999999999999996,
  "consistency_score_0_1": 0.30000000000000004,
  "outlier_score_0_1": 0.54,
  "dataset_health_score_0_100": 48.15,
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
 k    score  silhouette  min_cluster_count  max_cluster_count  low_count_fraction  mean_cluster_count  target_count_penalty       inertia
 7 0.461861    0.524718                 10                 20            0.000000           14.285714              0.785714  37934.777978
 6 0.455908    0.542574                 11                 20            0.000000           16.666667              1.083333  46917.732319
 9 0.452670    0.483782                  8                 20            0.000000           11.111111              0.388889  28050.930846
 5 0.450912    0.570912                 20                 20            0.000000           20.000000              1.500000  55371.198533
 8 0.446510    0.491510                  8                 20            0.000000           12.500000              0.562500  32361.472813
10 0.439341    0.459341                  8                 12            0.000000           10.000000              0.250000  23230.456400
11 0.433225    0.475952                  5                 12            0.090909            9.090909              0.136364  19869.859031
12 0.402095    0.463761                  4                 12            0.166667            8.333333              0.041667  18506.633723
 4 0.321182    0.491182                 21                 31            0.000000           25.000000              2.125000 103064.326550
 3 0.252416    0.505749                 22                 52            0.000000           33.333333              3.166667 164268.743073
```

## Data-driven auto-region health summary

```json
{
  "selected_auto_regions": 7,
  "total_auto_regions": 7,
  "ok_auto_regions": 0,
  "low_count_auto_regions": 0,
  "curate_or_rerecord_auto_regions": 7,
  "num_auto_region_outlier_flags": 33,
  "count_score_0_1": 1.0,
  "consistency_score_0_1": 0.0,
  "outlier_score_0_1": 0.6699999999999999,
  "auto_region_health_score_0_100": 45.05,
  "note": "Auto-regions are data-driven diagnostic regions, not ground-truth block positions. Use them with fixed-grid and RGB overlays."
}
```

## Data-driven auto-region advice

```
auto_region_id             status                                                         flags  needed_demos  num_episodes  num_high  num_low  high_low_balance_ratio  eef_z_max_range  prelift_wrist_iqr  prelift_elbow_iqr                                                 episode_ids                                                                                                                                                                                                                                                                                                advice
     region_00 CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE             0            11         5        6                0.833333         0.139389          12.682605           6.143741                            40,45,46,47,49,50,51,52,56,58,59                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos
     region_01 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0            12         6        6                1.000000         0.141690          17.350711          35.042855                         60,61,62,65,67,68,70,71,73,74,76,79 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
     region_02 CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE             0            10         5        5                1.000000         0.125317          11.910079           8.953913                               41,42,43,44,48,53,54,55,57,98                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos
     region_03 CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE             0            14         8        6                0.750000         0.132437           9.180609          13.992905                   63,64,66,69,72,75,77,78,82,83,84,89,93,99                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos
     region_04 CURATE_OR_RERECORD                  HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL             0            13         6        7                0.857143         0.136298          19.587744           9.484334                      80,81,85,86,87,88,90,91,92,94,95,96,97                                                                                 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy
     region_05 CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE             0            20        10       10                1.000000         0.142622          12.935233          14.681396 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos
     region_06 CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             0            20        10       10                1.000000         0.141116          22.525903          39.231769           0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 inspect/re-record episodes with abnormal EEF peak height; region contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
```

## Fixed-grid advice

```
     grid_id  grid_x  grid_y                 status                                                         flags  num_episodes  num_high  num_low  high_low_balance_ratio  eef_z_max_range  prelift_wrist_iqr  prelift_elbow_iqr                                     episode_ids                                                                                                                                                                                                                                                                                                   advice
cell_x00_y00       0       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                                                 record demos in this spatial cell if it is inside the intended workspace
cell_x01_y00       1       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                                                 record demos in this spatial cell if it is inside the intended workspace
cell_x02_y00       2       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                                                 record demos in this spatial cell if it is inside the intended workspace
cell_x03_y00       3       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                                                 record demos in this spatial cell if it is inside the intended workspace
cell_x04_y00       4       0 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                                                 record demos in this spatial cell if it is inside the intended workspace
cell_x00_y01       0       1     CURATE_OR_RERECORD                                 LOW_COUNT,MISSING_OR_LOW_HIGH             1         0        1                0.000000         0.000000           0.000000           0.000000                                              50                                                                                                                                                                                                          resume recording: add at least 4 demos in this cell; add at least 1 high-EEF demos in this cell
cell_x01_y01       1       1     CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE            11         4        7                0.571429         0.138653           8.834622           8.788816                42,44,46,47,51,52,53,55,56,57,59                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos
cell_x02_y01       2       1 EMPTY_RECORD_IF_NEEDED                                                    EMPTY_CELL             0         0        0                     NaN              NaN                NaN                NaN                                                                                                                                                                                                                                                                                 record demos in this spatial cell if it is inside the intended workspace
cell_x03_y01       3       1     CURATE_OR_RERECORD                  HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL             9         4        5                0.800000         0.130124          17.398309           6.660110                      22,24,25,29,33,34,35,38,39                                                                                 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy
cell_x04_y01       4       1     CURATE_OR_RERECORD        LOW_COUNT,HIGH_EEF_STD,HIGH_EEF_RANGE,ELBOW_MULTIMODAL             2         1        1                1.000000         0.107201           5.334457          15.543768                                           26,32                            resume recording: add at least 3 demos in this cell; inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x00_y02       0       2     CURATE_OR_RERECORD                                  LOW_COUNT,MISSING_OR_LOW_LOW             2         2        0                0.000000         0.022987           3.225570           4.137980                                           45,49                                                                                                                                                                                                           resume recording: add at least 3 demos in this cell; add at least 1 low-EEF demos in this cell
cell_x01_y02       1       2     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             7         3        4                0.750000         0.126586          17.592356          33.300548                            40,41,43,54,58,74,77 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x02_y02       2       2     CURATE_OR_RERECORD                  HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL            16         8        8                1.000000         0.136298          19.538314          10.464382 48,80,81,82,85,86,88,89,90,91,93,94,95,97,98,99                                                                                 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy
cell_x03_y02       3       2     CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE             5         3        2                0.666667         0.140335           9.109215           9.877758                                  20,21,28,30,36                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos
cell_x04_y02       4       2     CURATE_OR_RERECORD                         LOW_COUNT,HIGH_EEF_STD,HIGH_EEF_RANGE             4         2        2                1.000000         0.123372           9.788379          10.508290                                     23,27,31,37                                                                                                            resume recording: add at least 1 demos in this cell; inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos
cell_x00_y03       0       3     CURATE_OR_RERECORD                                 LOW_COUNT,MISSING_OR_LOW_HIGH             1         0        1                0.000000         0.000000           0.000000           0.000000                                              76                                                                                                                                                                                                          resume recording: add at least 4 demos in this cell; add at least 1 high-EEF demos in this cell
cell_x01_y03       1       3     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL            16         9        7                0.777778         0.145785          15.703144          34.398272 60,61,62,63,64,65,66,67,68,70,71,72,73,75,78,79 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x02_y03       2       3     CURATE_OR_RERECORD                                   HIGH_EEF_STD,HIGH_EEF_RANGE             6         4        2                0.500000         0.126238           4.053013          10.431010                               69,83,84,87,92,96                                                                                                                                                                 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos
cell_x03_y03       3       3     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL            14         6        8                0.750000         0.141116          18.244059          40.029155             0,1,5,6,8,9,10,11,12,15,16,17,18,19 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
cell_x04_y03       4       3     CURATE_OR_RERECORD HIGH_EEF_STD,HIGH_EEF_RANGE,WRIST_MULTIMODAL,ELBOW_MULTIMODAL             6         4        2                0.500000         0.116800          20.935085          19.687017                                   2,3,4,7,13,14 inspect/re-record episodes with abnormal EEF peak height; same cell contains mixed lift outcomes; curate or re-record inconsistent demos; pre-lift wrist style varies too much; re-record with consistent wrist strategy; pre-lift elbow style varies too much; re-record with consistent elbow strategy
```

## Data-driven auto-region metrics

```
auto_region_id  num_episodes  num_high  num_low  high_low_balance_ratio                                                 episode_ids  center_cx  center_cy  eef_z_max_mean  eef_z_max_std  eef_z_max_range  prelift_wrist_mean  prelift_wrist_iqr  prelift_elbow_mean  prelift_elbow_iqr
     region_00            11         5        6                0.833333                            40,45,46,47,49,50,51,52,56,58,59 326.802594 225.414482        0.126083       0.061620         0.139389          -91.181023          12.682605          -52.246861           6.143741
     region_01            12         6        6                1.000000                         60,61,62,65,67,68,70,71,73,74,76,79 330.757292 306.248041        0.127042       0.058186         0.141690          -82.748142          17.350711          -30.680062          35.042855
     region_02            10         5        5                1.000000                               41,42,43,44,48,53,54,55,57,98 369.345065 229.184411        0.128294       0.057739         0.125317          -88.632519          11.910079          -52.125896           8.953913
     region_03            14         8        6                0.750000                   63,64,66,69,72,75,77,78,82,83,84,89,93,99 378.240543 291.470611        0.130786       0.057751         0.132437          -82.362140           9.180609          -37.830948          13.992905
     region_04            13         6        7                0.857143                      80,81,85,86,87,88,90,91,92,94,95,96,97 425.489898 270.338541        0.121300       0.057888         0.136298          -86.932117          19.587744          -42.586548           9.484334
     region_05            20        10       10                1.000000 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 492.352910 227.853200        0.135622       0.059673         0.142622          -87.843884          12.935233          -51.610580          14.681396
     region_06            20        10       10                1.000000           0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 495.897885 308.022780        0.126712       0.054938         0.141116          -80.915681          22.525903          -33.335324          39.231769
```

## Fixed-grid metrics

```
     grid_id  num_episodes  num_high  num_low  high_low_balance_ratio                                     episode_ids  eef_z_max_mean  eef_z_max_std  eef_z_max_range  prelift_wrist_mean  prelift_wrist_iqr  prelift_elbow_mean  prelift_elbow_iqr
cell_x00_y00             0         0        0                     NaN                                                             NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x01_y00             0         0        0                     NaN                                                             NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x02_y00             0         0        0                     NaN                                                             NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x03_y00             0         0        0                     NaN                                                             NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x04_y00             0         0        0                     NaN                                                             NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x00_y01             1         0        1                0.000000                                              50        0.075045       0.000000         0.000000          -99.033425           0.000000          -55.444709           0.000000
cell_x01_y01            11         4        7                0.571429                42,44,46,47,51,52,53,55,56,57,59        0.114211       0.057305         0.138653          -94.175933           8.834622          -57.066946           8.788816
cell_x02_y01             0         0        0                     NaN                                                             NaN            NaN              NaN                 NaN                NaN                 NaN                NaN
cell_x03_y01             9         4        5                0.800000                      22,24,25,29,33,34,35,38,39        0.124474       0.058008         0.130124          -88.914518          17.398309          -51.186518           6.660110
cell_x04_y01             2         1        1                1.000000                                           26,32        0.138386       0.053601         0.107201          -82.920222           5.334457          -48.233104          15.543768
cell_x00_y02             2         2        0                0.000000                                           45,49        0.191608       0.011493         0.022987          -87.723795           3.225570          -55.128566           4.137980
cell_x01_y02             7         3        4                0.750000                            40,41,43,54,58,74,77        0.120314       0.060032         0.126586          -83.041043          17.592356          -40.120235          33.300548
cell_x02_y02            16         8        8                1.000000 48,80,81,82,85,86,88,89,90,91,93,94,95,97,98,99        0.124733       0.057967         0.136298          -87.507780          19.538314          -46.175970          10.464382
cell_x03_y02             5         3        2                0.666667                                  20,21,28,30,36        0.151890       0.062222         0.140335          -87.526819           9.109215          -53.070114           9.877758
cell_x04_y02             4         2        2                1.000000                                     23,27,31,37        0.138991       0.057719         0.123372          -88.293121           9.788379          -52.429043          10.508290
cell_x00_y03             1         0        1                0.000000                                              76        0.070737       0.000000         0.000000          -74.773917           0.000000          -12.139947           0.000000
cell_x01_y03            16         9        7                0.777778 60,61,62,63,64,65,66,67,68,70,71,72,73,75,78,79        0.132221       0.058710         0.145785          -81.744957          15.703144          -31.225850          34.398272
cell_x02_y03             6         4        2                0.500000                               69,83,84,87,92,96        0.143911       0.054084         0.126238          -81.826724           4.053013          -37.090769          10.431010
cell_x03_y03            14         6        8                0.750000             0,1,5,6,8,9,10,11,12,15,16,17,18,19        0.118662       0.054944         0.141116          -79.232809          18.244059          -29.881170          40.029155
cell_x04_y03             6         4        2                0.500000                                   2,3,4,7,13,14        0.145497       0.050127         0.116800          -84.842382          20.935085          -41.395017          19.687017
```

## Auto-region outlier episodes

```
 episode_idx auto_region_id                flag                     metric      value  robust_z  eef_z_max  prelift_wrist  prelift_elbow  peak_episode_step
          40      region_00    EEF_PEAK_OUTLIER                  eef_z_max   0.191546  6.933676   0.191546     -86.167760     -51.433188                158
          45      region_00    EEF_PEAK_OUTLIER                  eef_z_max   0.203101  7.621432   0.203101     -90.949364     -59.266545                192
          46      region_00    EEF_PEAK_OUTLIER                  eef_z_max   0.202365  7.577616   0.202365     -92.055063     -61.978362                191
          46      region_00       ELBOW_OUTLIER action.elbow_flex.pos.mean -61.978362 -2.622590   0.202365     -92.055063     -61.978362                191
          47      region_00    EEF_PEAK_OUTLIER                  eef_z_max   0.188890  6.775642   0.188890     -83.407168     -63.650414                198
          47      region_00       ELBOW_OUTLIER action.elbow_flex.pos.mean -63.650414 -3.121133   0.188890     -83.407168     -63.650414                198
          49      region_00    EEF_PEAK_OUTLIER                  eef_z_max   0.180115  6.253345   0.180115     -84.498225     -50.990586                213
          58      region_00       ELBOW_OUTLIER action.elbow_flex.pos.mean -21.455670  9.459755   0.071448     -73.448540     -21.455670                157
          54      region_02       ELBOW_OUTLIER action.elbow_flex.pos.mean -25.628776  5.080176   0.064960     -76.414161     -25.628776                169
          98      region_02       ELBOW_OUTLIER action.elbow_flex.pos.mean -17.851623  6.251425   0.065082     -68.212940     -17.851623                130
          72      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.061330 -2.714533   0.061330     -95.006042     -50.800900                206
          72      region_03 PEAK_TIMING_OUTLIER          peak_episode_step 206.000000  2.578971   0.061330     -95.006042     -50.800900                206
          75      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.062987 -2.671111   0.062987     -76.831545     -19.453421                170
          75      region_03       ELBOW_OUTLIER action.elbow_flex.pos.mean -19.453421  2.625661   0.062987     -76.831545     -19.453421                170
          77      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.065735 -2.599071   0.065735     -60.897740      -6.231558                123
          77      region_03       ELBOW_OUTLIER action.elbow_flex.pos.mean  -6.231558  4.121724   0.065735     -60.897740      -6.231558                123
          77      region_03 PEAK_TIMING_OUTLIER          peak_episode_step 123.000000 -4.007324   0.065735     -60.897740      -6.231558                123
          77      region_03       WRIST_OUTLIER action.wrist_flex.pos.mean -60.897740  2.908932   0.065735     -60.897740      -6.231558                123
          78      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.065014 -2.617970   0.065014     -64.969064      -3.554869                144
          78      region_03       ELBOW_OUTLIER action.elbow_flex.pos.mean  -3.554869  4.424593   0.065014     -64.969064      -3.554869                144
          99      region_03    EEF_PEAK_OUTLIER                  eef_z_max   0.061915 -2.699217   0.061915     -97.437116     -48.875930                181
          80      region_04    EEF_PEAK_OUTLIER                  eef_z_max   0.189930  5.643929   0.189930     -79.804488     -35.752424                165
          81      region_04    EEF_PEAK_OUTLIER                  eef_z_max   0.185293  5.419056   0.185293     -90.700400     -54.025573                182
          85      region_04    EEF_PEAK_OUTLIER                  eef_z_max   0.189003  5.598974   0.189003     -90.334274     -48.995363                183
          86      region_04    EEF_PEAK_OUTLIER                  eef_z_max   0.195945  5.935610   0.195945     -89.426279     -61.669243                188
          87      region_04    EEF_PEAK_OUTLIER                  eef_z_max   0.174106  4.876506   0.174106     -79.870392     -43.438247                182
          88      region_04    EEF_PEAK_OUTLIER                  eef_z_max   0.165410  4.454752   0.165410     -70.021601     -42.096389                185
          95      region_04       ELBOW_OUTLIER action.elbow_flex.pos.mean -13.081354  5.044087   0.073555     -69.098963     -13.081354                136
          95      region_04 PEAK_TIMING_OUTLIER          peak_episode_step 136.000000 -2.542346   0.073555     -69.098963     -13.081354                136
          96      region_04       ELBOW_OUTLIER action.elbow_flex.pos.mean  -7.545314  5.840948   0.068461     -65.525571      -7.545314                123
          96      region_04 PEAK_TIMING_OUTLIER          peak_episode_step 123.000000 -3.216846   0.068461     -65.525571      -7.545314                123
          30      region_05       ELBOW_OUTLIER action.elbow_flex.pos.mean -28.284390  2.826337   0.080181     -74.891079     -28.284390                193
          37      region_05       ELBOW_OUTLIER action.elbow_flex.pos.mean -28.256288  2.829254   0.087607     -77.058543     -28.256288                158
```

## Fixed-grid outlier episodes

```
 episode_idx      grid_id                flag                     metric       value  robust_z  eef_z_max  prelift_wrist  prelift_elbow  peak_episode_step
          42 cell_x01_y01    EEF_PEAK_OUTLIER                  eef_z_max    0.176917 10.178605   0.176917     -84.168710     -62.048616                202
          42 cell_x01_y01       WRIST_OUTLIER action.wrist_flex.pos.mean  -84.168710  7.668889   0.176917     -84.168710     -62.048616                202
          44 cell_x01_y01    EEF_PEAK_OUTLIER                  eef_z_max    0.189986 11.478236   0.189986     -87.017171     -63.566110                190
          44 cell_x01_y01       WRIST_OUTLIER action.wrist_flex.pos.mean  -87.017171  6.078698   0.189986     -87.017171     -63.566110                190
          46 cell_x01_y01    EEF_PEAK_OUTLIER                  eef_z_max    0.202365 12.709146   0.202365     -92.055063     -61.978362                191
          46 cell_x01_y01       WRIST_OUTLIER action.wrist_flex.pos.mean  -92.055063  3.266229   0.202365     -92.055063     -61.978362                191
          47 cell_x01_y01    EEF_PEAK_OUTLIER                  eef_z_max    0.188890 11.369237   0.188890     -83.407168     -63.650414                198
          47 cell_x01_y01       WRIST_OUTLIER action.wrist_flex.pos.mean  -83.407168  8.094029   0.188890     -83.407168     -63.650414                198
          40 cell_x01_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.191546 12.485522   0.191546     -86.167760     -51.433188                158
          41 cell_x01_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.186876 12.000110   0.186876     -92.765348     -59.645918                180
          43 cell_x01_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.190277 12.353624   0.190277     -92.282065     -62.406913                180
          74 cell_x01_y02 PEAK_TIMING_OUTLIER          peak_episode_step  211.000000  2.575364   0.071355     -99.311684     -54.039624                211
          77 cell_x01_y02       ELBOW_OUTLIER action.elbow_flex.pos.mean   -6.231558  2.778318   0.065735     -60.897740      -6.231558                123
          77 cell_x01_y02 PEAK_TIMING_OUTLIER          peak_episode_step  123.000000 -2.820636   0.065735     -60.897740      -6.231558                123
          78 cell_x01_y03 PEAK_TIMING_OUTLIER          peak_episode_step  144.000000 -2.658324   0.065014     -64.969064      -3.554869                144
          95 cell_x02_y02       ELBOW_OUTLIER action.elbow_flex.pos.mean  -13.081354  4.053753   0.073555     -69.098963     -13.081354                136
          95 cell_x02_y02 PEAK_TIMING_OUTLIER          peak_episode_step  136.000000 -3.096568   0.073555     -69.098963     -13.081354                136
          98 cell_x02_y02       ELBOW_OUTLIER action.elbow_flex.pos.mean  -17.851623  3.514312   0.065082     -68.212940     -17.851623                130
          98 cell_x02_y02 PEAK_TIMING_OUTLIER          peak_episode_step  130.000000 -3.464477   0.065082     -68.212940     -17.851623                130
          92 cell_x02_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.067529 -4.865718   0.067529     -99.392232     -46.501335                198
          92 cell_x02_y03       WRIST_OUTLIER action.wrist_flex.pos.mean  -99.392232 -4.942809   0.067529     -99.392232     -46.501335                198
          96 cell_x02_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.068461 -4.823330   0.068461     -65.525571      -7.545314                123
          96 cell_x02_y03       ELBOW_OUTLIER action.elbow_flex.pos.mean   -7.545314  6.510254   0.068461     -65.525571      -7.545314                123
          96 cell_x02_y03 PEAK_TIMING_OUTLIER          peak_episode_step  123.000000 -3.784694   0.068461     -65.525571      -7.545314                123
          96 cell_x02_y03       WRIST_OUTLIER action.wrist_flex.pos.mean  -65.525571  3.957519   0.068461     -65.525571      -7.545314                123
          22 cell_x03_y01    EEF_PEAK_OUTLIER                  eef_z_max    0.182167 11.224855   0.182167     -86.958592     -59.568639                191
          24 cell_x03_y01    EEF_PEAK_OUTLIER                  eef_z_max    0.199869 13.098470   0.199869     -92.179549     -64.683154                189
          25 cell_x03_y01    EEF_PEAK_OUTLIER                  eef_z_max    0.193700 12.445501   0.193700     -87.537071     -56.119151                181
          29 cell_x03_y01    EEF_PEAK_OUTLIER                  eef_z_max    0.180339 11.031368   0.180339     -82.601691     -49.515245                184
          33 cell_x03_y01       ELBOW_OUTLIER action.elbow_flex.pos.mean  -35.274694  2.843480   0.070866     -73.675538     -35.274694                169
          38 cell_x03_y01       ELBOW_OUTLIER action.elbow_flex.pos.mean  -33.953914  3.034741   0.073012     -77.278219     -33.953914                162
          30 cell_x03_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.080181 -4.031125   0.080181     -74.891079     -28.284390                193
          36 cell_x03_y02    EEF_PEAK_OUTLIER                  eef_z_max    0.072031 -4.321228   0.072031    -100.000000     -52.634537                214
          36 cell_x03_y02 PEAK_TIMING_OUTLIER          peak_episode_step  214.000000  7.082250   0.072031    -100.000000     -52.634537                214
           0 cell_x03_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.179363  9.010617   0.179363     -91.571779     -55.444710                172
           1 cell_x03_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.207857 11.483177   0.207857     -75.008238     -23.612477                155
           5 cell_x03_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.177965  8.889274   0.177965     -75.410976     -29.527892                162
           6 cell_x03_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.184051  9.417440   0.184051     -79.123495     -45.658283                173
           8 cell_x03_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.167506  7.981732   0.167506     -75.330428     -33.996065                165
           9 cell_x03_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.170888  8.275147   0.170888     -72.972579     -26.120556                162
          17 cell_x03_y03       WRIST_OUTLIER action.wrist_flex.pos.mean -100.000000 -2.599124   0.071130    -100.000000     -55.444710                208
          37 cell_x04_y02       ELBOW_OUTLIER action.elbow_flex.pos.mean  -28.256288  7.064840   0.087607     -77.058543     -28.256288                158
           2 cell_x04_y03       ELBOW_OUTLIER action.elbow_flex.pos.mean  -13.622313  2.923241   0.191977     -69.670122     -13.622313                152
          13 cell_x04_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.075178 -4.445019   0.075178     -99.582615     -50.597162                202
          14 cell_x04_y03    EEF_PEAK_OUTLIER                  eef_z_max    0.075230 -4.442629   0.075230    -100.000000     -54.355768                211
          14 cell_x04_y03 PEAK_TIMING_OUTLIER          peak_episode_step  211.000000  2.501271   0.075230    -100.000000     -54.355768                211
```

## How to act on this report

- Use `auto_region_recording_gap_overlay.png` to see dataset-driven regions directly on the RGB image.
- Use `grid_recording_gap_overlay.png` for a fixed, human-readable workspace map.
- `LOW_COUNT` means resume recording in that region.
- `HIGH_EEF_RANGE`, `WRIST_MULTIMODAL`, or `ELBOW_MULTIMODAL` means curate/re-record; adding random demos alone may worsen multimodality.
- Data-driven auto-regions are diagnostic partitions, not physical ground-truth block positions.
