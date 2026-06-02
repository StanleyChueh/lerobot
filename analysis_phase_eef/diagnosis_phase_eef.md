# Phase-aware EEF intervention diagnosis

This analysis uses EEF z height from observation.state via MuJoCo FK, so it can diagnose whether raw chunk index was comparing different task phases.

## Loaded data
```
condition
baseline    51
high        50
low         50
```

## Episode EEF summary
```
condition        episode  num_chunks  eef_z_p90  eef_z_max  eef_z_range  peak_chunk                                                               phase_sequence
 baseline episode_000000           6   0.094673   0.132376     0.118501           3   low_height > low_height > low_height > near_peak > mid_height > low_height
 baseline episode_000001           5   0.069794   0.077611     0.067262           2               low_height > low_height > near_peak > high_height > mid_height
 baseline episode_000002           5   0.059889   0.065174     0.049269           2               low_height > low_height > near_peak > high_height > low_height
 baseline episode_000003           5   0.068638   0.076181     0.062018           2               low_height > low_height > near_peak > high_height > mid_height
 baseline episode_000004           5   0.063491   0.071308     0.060607           2               low_height > low_height > near_peak > high_height > low_height
 baseline episode_000005           5   0.055878   0.061060     0.048138           2               low_height > low_height > near_peak > high_height > low_height
 baseline episode_000006           5   0.051057   0.053348     0.037443           2               low_height > mid_height > near_peak > high_height > mid_height
 baseline episode_000007           5   0.062655   0.071255     0.061943           2               low_height > low_height > near_peak > high_height > low_height
 baseline episode_000008           5   0.055452   0.059917     0.043924           2               low_height > low_height > near_peak > high_height > low_height
 baseline episode_000009           5   0.069465   0.077785     0.069492           2               low_height > low_height > near_peak > high_height > mid_height
     high episode_000000           5   0.081654   0.090976     0.081260           2               low_height > low_height > near_peak > high_height > low_height
     high episode_000001           5   0.098359   0.104742     0.097546           2              low_height > low_height > near_peak > high_height > high_height
     high episode_000002           5   0.094930   0.104576     0.098436           2              low_height > low_height > near_peak > high_height > high_height
     high episode_000003           4   0.067486   0.071588     0.065637           2                            low_height > low_height > near_peak > high_height
     high episode_000004           5   0.073126   0.078873     0.070543           2               low_height > low_height > near_peak > high_height > low_height
     high episode_000005           5   0.095966   0.110050     0.103679           2              low_height > low_height > near_peak > high_height > high_height
     high episode_000006           5   0.098498   0.109366     0.101543           2              low_height > low_height > near_peak > high_height > high_height
     high episode_000007           5   0.095985   0.102518     0.093911           2              low_height > low_height > near_peak > high_height > high_height
     high episode_000008           5   0.068715   0.075996     0.069799           2               low_height > low_height > near_peak > high_height > low_height
     high episode_000009           6   0.093062   0.108277     0.101577           2 low_height > low_height > near_peak > high_height > high_height > low_height
      low episode_000000           5   0.053688   0.061228     0.046327           2               low_height > mid_height > near_peak > high_height > low_height
      low episode_000001           5   0.051908   0.055620     0.039715           2               low_height > low_height > near_peak > high_height > low_height
      low episode_000002           5   0.053958   0.058954     0.043224           2               low_height > low_height > near_peak > high_height > low_height
      low episode_000003           5   0.053526   0.057894     0.041507           2               low_height > low_height > near_peak > high_height > low_height
      low episode_000004           5   0.052498   0.059538     0.046554           2               low_height > low_height > near_peak > high_height > low_height
      low episode_000005           5   0.052454   0.056696     0.040583           2               low_height > low_height > near_peak > high_height > low_height
      low episode_000006           5   0.053355   0.058801     0.042495           2               low_height > low_height > near_peak > high_height > low_height
      low episode_000007           5   0.052024   0.055798     0.039781           2               low_height > mid_height > near_peak > high_height > low_height
      low episode_000008           5   0.051750   0.055769     0.039390           2               low_height > mid_height > near_peak > high_height > low_height
      low episode_000009           5   0.046135   0.049011     0.032520           2               low_height > mid_height > near_peak > high_height > low_height
```

## High intervention EEF modes
```
       episode  episode_idx  eef_z_p90  eef_z_max  eef_z_range  peak_chunk high_eef_group
episode_000001            1   0.098359   0.104742     0.097546           2  high_eef_mode
episode_000002            2   0.094930   0.104576     0.098436           2  high_eef_mode
episode_000005            5   0.095966   0.110050     0.103679           2  high_eef_mode
episode_000006            6   0.098498   0.109366     0.101543           2  high_eef_mode
episode_000007            7   0.095985   0.102518     0.093911           2  high_eef_mode
episode_000000            0   0.081654   0.090976     0.081260           2   low_eef_mode
episode_000003            3   0.067486   0.071588     0.065637           2   low_eef_mode
episode_000004            4   0.073126   0.078873     0.070543           2   low_eef_mode
episode_000008            8   0.068715   0.075996     0.069799           2   low_eef_mode
episode_000009            9   0.093062   0.108277     0.101577           2   low_eef_mode
```

## Most unstable phase-aligned action dimensions
```
condition  phase_target  action_dim      action_key       mean       std        min        max  count     range
 baseline          0.00           1 action_values.1  60.999326 33.855956  33.871185 101.488495     10 67.617310
      low          0.25           1 action_values.1  61.605156 33.697457  34.188812 101.082512     10 66.893700
 baseline          0.50           1 action_values.1  39.173630 31.959044  18.816021  99.474251     10 80.658230
 baseline          0.25           1 action_values.1  87.745864 27.835615  33.920055 101.803894     10 67.883839
 baseline          0.50           2 action_values.2 -38.522981 25.551581 -55.918804   8.853174     10 64.771978
     high          0.75           5 action_values.5  25.536930 22.361443   4.071959  48.205086     10 44.133127
      low          0.25           2 action_values.2  -6.662335 20.676035 -27.273117  17.843540     10 45.116657
      low          0.00           1 action_values.1  92.927008 20.488897  34.739548 100.297195     10 65.557648
 baseline          0.25           2 action_values.2   5.482097 17.433462 -33.207150  17.797253     10 51.004402
 baseline          0.00           2 action_values.2  -4.129847 14.288621 -17.017498  16.122894     10 33.140392
 baseline          0.75           5 action_values.5  35.451846 12.876523   3.913954  45.858177     10 41.944223
      low          0.50           5 action_values.5  28.525817 12.386073  14.904083  47.533920     10 32.629837
 baseline          0.75           1 action_values.1  25.774314 12.012660  18.816021  59.332230     10 40.516209
      low          0.75           5 action_values.5  25.340800 10.969152  14.904083  43.545563     10 28.641479
 baseline          0.50           0 action_values.0   7.378074 10.217912 -10.569910  14.223358     10 24.793268
      low          0.00           2 action_values.2  11.100440  9.453120 -15.665026  15.429848     10 31.094873
 baseline          0.50           4 action_values.4   7.738258  9.363510  -8.216813  14.746068     10 22.962881
     high          1.00           3 action_values.3 -87.905862  8.116388 -97.607414 -78.092117     10 19.515297
      low          0.50           2 action_values.2 -49.290096  7.907222 -52.878960 -26.960564     10 25.918396
     high          0.25           2 action_values.2   3.866385  7.882671  -2.940422  14.643463     10 17.583885
      low          0.50           0 action_values.0  11.242680  7.768294 -10.844300  14.297379     10 25.141679
     high          1.00           1 action_values.1  54.919897  7.738694  44.825600  65.642120     10 20.816521
      low          0.50           4 action_values.4  13.036024  7.519813  -8.297896  16.485825     10 24.783721
 baseline          0.00           3 action_values.3 -79.419693  6.126397 -87.467720 -73.093941     10 14.373779
 baseline          1.00           3 action_values.3 -94.089143  5.341044 -97.363625 -79.177963     10 18.185661
 baseline          0.50           5 action_values.5  43.726008  5.287719  29.715862  47.955742     10 18.239880
      low          0.50           1 action_values.1  22.518288  5.157629  20.070267  37.020218     10 16.949951
 baseline          0.75           3 action_values.3 -92.968614  4.978646 -96.607864 -79.177963     10 17.429901
 baseline          1.00           0 action_values.0  -3.945366  4.961646  -8.292171   8.788286     10 17.080458
 baseline          1.00           1 action_values.1  46.523532  4.818225  42.245216  59.332230     10 17.087013
```

## Same-phase similar observation / different action pairs
```
condition        episode  save_idx  phase_target    eef_z nearest_condition nearest_episode  nearest_save_idx  nearest_eef_z  obs_distance  action_distance  mismatch_score
 baseline episode_000001         4          0.25 0.031464          baseline  episode_000003                 4       0.030955     14.621544         4.808608        0.830556
 baseline episode_000003         4          0.25 0.030955          baseline  episode_000001                 4       0.031464     14.621561         4.808608        0.823911
 baseline episode_000000         0          0.00 0.013875               low  episode_000007                 0       0.016017     17.458439         1.039413        0.732089
 baseline episode_000002         4          0.25 0.017172          baseline  episode_000005                 4       0.016895     14.129044         0.280561        0.654222
      low episode_000005         4          0.25 0.018406          baseline  episode_000004                 4       0.017286     18.626665         0.338563        0.606800
 baseline episode_000009         0          0.25 0.015721              high  episode_000006                 0       0.015705     19.250914         0.485327        0.606667
 baseline episode_000002         2          1.00 0.065174          baseline  episode_000004                 2       0.071308     17.794954         0.301513        0.597944
 baseline episode_000004         2          1.00 0.071308          baseline  episode_000002                 2       0.065174     17.794954         0.301513        0.597944
 baseline episode_000005         4          0.25 0.016895          baseline  episode_000004                 4       0.017286     12.217022         0.158775        0.523900
 baseline episode_000004         4          0.25 0.017286          baseline  episode_000005                 4       0.016895     12.217022         0.158775        0.523900
     high episode_000009         5          0.25 0.017498              high  episode_000008                 4       0.018336     14.440298         0.173787        0.497567
     high episode_000008         4          0.25 0.018336              high  episode_000009                 5       0.017498     14.440298         0.173787        0.497567
      low episode_000002         3          0.75 0.046465               low  episode_000006                 3       0.045187     21.151945         0.313569        0.493333
      low episode_000002         4          0.25 0.021358              high  episode_000000                 4       0.019786     20.498201         0.275732        0.476267
     high episode_000000         4          0.25 0.019786              high  episode_000004                 4       0.018653     18.313953         0.195985        0.472500
     high episode_000004         4          0.25 0.018653              high  episode_000000                 4       0.019786     18.313953         0.195985        0.472500
     high episode_000003         0          0.25 0.015610          baseline  episode_000009                 0       0.015721     23.364637         0.511755        0.471600
      low episode_000004         3          0.75 0.041939               low  episode_000000                 3       0.042379     24.446924         0.867253        0.467500
      low episode_000000         3          0.75 0.042379               low  episode_000004                 3       0.041939     24.446934         0.867253        0.455278
     high episode_000008         1          0.00 0.006197              high  episode_000005                 1       0.006372     19.031353         0.177962        0.445300
```

## Interpretation guide
1. If `eef_height_by_raw_chunk.png` has high spread at a chunk index, raw chunk alignment is invalid.
2. If high episodes split into high_eef_mode and low_eef_mode, high intervention is causing task-progress / EEF-height bifurcation.
3. If phase-aligned action variance is still high, the policy/intervention is unstable even after EEF-phase alignment.
4. If phase-aligned variance becomes small, the previous chunk-index variance was mainly phase misalignment.
5. Inspect `inspect_phase_mismatch_images/` to see whether same-phase images are visually similar.
6. Candidate height/action dims plotted: [1, 2]