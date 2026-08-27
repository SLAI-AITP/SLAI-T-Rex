\ Model Capital Budgeting Problem with Time Windows and Resource Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  44.5473867332 x[0] + 71.6409413985 x[1] + 24.2148642234 x[2]
   + 34.2489061255 x[3] + 89.4096564462 x[4]
Subject To
 budget: 4.9483052252 x[0] + 8.3907022426 x[1] + 19.7687293291 x[2]
   + 8.4874580623 x[3] + 11.4836909782 x[4] <= 141.4867895898
 dependency_0_1: - x[0] + x[1] <= 0
 dependency_0_2: - x[0] + x[2] <= 0
 dependency_1_0: x[0] - x[1] <= 0
 dependency_1_3: - x[1] + x[3] <= 0
 dependency_1_4: - x[1] + x[4] <= 0
 dependency_2_1: x[1] - x[2] <= 0
 dependency_2_3: - x[2] + x[3] <= 0
 dependency_2_4: - x[2] + x[4] <= 0
 dependency_3_1: x[1] - x[3] <= 0
 dependency_3_4: - x[3] + x[4] <= 0
 dependency_4_0: x[0] - x[4] <= 0
 dependency_4_2: x[2] - x[4] <= 0
 dependency_4_3: x[3] - x[4] <= 0
 time_window_start_0: x[0] >= 0
 time_window_end_0: x[0] <= 2
 time_window_start_1: x[1] >= 0
 time_window_end_1: x[1] <= 1
 time_window_start_2: x[2] >= 1
 time_window_end_2: x[2] <= 1
 time_window_start_3: x[3] >= 0
 time_window_end_3: x[3] <= 2
 time_window_start_4: x[4] >= 0
 time_window_end_4: x[4] <= 2
 resource_limit_0: 1.8172902651 x[0] + 4.5469932502 x[1]
   + 2.7825661564 x[2] + 8.3491091491 x[3] + 0.5208560824 x[4]
   <= 22.0585511136
 resource_limit_1: 2.6015631392 x[0] + 3.3750054807 x[1]
   + 9.5120411986 x[2] + 4.7030683621 x[3] + 9.519196923799999 x[4]
   <= 48.7185680833
 resource_limit_2: 3.2735266694 x[0] + 8.796399859999999 x[1]
   + 4.2156482676 x[2] + 7.4902426108 x[3] + 2.2422216572 x[4]
   <= 21.2077441262
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
