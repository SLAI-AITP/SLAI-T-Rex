\ Model Capital Budgeting Problem with Time Windows and Resource Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  7320.273731308014 x[0] + 1690.983818983618 x[1] + 4925.432721820192 x[2]
   + 8713.848901020283 x[3] + 1385.050480762255 x[4]
Subject To
 budget: x[0] + x[1] + x[2] + x[3] + x[4] <= 1e+06
 time_window_start_0: x[0] >= 0
 time_window_end_0: x[0] <= 1
 time_window_start_1: x[1] >= 0
 time_window_end_1: x[1] <= 1
 time_window_start_2: x[2] >= 0
 time_window_end_2: x[2] <= 1
 time_window_start_3: x[3] >= 0
 time_window_end_3: x[3] <= 1
 time_window_start_4: x[4] >= 0
 time_window_end_4: x[4] <= 1
 resource_limit_0: 1e+06 x[0] + 1e+06 x[1] + 1e+06 x[2] + 1e+06 x[3]
   + 1e+06 x[4] <= 1e+06
 resource_limit_1: 1e+06 x[0] + 1e+06 x[1] + 1e+06 x[2] + 1e+06 x[3]
   + 1e+06 x[4] <= 1e+06
 resource_limit_2: 1e+06 x[0] + 1e+06 x[1] + 1e+06 x[2] + 1e+06 x[3]
   + 1e+06 x[4] <= 1e+06
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
