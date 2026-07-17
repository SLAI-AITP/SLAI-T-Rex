\ Model Transportation Problem with Time Windows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2 x[0,0] + 2 x[0,1] + 2 x[0,2] + 2 x[1,0] + 2 x[1,1] + 2 x[1,2] + 2 x[2,0]
   + 2 x[2,1] + 2 x[2,2] + 2 x[3,0] + 2 x[3,1] + 2 x[3,2]
Subject To
 supply_0: x[0,0] + x[0,1] + x[0,2] <= 25
 supply_1: x[1,0] + x[1,1] + x[1,2] <= 25
 supply_2: x[2,0] + x[2,1] + x[2,2] <= 25
 supply_3: x[3,0] + x[3,1] + x[3,2] <= 25
 demand_0: x[0,0] + x[1,0] + x[2,0] + x[3,0] >= 1
 demand_1: x[0,1] + x[1,1] + x[2,1] + x[3,1] >= 1
 demand_2: x[0,2] + x[1,2] + x[2,2] + x[3,2] >= 1
 time_window_start_0_0: <= 0
 time_window_end_0_0: <= 0
 time_window_start_1_0: <= 0
 time_window_end_1_0: <= 0
 time_window_start_2_0: <= 0
 time_window_end_2_0: <= 0
 time_window_start_3_0: <= 0
 time_window_end_3_0: <= 0
 time_window_start_0_1: <= 0
 time_window_end_0_1: <= 0
 time_window_start_1_1: <= 0
 time_window_end_1_1: <= 0
 time_window_start_2_1: <= 0
 time_window_end_2_1: <= 0
 time_window_start_3_1: <= 0
 time_window_end_3_1: <= 0
 time_window_start_0_2: <= 0
 time_window_end_0_2: <= 0
 time_window_start_1_2: <= 0
 time_window_end_1_2: <= 0
 time_window_start_2_2: <= 0
 time_window_end_2_2: <= 0
 time_window_start_3_2: <= 0
 time_window_end_3_2: <= 0
Bounds
End
