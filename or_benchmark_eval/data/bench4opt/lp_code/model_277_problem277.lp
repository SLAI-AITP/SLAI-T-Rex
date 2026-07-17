\ Model Transportation Problem with Time Windows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  4.146485268696576 x[0,0] + 10.27582874629765 x[0,1]
   + 24.03342844568322 x[0,2] + 60.59985417594805 x[1,0]
   + 56.56326123092268 x[1,1] + 71.88594167931794 x[1,2]
   + 70.43117238543336 x[2,0] + 42.53246227520422 x[2,1]
   + 45.47169558210151 x[2,2] + 28.54088011483561 x[3,0]
   + 87.06073175850047 x[3,1] + 76.12192934584697 x[3,2]
Subject To
 supply_0: x[0,0] + x[0,1] + x[0,2] <= 7673.954497838496
 supply_1: x[1,0] + x[1,1] + x[1,2] <= 3204.026684231286
 supply_2: x[2,0] + x[2,1] + x[2,2] <= 2255.841356726295
 supply_3: x[3,0] + x[3,1] + x[3,2] <= 1922.456585543568
 demand_0: x[0,0] + x[1,0] + x[2,0] + x[3,0] >= 7666.009702009083
 demand_1: x[0,1] + x[1,1] + x[2,1] + x[3,1] >= 5908.298803735148
 demand_2: x[0,2] + x[1,2] + x[2,2] + x[3,2] >= 6314.432612041357
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
 time_window_start_1_1: <= -1
 time_window_end_1_1: <= 0
 time_window_start_2_1: <= 0
 time_window_end_2_1: <= -1
 time_window_start_3_1: <= 0
 time_window_end_3_1: <= -1
 time_window_start_0_2: <= -1
 time_window_end_0_2: <= 0
 time_window_start_1_2: <= 0
 time_window_end_1_2: <= 0
 time_window_start_2_2: <= -1
 time_window_end_2_2: <= 0
 time_window_start_3_2: <= 0
 time_window_end_3_2: <= -1
Bounds
End
