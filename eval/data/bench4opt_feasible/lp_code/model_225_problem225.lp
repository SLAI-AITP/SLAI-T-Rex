\ Model Transportation Problem with Time Windows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - x[0,0] - x[0,1] - x[0,2] - x[1,0] - x[1,1] - x[1,2] - x[2,0] - x[2,1]
   - x[2,2] - x[3,0] - x[3,1] - x[3,2] - x[4,0] - x[4,1] - x[4,2]
Subject To
 collection_point_0: x[0,0] + x[0,1] + x[0,2] <= 10
 collection_point_1: x[1,0] + x[1,1] + x[1,2] <= 10
 collection_point_2: x[2,0] + x[2,1] + x[2,2] <= 10
 collection_point_3: x[3,0] + x[3,1] + x[3,2] <= 10
 collection_point_4: x[4,0] + x[4,1] + x[4,2] <= 10
 facility_0: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0]
   <= 26.66666666666667
 facility_1: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1]
   <= 26.66666666666667
 facility_2: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2]
   <= 26.66666666666667
 pickup_time_window_0_0_start: <= 0
 pickup_time_window_0_0_end: <= 0
 pickup_time_window_0_1_start: <= 0
 pickup_time_window_0_1_end: <= 0
 pickup_time_window_0_2_start: <= 0
 pickup_time_window_0_2_end: <= 0
 pickup_time_window_1_0_start: <= 0
 pickup_time_window_1_0_end: <= 0
 pickup_time_window_1_1_start: <= 0
 pickup_time_window_1_1_end: <= 0
 pickup_time_window_1_2_start: <= 0
 pickup_time_window_1_2_end: <= 0
 pickup_time_window_2_0_start: <= 0
 pickup_time_window_2_0_end: <= 0
 pickup_time_window_2_1_start: <= 0
 pickup_time_window_2_1_end: <= 0
 pickup_time_window_2_2_start: <= 0
 pickup_time_window_2_2_end: <= 0
 pickup_time_window_3_0_start: <= 0
 pickup_time_window_3_0_end: <= 0
 pickup_time_window_3_1_start: <= 0
 pickup_time_window_3_1_end: <= 0
 pickup_time_window_3_2_start: <= 0
 pickup_time_window_3_2_end: <= 0
 pickup_time_window_4_0_start: <= 0
 pickup_time_window_4_0_end: <= 0
 pickup_time_window_4_1_start: <= 0
 pickup_time_window_4_1_end: <= 0
 pickup_time_window_4_2_start: <= 0
 pickup_time_window_4_2_end: <= 0
 delivery_time_window_0_0_start: <= 0
 delivery_time_window_0_0_end: <= 0
 delivery_time_window_0_1_start: <= 0
 delivery_time_window_0_1_end: <= 0
 delivery_time_window_0_2_start: <= 0
 delivery_time_window_0_2_end: <= 0
 delivery_time_window_1_0_start: <= 0
 delivery_time_window_1_0_end: <= 0
 delivery_time_window_1_1_start: <= 0
 delivery_time_window_1_1_end: <= 0
 delivery_time_window_1_2_start: <= 0
 delivery_time_window_1_2_end: <= 0
 delivery_time_window_2_0_start: <= 0
 delivery_time_window_2_0_end: <= 0
 delivery_time_window_2_1_start: <= 0
 delivery_time_window_2_1_end: <= 0
 delivery_time_window_2_2_start: <= 0
 delivery_time_window_2_2_end: <= 0
 delivery_time_window_3_0_start: <= 0
 delivery_time_window_3_0_end: <= 0
 delivery_time_window_3_1_start: <= 0
 delivery_time_window_3_1_end: <= 0
 delivery_time_window_3_2_start: <= 0
 delivery_time_window_3_2_end: <= 0
 delivery_time_window_4_0_start: <= 0
 delivery_time_window_4_0_end: <= 0
 delivery_time_window_4_1_start: <= 0
 delivery_time_window_4_1_end: <= 0
 delivery_time_window_4_2_start: <= 0
 delivery_time_window_4_2_end: <= 0
Bounds
End
