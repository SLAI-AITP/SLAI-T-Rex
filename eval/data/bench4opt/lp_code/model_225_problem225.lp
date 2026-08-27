\ Model Transportation Problem with Time Windows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  38.86548226628583 x[0,0] + 87.09240824576014 x[0,1]
   + 90.76996384093269 x[0,2] + 38.18890784639677 x[1,0]
   + 68.5903050560526 x[1,1] + 66.51746127442135 x[1,2]
   + 54.39072612796716 x[2,0] + 65.6998757434515 x[2,1]
   + 35.42921882677587 x[2,2] + 18.66888927140706 x[3,0]
   + 54.18859015340212 x[3,1] + 53.35541141485692 x[3,2]
   + 73.05795594970668 x[4,0] + 23.04633138066038 x[4,1]
   + 1.34385619954587 x[4,2]
Subject To
 collection_point_0: x[0,0] + x[0,1] + x[0,2] <= 32.70313615205596
 collection_point_1: x[1,0] + x[1,1] + x[1,2] <= 409.458068883429
 collection_point_2: x[2,0] + x[2,1] + x[2,2] <= 769.2030905208992
 collection_point_3: x[3,0] + x[3,1] + x[3,2] <= 766.061855240788
 collection_point_4: x[4,0] + x[4,1] + x[4,2] <= 978.3549516391769
 facility_0: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0]
   <= 646.2349372790663
 facility_1: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1]
   <= 420.9415769434456
 facility_2: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2]
   <= 992.8637419822981
 pickup_time_window_0_0_start: <= -1
 pickup_time_window_0_0_end: <= 0
 pickup_time_window_0_1_start: <= 0
 pickup_time_window_0_1_end: <= 0
 pickup_time_window_0_2_start: <= -1
 pickup_time_window_0_2_end: <= 0
 pickup_time_window_1_0_start: <= -1
 pickup_time_window_1_0_end: <= 0
 pickup_time_window_1_1_start: <= -1
 pickup_time_window_1_1_end: <= 0
 pickup_time_window_1_2_start: <= -1
 pickup_time_window_1_2_end: <= 0
 pickup_time_window_2_0_start: <= -1
 pickup_time_window_2_0_end: <= 0
 pickup_time_window_2_1_start: <= -1
 pickup_time_window_2_1_end: <= 0
 pickup_time_window_2_2_start: <= -1
 pickup_time_window_2_2_end: <= 0
 pickup_time_window_3_0_start: <= -1
 pickup_time_window_3_0_end: <= 0
 pickup_time_window_3_1_start: <= -1
 pickup_time_window_3_1_end: <= 0
 pickup_time_window_3_2_start: <= -1
 pickup_time_window_3_2_end: <= 0
 pickup_time_window_4_0_start: <= -1
 pickup_time_window_4_0_end: <= 0
 pickup_time_window_4_1_start: <= -1
 pickup_time_window_4_1_end: <= 0
 pickup_time_window_4_2_start: <= -1
 pickup_time_window_4_2_end: <= 0
 delivery_time_window_0_0_start: <= -1
 delivery_time_window_0_0_end: <= 0
 delivery_time_window_0_1_start: <= -1
 delivery_time_window_0_1_end: <= 0
 delivery_time_window_0_2_start: <= -1
 delivery_time_window_0_2_end: <= 0
 delivery_time_window_1_0_start: <= -1
 delivery_time_window_1_0_end: <= 0
 delivery_time_window_1_1_start: <= -1
 delivery_time_window_1_1_end: <= 0
 delivery_time_window_1_2_start: <= -1
 delivery_time_window_1_2_end: <= 0
 delivery_time_window_2_0_start: <= -1
 delivery_time_window_2_0_end: <= 0
 delivery_time_window_2_1_start: <= -1
 delivery_time_window_2_1_end: <= 0
 delivery_time_window_2_2_start: <= -1
 delivery_time_window_2_2_end: <= 0
 delivery_time_window_3_0_start: <= -1
 delivery_time_window_3_0_end: <= 0
 delivery_time_window_3_1_start: <= -1
 delivery_time_window_3_1_end: <= 0
 delivery_time_window_3_2_start: <= -1
 delivery_time_window_3_2_end: <= 0
 delivery_time_window_4_0_start: <= -1
 delivery_time_window_4_0_end: <= 0
 delivery_time_window_4_1_start: <= -1
 delivery_time_window_4_1_end: <= 0
 delivery_time_window_4_2_start: <= -1
 delivery_time_window_4_2_end: <= 0
Bounds
End
