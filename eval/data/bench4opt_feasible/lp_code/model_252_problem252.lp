\ Model Transportation Problem with Capacity Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2 x[0,0] + 2 x[0,1] + 2 x[0,2] + 2 x[0,3] + 2 x[0,4] + 2 x[0,5] + 2 x[1,0]
   + 2 x[1,1] + 2 x[1,2] + 2 x[1,3] + 2 x[1,4] + 2 x[1,5] + 2 x[2,0]
   + 2 x[2,1] + 2 x[2,2] + 2 x[2,3] + 2 x[2,4] + 2 x[2,5] + 2 x[3,0]
   + 2 x[3,1] + 2 x[3,2] + 2 x[3,3] + 2 x[3,4] + 2 x[3,5] + 2 x[4,0]
   + 2 x[4,1] + 2 x[4,2] + 2 x[4,3] + 2 x[4,4] + 2 x[4,5] + 2 x[5,0]
   + 2 x[5,1] + 2 x[5,2] + 2 x[5,3] + 2 x[5,4] + 2 x[5,5] + 2 x[6,0]
   + 2 x[6,1] + 2 x[6,2] + 2 x[6,3] + 2 x[6,4] + 2 x[6,5] + 2 x[7,0]
   + 2 x[7,1] + 2 x[7,2] + 2 x[7,3] + 2 x[7,4] + 2 x[7,5]
Subject To
 supply_0: x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] <= 12.5
 supply_1: x[1,0] + x[1,1] + x[1,2] + x[1,3] + x[1,4] + x[1,5] <= 12.5
 supply_2: x[2,0] + x[2,1] + x[2,2] + x[2,3] + x[2,4] + x[2,5] <= 12.5
 supply_3: x[3,0] + x[3,1] + x[3,2] + x[3,3] + x[3,4] + x[3,5] <= 12.5
 supply_4: x[4,0] + x[4,1] + x[4,2] + x[4,3] + x[4,4] + x[4,5] <= 12.5
 supply_5: x[5,0] + x[5,1] + x[5,2] + x[5,3] + x[5,4] + x[5,5] <= 12.5
 supply_6: x[6,0] + x[6,1] + x[6,2] + x[6,3] + x[6,4] + x[6,5] <= 12.5
 supply_7: x[7,0] + x[7,1] + x[7,2] + x[7,3] + x[7,4] + x[7,5] <= 12.5
 demand_0: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] + x[5,0] + x[6,0]
   + x[7,0] >= 1
 demand_1: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] + x[5,1] + x[6,1]
   + x[7,1] >= 1
 demand_2: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] + x[5,2] + x[6,2]
   + x[7,2] >= 1
 demand_3: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] + x[5,3] + x[6,3]
   + x[7,3] >= 1
 demand_4: x[0,4] + x[1,4] + x[2,4] + x[3,4] + x[4,4] + x[5,4] + x[6,4]
   + x[7,4] >= 1
 demand_5: x[0,5] + x[1,5] + x[2,5] + x[3,5] + x[4,5] + x[5,5] + x[6,5]
   + x[7,5] >= 1
 capacity_0_0: x[0,0] <= 1000
 capacity_0_1: x[0,1] <= 1000
 capacity_0_2: x[0,2] <= 1000
 capacity_0_3: x[0,3] <= 1000
 capacity_0_4: x[0,4] <= 1000
 capacity_0_5: x[0,5] <= 1000
 capacity_1_0: x[1,0] <= 1000
 capacity_1_1: x[1,1] <= 1000
 capacity_1_2: x[1,2] <= 1000
 capacity_1_3: x[1,3] <= 1000
 capacity_1_4: x[1,4] <= 1000
 capacity_1_5: x[1,5] <= 1000
 capacity_2_0: x[2,0] <= 1000
 capacity_2_1: x[2,1] <= 1000
 capacity_2_2: x[2,2] <= 1000
 capacity_2_3: x[2,3] <= 1000
 capacity_2_4: x[2,4] <= 1000
 capacity_2_5: x[2,5] <= 1000
 capacity_3_0: x[3,0] <= 1000
 capacity_3_1: x[3,1] <= 1000
 capacity_3_2: x[3,2] <= 1000
 capacity_3_3: x[3,3] <= 1000
 capacity_3_4: x[3,4] <= 1000
 capacity_3_5: x[3,5] <= 1000
 capacity_4_0: x[4,0] <= 1000
 capacity_4_1: x[4,1] <= 1000
 capacity_4_2: x[4,2] <= 1000
 capacity_4_3: x[4,3] <= 1000
 capacity_4_4: x[4,4] <= 1000
 capacity_4_5: x[4,5] <= 1000
 capacity_5_0: x[5,0] <= 1000
 capacity_5_1: x[5,1] <= 1000
 capacity_5_2: x[5,2] <= 1000
 capacity_5_3: x[5,3] <= 1000
 capacity_5_4: x[5,4] <= 1000
 capacity_5_5: x[5,5] <= 1000
 capacity_6_0: x[6,0] <= 1000
 capacity_6_1: x[6,1] <= 1000
 capacity_6_2: x[6,2] <= 1000
 capacity_6_3: x[6,3] <= 1000
 capacity_6_4: x[6,4] <= 1000
 capacity_6_5: x[6,5] <= 1000
 capacity_7_0: x[7,0] <= 1000
 capacity_7_1: x[7,1] <= 1000
 capacity_7_2: x[7,2] <= 1000
 capacity_7_3: x[7,3] <= 1000
 capacity_7_4: x[7,4] <= 1000
 capacity_7_5: x[7,5] <= 1000
Bounds
End
