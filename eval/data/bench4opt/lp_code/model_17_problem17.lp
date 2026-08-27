\ Model Production Planning Problem with Resource Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  5177.43730819554 production[0] + 6038.42509479551 production[1]
   + 4925.196199105472 production[2] + 6822.298460832988 production[3]
   + 1333.033612503964 production[4] + 2434.226397085247 production[5]
Subject To
 demand_0: production[0] >= 1113.937829666871
 demand_1: production[1] >= 1662.785813611107
 demand_2: production[2] >= 4115.921823307677
 demand_3: production[3] >= 3464.018206880819
 demand_4: production[4] >= 1291.93349852653
 demand_5: production[5] >= 4336.344578788025
 storage_0: production[0] <= 9674.082014726593
 storage_1: production[1] <= 5372.577689886866
 storage_2: production[2] <= 1179.178673758696
 storage_3: production[3] <= 8736.256026242296
 storage_4: production[4] <= 1755.365869590214
 storage_5: production[5] <= 6414.397661997471
 shelf_life_0: production[0] <= 8804.54601773795
 shelf_life_1: production[1] <= 9105.173964767129
 shelf_life_2: production[2] <= 4663.061664037102
 shelf_life_3: production[3] <= 4270.908207036917
 shelf_life_4: production[4] <= 6916.875429238931
 shelf_life_5: production[5] <= 1812.756632060225
 resource_0: 94.37305206165638 production[0]
   + 73.71356030199381 production[1] + 95.01777553362004 production[2]
   + 24.56960866885118 production[3] + 58.84829583753942 production[4]
   + 42.8640308498057 production[5] <= 4955.278714562803
 resource_1: 72.69921086305585 production[0]
   + 93.36060929256392 production[1] + 93.2240068987068 production[2]
   + 13.10540651191694 production[3] + 39.60511621667681 production[4]
   + 15.28351296167513 production[5] <= 3288.753284261371
 resource_2: 14.91545503860767 production[0]
   + 52.6737983283214 production[1] + 25.92899778596057 production[2]
   + 80.8620373161705 production[3] + 75.08904869595123 production[4]
   + 18.34027128813233 production[5] <= 6305.328499550586
 resource_3: 12.08130402510139 production[0]
   + 83.06917668477858 production[1] + 88.57192137043647 production[2]
   + 33.59930518568368 production[3] + 24.35786983828626 production[4]
   + 97.02242411807318 production[5] <= 5413.227221378905
Bounds
End
