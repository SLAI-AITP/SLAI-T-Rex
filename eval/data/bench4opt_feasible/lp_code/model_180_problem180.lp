\ Model Production Planning Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  354155.572855112 production[0] + 800318.7571649547 production[1]
   + 159083.3158750526 production[2] + 293641.0140961756 production[3]
   + 970541.9386695182 production[4] + 652984.8135941878 production[5]
   + 354155.572855112 production[6] + 800318.7571649547 production[7]
   + 159083.3158750526 production[8]
Subject To
 demand_0: production[0] >= 1
 demand_1: production[1] >= 1
 demand_2: production[2] >= 1
 demand_3: production[3] >= 1
 demand_4: production[4] >= 1
 demand_5: production[5] >= 1
 demand_6: production[6] >= 1
 demand_7: production[7] >= 1
 demand_8: production[8] >= 1
 storage_0: production[0] <= 1e+06
 storage_1: production[1] <= 1e+06
 storage_2: production[2] <= 1e+06
 storage_3: production[3] <= 1e+06
 storage_4: production[4] <= 1e+06
 storage_5: production[5] <= 1e+06
 storage_6: production[6] <= 1e+06
 storage_7: production[7] <= 1e+06
 storage_8: production[8] <= 1e+06
 shelf_life_0: production[0] <= 1e+06
 shelf_life_1: production[1] <= 1e+06
 shelf_life_2: production[2] <= 1e+06
 shelf_life_3: production[3] <= 1e+06
 shelf_life_4: production[4] <= 1e+06
 shelf_life_5: production[5] <= 1e+06
 shelf_life_6: production[6] <= 1e+06
 shelf_life_7: production[7] <= 1e+06
 shelf_life_8: production[8] <= 1e+06
Bounds
End
