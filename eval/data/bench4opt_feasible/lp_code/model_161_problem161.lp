\ Model Production Planning Problem with Quality Control
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  6292.527017172279 production[0] + 2243.016345408499 production[1]
   + 1842.478769497382 production[2] + 8674.343846159236 production[3]
   + 7751.580273139443 production[4] + 6292.527017172279 production[5]
   + 2243.016345408499 production[6] + 1842.478769497382 production[7]
Subject To
 demand_0: production[0] >= 1
 demand_1: production[1] >= 1
 demand_2: production[2] >= 1
 demand_3: production[3] >= 1
 demand_4: production[4] >= 1
 demand_5: production[5] >= 1
 demand_6: production[6] >= 1
 demand_7: production[7] >= 1
 storage_0: production[0] <= 1e+06
 storage_1: production[1] <= 1e+06
 storage_2: production[2] <= 1e+06
 storage_3: production[3] <= 1e+06
 storage_4: production[4] <= 1e+06
 storage_5: production[5] <= 1e+06
 storage_6: production[6] <= 1e+06
 storage_7: production[7] <= 1e+06
 shelf_life_0: production[0] <= 1e+06
 shelf_life_1: production[1] <= 1e+06
 shelf_life_2: production[2] <= 1e+06
 shelf_life_3: production[3] <= 1e+06
 shelf_life_4: production[4] <= 1e+06
 shelf_life_5: production[5] <= 1e+06
 shelf_life_6: production[6] <= 1e+06
 shelf_life_7: production[7] <= 1e+06
 quality_0: 100 production[0] >= 0
 quality_1: 100 production[1] >= 0
 quality_2: 100 production[2] >= 0
 quality_3: 100 production[3] >= 0
 quality_4: 100 production[4] >= 0
 quality_5: 100 production[5] >= 0
 quality_6: 100 production[6] >= 0
 quality_7: 100 production[7] >= 0
Bounds
End
