\ Model Production Planning Problem with Shelf-life Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  888177.6290075566 production[0] + 674774.3693754023 production[1]
   + 827639.8652685044 production[2] + 996280.7586626434 production[3]
   + 634542.028407728 production[4] + 782735.897357578 production[5]
   + 784593.8752854894 production[6] + 406680.2882755318 production[7]
   + 58345.28197962151 production[8] + 720491.9749873247 production[9]
   + 285663.8655577677 production[10] + 56841.10314090879 production[11]
   + 477981.6791184538 production[12] + 40209.88893240192 production[13]
   + 46337.56382059698 inventory[0] + 164205.0552886228 inventory[1]
   + 959016.9100149936 inventory[2] + 702030.1527115175 inventory[3]
   + 525975.0694698681 inventory[4] + 49806.66098109034 inventory[5]
   + 763905.4630549821 inventory[6] + 183259.8349896504 inventory[7]
   + 751731.1802392708 inventory[8] + 496681.286291723 inventory[9]
   + 216656.7882125986 inventory[10] + 222963.9542149099 inventory[11]
   + 851009.6454194078 inventory[12] + 768740.0188625584 inventory[13]
Subject To
 production_capacity_0: production[0] <= 1e+06
 production_capacity_1: production[1] <= 1e+06
 production_capacity_2: production[2] <= 1e+06
 production_capacity_3: production[3] <= 1e+06
 production_capacity_4: production[4] <= 1e+06
 production_capacity_5: production[5] <= 1e+06
 production_capacity_6: production[6] <= 1e+06
 production_capacity_7: production[7] <= 1e+06
 production_capacity_8: production[8] <= 1e+06
 production_capacity_9: production[9] <= 1e+06
 production_capacity_10: production[10] <= 1e+06
 production_capacity_11: production[11] <= 1e+06
 production_capacity_12: production[12] <= 1e+06
 production_capacity_13: production[13] <= 1e+06
 demand_satisfaction_0: production[0] + inventory[0] >= 1
 demand_satisfaction_1: production[1] + inventory[1] >= 1
 demand_satisfaction_2: production[2] + inventory[2] >= 1
 demand_satisfaction_3: production[3] + inventory[3] >= 1
 demand_satisfaction_4: production[4] + inventory[4] >= 1
 demand_satisfaction_5: production[5] + inventory[5] >= 1
 demand_satisfaction_6: production[6] + inventory[6] >= 1
 demand_satisfaction_7: production[7] + inventory[7] >= 1
 demand_satisfaction_8: production[8] + inventory[8] >= 1
 demand_satisfaction_9: production[9] + inventory[9] >= 1
 demand_satisfaction_10: production[10] + inventory[10] >= 1
 demand_satisfaction_11: production[11] + inventory[11] >= 1
 demand_satisfaction_12: production[12] + inventory[12] >= 1
 demand_satisfaction_13: production[13] + inventory[13] >= 1
 shelf_life_0: inventory[0] <= 1e+06
 shelf_life_1: inventory[1] <= 1e+06
 shelf_life_2: inventory[2] <= 1e+06
 shelf_life_3: inventory[3] <= 1e+06
 shelf_life_4: inventory[4] <= 1e+06
 shelf_life_5: inventory[5] <= 1e+06
 shelf_life_6: inventory[6] <= 1e+06
 shelf_life_7: inventory[7] <= 1e+06
 shelf_life_8: inventory[8] <= 1e+06
 shelf_life_9: inventory[9] <= 1e+06
 shelf_life_10: inventory[10] <= 1e+06
 shelf_life_11: inventory[11] <= 1e+06
 shelf_life_12: inventory[12] <= 1e+06
 shelf_life_13: inventory[13] <= 1e+06
Bounds
End
