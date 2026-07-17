\ Model Production Planning Problem with Nutritional Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  875149.364717176 production[0] + 876935.4648311235 production[1]
   + 94175.44759522105 production[2] + 863431.506981761 production[3]
   + 845801.7642902095 production[4]
Subject To
 demand_0: production[0] >= 1
 demand_1: production[1] >= 1
 demand_2: production[2] >= 1
 demand_3: production[3] >= 1
 demand_4: production[4] >= 1
 storage_0: production[0] <= 1e+06
 storage_1: production[1] <= 1e+06
 storage_2: production[2] <= 1e+06
 storage_3: production[3] <= 1e+06
 storage_4: production[4] <= 1e+06
 shelf_life_0: production[0] <= 1e+06
 shelf_life_1: production[1] <= 1e+06
 shelf_life_2: production[2] <= 1e+06
 shelf_life_3: production[3] <= 1e+06
 shelf_life_4: production[4] <= 1e+06
 nutrition_0: 503100.7040666759 production[0]
   + 23810.69949137765 production[1] + 904014.3102349248 production[2]
   + 176043.9537020153 production[3] + 889086.0839086729 production[4]
   >= 0
 nutrition_1: 8966.112919538793 production[0]
   + 460526.7136966417 production[1] + 225414.109984753 production[2]
   + 987666.5241372171 production[3] + 410494.6446482999 production[4]
   >= 0
 nutrition_2: 983581.1034847872 production[0]
   + 260692.7760049469 production[1] + 851076.3149757243 production[2]
   + 891285.409309926 production[3] + 60580.01068375017 production[4] >= 0
 nutrition_3: 503100.7040666759 production[0]
   + 23810.69949137765 production[1] + 904014.3102349248 production[2]
   + 176043.9537020153 production[3] + 889086.0839086729 production[4]
   >= 0
Bounds
End
