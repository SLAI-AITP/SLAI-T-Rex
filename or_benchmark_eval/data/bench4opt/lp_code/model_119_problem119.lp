\ Model Production Planning Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  440135.8161585915 production[0] + 957834.6706267439 production[1]
   + 807680.9325761118 production[2] + 379273.6935835545 production[3]
   + 22271.65592509108 production[4] + 66219.14772844499 inventory[0]
   + 25682.35822766586 inventory[1] + 37892.8748825168 inventory[2]
   + 80852.75045135333 inventory[3] + 4943.104296567828 inventory[4]
Subject To
 production_capacity_0: production[0] <= 63894.04171766926
 production_capacity_1: production[1] <= 32892.57140916979
 production_capacity_2: production[2] <= 16172.66404494155
 production_capacity_3: production[3] <= 46593.52981142205
 production_capacity_4: production[4] <= 95868.7698480774
 demand_satisfaction_0: production[0] + inventory[0] >= 199.6755989600899
 demand_satisfaction_1: production[1] + inventory[1] >= 5917.174059639755
 demand_satisfaction_2: production[2] + inventory[2] >= 1968.563596109846
 demand_satisfaction_3: production[3] + inventory[3] >= 1550.097650514831
 demand_satisfaction_4: production[4] + inventory[4] >= 5679.908626032999
Bounds
End
