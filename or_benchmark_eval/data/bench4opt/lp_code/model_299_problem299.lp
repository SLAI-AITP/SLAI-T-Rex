\ Model Production Planning Problem with Shelf-life Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  452526.4042416983 production[0] + 368777.7059383985 production[1]
   + 673633.1731633837 production[2] + 693529.8644675842 production[3]
   + 978106.3601146266 production[4] + 883247.1970287112 production[5]
   + 189165.228828872 production[6] + 884095.0022296328 production[7]
   + 88221.44140370356 production[8] + 391291.8203178637 production[9]
   + 179633.0086893164 production[10] + 958298.2486014835 production[11]
   + 77870.36258172651 production[12] + 165601.4506270497 production[13]
   + 112912.5560848789 inventory[0] + 148806.0889306098 inventory[1]
   + 151347.5708225231 inventory[2] + 205051.9023177545 inventory[3]
   + 952152.6621032426 inventory[4] + 23667.57828650079 inventory[5]
   + 417120.1221168416 inventory[6] + 682720.9713972622 inventory[7]
   + 931789.7507615039 inventory[8] + 474431.0128193612 inventory[9]
   + 420709.843050064 inventory[10] + 737971.5346845966 inventory[11]
   + 22460.81616152286 inventory[12] + 189871.5980268013 inventory[13]
Subject To
 production_capacity_0: production[0] <= 898383.6409342207
 production_capacity_1: production[1] <= 307188.5096770926
 production_capacity_2: production[2] <= 789478.438122179
 production_capacity_3: production[3] <= 227555.1561695253
 production_capacity_4: production[4] <= 372530.4142739871
 production_capacity_5: production[5] <= 288858.1466449796
 production_capacity_6: production[6] <= 845668.1955327959
 production_capacity_7: production[7] <= 149189.9970678074
 production_capacity_8: production[8] <= 285097.6517310038
 production_capacity_9: production[9] <= 956257.6416373007
 production_capacity_10: production[10] <= 619011.5996316634
 production_capacity_11: production[11] <= 161629.092597219
 production_capacity_12: production[12] <= 535781.3687917689
 production_capacity_13: production[13] <= 183098.2299538917
 demand_satisfaction_0: production[0] + inventory[0] >= 23014.58191068391
 demand_satisfaction_1: production[1] + inventory[1] >= 802187.1135018448
 demand_satisfaction_2: production[2] + inventory[2] >= 835112.5710941944
 demand_satisfaction_3: production[3] + inventory[3] >= 520100.8624734532
 demand_satisfaction_4: production[4] + inventory[4] >= 336541.4920619132
 demand_satisfaction_5: production[5] + inventory[5] >= 579125.900964809
 demand_satisfaction_6: production[6] + inventory[6] >= 656104.7411665705
 demand_satisfaction_7: production[7] + inventory[7] >= 787037.5385580964
 demand_satisfaction_8: production[8] + inventory[8] >= 486662.2432229213
 demand_satisfaction_9: production[9] + inventory[9] >= 666189.490505805
 demand_satisfaction_10: production[10] + inventory[10]
   >= 530636.0636879947
 demand_satisfaction_11: production[11] + inventory[11]
   >= 548920.2435971128
 demand_satisfaction_12: production[12] + inventory[12]
   >= 458549.0849672128
 demand_satisfaction_13: production[13] + inventory[13]
   >= 567684.4092519751
 shelf_life_0: inventory[0] <= 176183.9854306902
 shelf_life_1: inventory[1] <= 708403.0460578681
 shelf_life_2: inventory[2] <= 904806.4359999418
 shelf_life_3: inventory[3] <= 794662.8230089792
 shelf_life_4: inventory[4] <= 162093.4585084574
 shelf_life_5: inventory[5] <= 117800.4174687736
 shelf_life_6: inventory[6] <= 993779.2650260717
 shelf_life_7: inventory[7] <= 574950.6457995402
 shelf_life_8: inventory[8] <= 441447.7491166203
 shelf_life_9: inventory[9] <= 370302.4712004668
 shelf_life_10: inventory[10] <= 205950.2065474148
 shelf_life_11: inventory[11] <= 707427.2468138813
 shelf_life_12: inventory[12] <= 665887.7283008904
 shelf_life_13: inventory[13] <= 978044.5404760775
Bounds
End
