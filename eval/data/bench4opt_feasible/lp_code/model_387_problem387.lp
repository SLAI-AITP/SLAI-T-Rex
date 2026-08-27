\ Model Production Planning Problem with Environmental Impact
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  55896.07317234478 production[0] + 720444.2686527272 production[1]
   + 91143.73094905134 production[2] + 218078.2497364437 production[3]
   + 422071.9137000486 production[4] + 990952.5596140927 production[5]
   + 426358.9022886649 production[6] + 465392.2653383671 production[7]
   + 711391.5531923301 production[8] + 930616.667639947 production[9]
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
 demand_9: production[9] >= 1
 storage_0: production[0] <= 1e+06
 storage_1: production[1] <= 1e+06
 storage_2: production[2] <= 1e+06
 storage_3: production[3] <= 1e+06
 storage_4: production[4] <= 1e+06
 storage_5: production[5] <= 1e+06
 storage_6: production[6] <= 1e+06
 storage_7: production[7] <= 1e+06
 storage_8: production[8] <= 1e+06
 storage_9: production[9] <= 1e+06
 shelf_life_0: production[0] <= 1e+06
 shelf_life_1: production[1] <= 1e+06
 shelf_life_2: production[2] <= 1e+06
 shelf_life_3: production[3] <= 1e+06
 shelf_life_4: production[4] <= 1e+06
 shelf_life_5: production[5] <= 1e+06
 shelf_life_6: production[6] <= 1e+06
 shelf_life_7: production[7] <= 1e+06
 shelf_life_8: production[8] <= 1e+06
 shelf_life_9: production[9] <= 1e+06
 emission_limit: 2574.404377021837 production[0]
   + 5147.681345021961 production[1] + 5452.514853699539 production[2]
   + 3374.63496773828 production[3] + 4873.516277826005 production[4]
   + 728.944177488533 production[5] + 1749.284966221532 production[6]
   + 4737.214314468072 production[7] + 3348.372171409565 production[8]
   + 1659.241982908652 production[9] <= 1e+06
Bounds
End
