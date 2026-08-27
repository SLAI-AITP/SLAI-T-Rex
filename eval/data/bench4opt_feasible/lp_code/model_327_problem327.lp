\ Model Production Planning Problem with Storage
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  power_output[0] + power_output[1] + power_output[2] + power_output[3]
   + power_output[4] + power_output[5] + power_output[6] + power_output[7]
   + power_output[8] + 15.74845527294551 storage_level
Subject To
 demand: power_output[0] + power_output[1] + power_output[2]
   + power_output[3] + power_output[4] + power_output[5] + power_output[6]
   + power_output[7] + power_output[8] + storage_level >= 1
 capacity_0: power_output[0] <= 1e+06
 capacity_1: power_output[1] <= 1e+06
 capacity_2: power_output[2] <= 1e+06
 capacity_3: power_output[3] <= 1e+06
 capacity_4: power_output[4] <= 1e+06
 capacity_5: power_output[5] <= 1e+06
 capacity_6: power_output[6] <= 1e+06
 capacity_7: power_output[7] <= 1e+06
 capacity_8: power_output[8] <= 1e+06
 emission_cap: 1.298278750195748 power_output[0]
   + 1.497962279638249 power_output[1] + 1.156691214090214 power_output[2]
   + 1.421866706991664 power_output[3] + 1.368190057038761 power_output[4]
   + 1.87132932598305 power_output[5] + 1.328156138004935 power_output[6]
   + 1.313797441663314 power_output[7] + 1.807454162554989 power_output[8]
   <= 1e+06
 storage_capacity: storage_level <= 1e+06
 initial_storage: storage_level >= 985.1416856946232
Bounds
End
