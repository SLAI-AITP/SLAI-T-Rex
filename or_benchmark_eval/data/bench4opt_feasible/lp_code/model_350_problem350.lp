\ Model Network Flow Problem with Pressure Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  48.03527563670976 flow[0] + 27.2758484981678 flow[1]
   + 50.50070104567892 flow[2] + 47.8077327755918 flow[3]
   + 31.1252211214469 flow[4] + 32.26360515929058 flow[5]
   + 37.26076985946177 flow[6] + 49.9924818907428 flow[7]
   + 27.67177732458327 flow[8]
Subject To
 cap_0: flow[0] <= 1000
 cap_1: flow[1] <= 1000
 cap_2: flow[2] <= 1000
 cap_3: flow[3] <= 1000
 cap_4: flow[4] <= 1000
 cap_5: flow[5] <= 1000
 cap_6: flow[6] <= 1000
 cap_7: flow[7] <= 1000
 cap_8: flow[8] <= 1000
 flow_conservation_0: <= 0
 flow_conservation_1: flow[0] + flow[4] - flow[5] - flow[6] = 10
 flow_conservation_2: - flow[1] = 0
 flow_conservation_3: - flow[4] - flow[7] = 0
 flow_conservation_4: - flow[0] - flow[2] - flow[3] - flow[8] = -10
 flow_conservation_5: flow[6] + flow[7] = 0
 flow_conservation_6: flow[8] = 0
 pressure_min_0: pressure[0] >= 2.46843622990361
 pressure_max_0: pressure[0] <= 2.792089801066342
 pressure_min_1: pressure[1] >= 5.818326529532255
 pressure_max_1: pressure[1] <= 11.48678884345485
 pressure_min_2: pressure[2] >= 7.891896659715956
 pressure_max_2: pressure[2] <= 17.03784097647549
 pressure_min_3: pressure[3] >= 2.974007466435184
 pressure_max_3: pressure[3] <= 8.857426338423595
 pressure_min_4: pressure[4] >= 1.600337961370329
 pressure_max_4: pressure[4] <= 1.866645586881794
 pressure_min_5: pressure[5] >= 9.990942438002566
 pressure_max_5: pressure[5] <= 10.53716565546639
 pressure_min_6: pressure[6] >= 9.691686573062096
 pressure_max_6: pressure[6] <= 11.75803457178455
Bounds
End
