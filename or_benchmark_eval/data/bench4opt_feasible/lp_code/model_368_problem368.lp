\ Model Network Flow Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  33.95213511660522 flow[0] + 18.28573992449834 flow[1]
   + 49.0148716010091 flow[2] + 79.05370627250744 flow[3]
   + 82.63953697351288 flow[4] + 8.080375195673284 flow[5]
   + 12.26920650372609 flow[6] + 25.89288476857709 flow[7]
   + 20.49287988613229 flow[8] + 22.30602492067569 flow[9]
   + 4.228918248842907 flow[10] + 46.04572056110174 flow[11]
   + 75.42254934839212 flow[12] + 18.18830623563614 flow[13]
   + 14.42980786264895 flow[14] + 18.04733197700588 flow[15]
   + 11.59858660863532 flow[16] + 54.41445519338989 flow[17]
   + 98.27372033533732 flow[18] + 18.74939951260433 flow[19]
Subject To
 capacity_0: flow[0] <= 1000
 capacity_1: flow[1] <= 1000
 capacity_2: flow[2] <= 1000
 capacity_3: flow[3] <= 1000
 capacity_4: flow[4] <= 1000
 capacity_5: flow[5] <= 1000
 capacity_6: flow[6] <= 1000
 capacity_7: flow[7] <= 1000
 capacity_8: flow[8] <= 1000
 capacity_9: flow[9] <= 1000
 capacity_10: flow[10] <= 1000
 capacity_11: flow[11] <= 1000
 capacity_12: flow[12] <= 1000
 capacity_13: flow[13] <= 1000
 capacity_14: flow[14] <= 1000
 capacity_15: flow[15] <= 1000
 capacity_16: flow[16] <= 1000
 capacity_17: flow[17] <= 1000
 capacity_18: flow[18] <= 1000
 capacity_19: flow[19] <= 1000
 flow_conservation_0: flow[3] - flow[5] + flow[8] - flow[18] = 0
 flow_conservation_1: - flow[12] + flow[13] + flow[18] = 0
 flow_conservation_2: flow[7] - flow[13] + flow[15] - flow[17] = 0
 flow_conservation_3: - flow[1] - flow[7] - flow[8] + flow[19] = 0
 flow_conservation_4: - flow[4] + flow[5] = 0
 flow_conservation_5: - flow[0] - flow[9] + flow[10] - flow[11] + flow[16]
   = -10
 flow_conservation_6: - flow[2] - flow[6] + flow[12] + flow[14] - flow[16]
   = 0
 flow_conservation_7: flow[0] + flow[1] - flow[3] + flow[4] - flow[10]
   + flow[17] = 10
 flow_conservation_8: flow[2] + flow[6] - flow[15] = 0
 flow_conservation_9: flow[9] + flow[11] - flow[14] - flow[19] = 0
Bounds
End
