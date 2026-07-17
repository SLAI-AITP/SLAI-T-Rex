\ Model Network Flow Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  67.31874824080882 flow[0] + 75.39438068084705 flow[1]
   + 90.30669522319837 flow[2] + 16.54479075750672 flow[3]
   + 28.63486593533748 flow[4] + 37.55814260620647 flow[5]
   + 66.99683254839748 flow[6] + 42.35738815015191 flow[7]
   + 13.56377829709444 flow[8] + 45.17161554122904 flow[9]
   + 72.22400486494161 flow[10] + 73.30447721565352 flow[11]
   + 23.72501650699119 flow[12] + 100.8831205725232 flow[13]
   + 90.74375978614859 flow[14] + 33.89610417650953 flow[15]
   + 6.936768917536009 flow[16] + 47.14051075583311 flow[17]
   + 59.6456702096854 flow[18] + 8.759998507279739 flow[19]
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
 cap_9: flow[9] <= 1000
 cap_10: flow[10] <= 1000
 cap_11: flow[11] <= 1000
 cap_12: flow[12] <= 1000
 cap_13: flow[13] <= 1000
 cap_14: flow[14] <= 1000
 cap_15: flow[15] <= 1000
 cap_16: flow[16] <= 1000
 cap_17: flow[17] <= 1000
 cap_18: flow[18] <= 1000
 cap_19: flow[19] <= 1000
 flow_conservation_0: flow[9] - flow[11] + flow[16] - flow[18] = 0
 flow_conservation_1: - flow[1] - flow[8] - flow[9] - flow[14] = 0
 flow_conservation_2: - flow[4] + flow[5] - flow[7] + flow[11] + flow[15]
   + flow[19] = 0
 flow_conservation_3: - flow[5] - flow[13] + flow[14] - flow[15] - flow[17]
   = 0
 flow_conservation_4: - flow[0] + flow[2] - flow[12] = -10
 flow_conservation_5: flow[10] = 0
 flow_conservation_6: flow[1] + flow[4] - flow[6] + flow[17] = 0
 flow_conservation_7: - flow[2] - flow[3] + flow[6] + flow[12] - flow[19]
   = 0
 flow_conservation_8: flow[3] + flow[8] - flow[10] = 0
 flow_conservation_9: flow[0] + flow[7] + flow[13] - flow[16] + flow[18]
   = 10
Bounds
End
