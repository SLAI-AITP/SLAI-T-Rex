\ Model Network Flow Problem with Perishable Goods
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  3 flow[0] + 3 flow[1] + 3 flow[2] + 3 flow[3] + 3 flow[4] + 3 flow[5]
   + 3 flow[6] + 3 flow[7] + 3 flow[8] + 3 flow[9] + 3 flow[10]
   + 3 flow[11] + 3 flow[12] + 3 flow[13] + 3 flow[14] + 3 flow[15]
   + 3 flow[16]
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
 flow_conservation_0: - flow[1] - flow[8] + flow[10] - flow[11] - flow[14]
   = 1
 flow_conservation_1: flow[5] - flow[7] - flow[10] - flow[15] = 1
 flow_conservation_2: - flow[0] - flow[4] = -9
 flow_conservation_3: flow[1] + flow[2] - flow[3] + flow[4] - flow[5]
   - flow[9] + flow[13] + flow[15] = 1
 flow_conservation_4: flow[9] + flow[16] = 1
 flow_conservation_5: flow[7] + flow[11] + flow[12] = 1
 flow_conservation_6: flow[0] - flow[6] = 1
 flow_conservation_7: flow[8] - flow[13] - flow[16] = 1
 flow_conservation_8: - flow[2] + flow[3] - flow[12] + flow[14] = 1
 flow_conservation_9: flow[6] = 1
Bounds
End
