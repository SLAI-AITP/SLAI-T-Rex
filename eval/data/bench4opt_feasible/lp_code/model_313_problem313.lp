\ Model Network Flow Problem with Time-Dependent Costs
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  6.45 flow[0,0] + 2.37 flow[0,1] + 5.45 flow[0,2] + 8.31 flow[0,3]
   + 3.74 flow[1,0] + 7.62 flow[1,1] + 8.82 flow[1,2] + 6.57 flow[1,3]
   + 10.35 flow[2,0] + 10.13 flow[2,1] + 4.71 flow[2,2] + 3.67 flow[2,3]
   + 8.43 flow[3,0] + 4.55 flow[3,1] + 5.2 flow[3,2] + 4.54 flow[3,3]
   + 4.73 flow[4,0] + 8.46 flow[4,1] + 9.11 flow[4,2] + 2 flow[4,3]
   + 6.43 flow[5,0] + 3.12 flow[5,1] + 9.42 flow[5,2] + 9.86 flow[5,3]
   + 5.21 flow[6,0] + 4.85 flow[6,1] + 3.48 flow[6,2] + 5.99 flow[6,3]
   + 4.71 flow[7,0] + 5.06 flow[7,1] + 10.48 flow[7,2] + 10.01 flow[7,3]
   + 4.95 flow[8,0] + 2.09 flow[8,1] + 3.3 flow[8,2] + 9.9 flow[8,3]
   + 2.39 flow[9,0] + 3.93 flow[9,1] + 9.86 flow[9,2] + 8.32 flow[9,3]
   + 6.45 flow[10,0] + 3.66 flow[10,1] + 2.96 flow[10,2] + 5.96 flow[10,3]
   + 7.84 flow[11,0] + 9.29 flow[11,1] + 8.61 flow[11,2] + 5.58 flow[11,3]
   + 9.62 flow[12,0] + 8.43 flow[12,1] + 4.1 flow[12,2] + 3.26 flow[12,3]
   + 4.64 flow[13,0] + 5.04 flow[13,1] + 5.86 flow[13,2] + 3.18 flow[13,3]
   + 2.54 flow[14,0] + 5.48 flow[14,1] + 6.91 flow[14,2] + 10.06 flow[14,3]
   + 6.45 flow[15,0] + 2.37 flow[15,1] + 5.45 flow[15,2] + 8.31 flow[15,3]
   + 3.74 flow[16,0] + 7.62 flow[16,1] + 8.82 flow[16,2] + 6.57 flow[16,3]
   + 10.35 flow[17,0] + 10.13 flow[17,1] + 4.71 flow[17,2]
   + 3.67 flow[17,3] + 8.43 flow[18,0] + 4.55 flow[18,1] + 5.2 flow[18,2]
   + 4.54 flow[18,3] + 4.73 flow[19,0] + 8.46 flow[19,1] + 9.11 flow[19,2]
   + 2 flow[19,3]
Subject To
 capacity_0_0: flow[0,0] <= 1000
 capacity_0_1: flow[0,1] <= 1000
 capacity_0_2: flow[0,2] <= 1000
 capacity_0_3: flow[0,3] <= 1000
 capacity_1_0: flow[1,0] <= 1000
 capacity_1_1: flow[1,1] <= 1000
 capacity_1_2: flow[1,2] <= 1000
 capacity_1_3: flow[1,3] <= 1000
 capacity_2_0: flow[2,0] <= 1000
 capacity_2_1: flow[2,1] <= 1000
 capacity_2_2: flow[2,2] <= 1000
 capacity_2_3: flow[2,3] <= 1000
 capacity_3_0: flow[3,0] <= 1000
 capacity_3_1: flow[3,1] <= 1000
 capacity_3_2: flow[3,2] <= 1000
 capacity_3_3: flow[3,3] <= 1000
 capacity_4_0: flow[4,0] <= 1000
 capacity_4_1: flow[4,1] <= 1000
 capacity_4_2: flow[4,2] <= 1000
 capacity_4_3: flow[4,3] <= 1000
 capacity_5_0: flow[5,0] <= 1000
 capacity_5_1: flow[5,1] <= 1000
 capacity_5_2: flow[5,2] <= 1000
 capacity_5_3: flow[5,3] <= 1000
 capacity_6_0: flow[6,0] <= 1000
 capacity_6_1: flow[6,1] <= 1000
 capacity_6_2: flow[6,2] <= 1000
 capacity_6_3: flow[6,3] <= 1000
 capacity_7_0: flow[7,0] <= 1000
 capacity_7_1: flow[7,1] <= 1000
 capacity_7_2: flow[7,2] <= 1000
 capacity_7_3: flow[7,3] <= 1000
 capacity_8_0: flow[8,0] <= 1000
 capacity_8_1: flow[8,1] <= 1000
 capacity_8_2: flow[8,2] <= 1000
 capacity_8_3: flow[8,3] <= 1000
 capacity_9_0: flow[9,0] <= 1000
 capacity_9_1: flow[9,1] <= 1000
 capacity_9_2: flow[9,2] <= 1000
 capacity_9_3: flow[9,3] <= 1000
 capacity_10_0: flow[10,0] <= 1000
 capacity_10_1: flow[10,1] <= 1000
 capacity_10_2: flow[10,2] <= 1000
 capacity_10_3: flow[10,3] <= 1000
 capacity_11_0: flow[11,0] <= 1000
 capacity_11_1: flow[11,1] <= 1000
 capacity_11_2: flow[11,2] <= 1000
 capacity_11_3: flow[11,3] <= 1000
 capacity_12_0: flow[12,0] <= 1000
 capacity_12_1: flow[12,1] <= 1000
 capacity_12_2: flow[12,2] <= 1000
 capacity_12_3: flow[12,3] <= 1000
 capacity_13_0: flow[13,0] <= 1000
 capacity_13_1: flow[13,1] <= 1000
 capacity_13_2: flow[13,2] <= 1000
 capacity_13_3: flow[13,3] <= 1000
 capacity_14_0: flow[14,0] <= 1000
 capacity_14_1: flow[14,1] <= 1000
 capacity_14_2: flow[14,2] <= 1000
 capacity_14_3: flow[14,3] <= 1000
 capacity_15_0: flow[15,0] <= 1000
 capacity_15_1: flow[15,1] <= 1000
 capacity_15_2: flow[15,2] <= 1000
 capacity_15_3: flow[15,3] <= 1000
 capacity_16_0: flow[16,0] <= 1000
 capacity_16_1: flow[16,1] <= 1000
 capacity_16_2: flow[16,2] <= 1000
 capacity_16_3: flow[16,3] <= 1000
 capacity_17_0: flow[17,0] <= 1000
 capacity_17_1: flow[17,1] <= 1000
 capacity_17_2: flow[17,2] <= 1000
 capacity_17_3: flow[17,3] <= 1000
 capacity_18_0: flow[18,0] <= 1000
 capacity_18_1: flow[18,1] <= 1000
 capacity_18_2: flow[18,2] <= 1000
 capacity_18_3: flow[18,3] <= 1000
 capacity_19_0: flow[19,0] <= 1000
 capacity_19_1: flow[19,1] <= 1000
 capacity_19_2: flow[19,2] <= 1000
 capacity_19_3: flow[19,3] <= 1000
 flow_conservation_0_0: flow[0,0] - flow[2,0] + flow[12,0] + flow[15,0]
   - flow[17,0] = 1
 flow_conservation_0_1: flow[0,1] - flow[2,1] + flow[12,1] + flow[15,1]
   - flow[17,1] = 1
 flow_conservation_0_2: flow[0,2] - flow[2,2] + flow[12,2] + flow[15,2]
   - flow[17,2] = 1
 flow_conservation_0_3: flow[0,3] - flow[2,3] + flow[12,3] + flow[15,3]
   - flow[17,3] = 1
 flow_conservation_1_0: - flow[0,0] - flow[4,0] + flow[10,0] - flow[11,0]
   - flow[15,0] - flow[19,0] = 1
 flow_conservation_1_1: - flow[0,1] - flow[4,1] + flow[10,1] - flow[11,1]
   - flow[15,1] - flow[19,1] = 1
 flow_conservation_1_2: - flow[0,2] - flow[4,2] + flow[10,2] - flow[11,2]
   - flow[15,2] - flow[19,2] = 1
 flow_conservation_1_3: - flow[0,3] - flow[4,3] + flow[10,3] - flow[11,3]
   - flow[15,3] - flow[19,3] = 1
 flow_conservation_2_0: flow[13,0] = 1
 flow_conservation_2_1: flow[13,1] = 1
 flow_conservation_2_2: flow[13,2] = 1
 flow_conservation_2_3: flow[13,3] = 1
 flow_conservation_3_0: flow[1,0] - flow[5,0] - flow[7,0] + flow[16,0] = 1
 flow_conservation_3_1: flow[1,1] - flow[5,1] - flow[7,1] + flow[16,1] = 1
 flow_conservation_3_2: flow[1,2] - flow[5,2] - flow[7,2] + flow[16,2] = 1
 flow_conservation_3_3: flow[1,3] - flow[5,3] - flow[7,3] + flow[16,3] = 1
 flow_conservation_4_0: flow[4,0] - flow[8,0] + flow[19,0] = 1
 flow_conservation_4_1: flow[4,1] - flow[8,1] + flow[19,1] = 1
 flow_conservation_4_2: flow[4,2] - flow[8,2] + flow[19,2] = 1
 flow_conservation_4_3: flow[4,3] - flow[8,3] + flow[19,3] = 1
 flow_conservation_5_0: flow[5,0] + flow[7,0] + flow[8,0] - flow[10,0] = 1
 flow_conservation_5_1: flow[5,1] + flow[7,1] + flow[8,1] - flow[10,1] = 1
 flow_conservation_5_2: flow[5,2] + flow[7,2] + flow[8,2] - flow[10,2] = 1
 flow_conservation_5_3: flow[5,3] + flow[7,3] + flow[8,3] - flow[10,3] = 1
 flow_conservation_6_0: - flow[1,0] + flow[6,0] + flow[9,0] + flow[11,0]
   - flow[13,0] - flow[16,0] = 1
 flow_conservation_6_1: - flow[1,1] + flow[6,1] + flow[9,1] + flow[11,1]
   - flow[13,1] - flow[16,1] = 1
 flow_conservation_6_2: - flow[1,2] + flow[6,2] + flow[9,2] + flow[11,2]
   - flow[13,2] - flow[16,2] = 1
 flow_conservation_6_3: - flow[1,3] + flow[6,3] + flow[9,3] + flow[11,3]
   - flow[13,3] - flow[16,3] = 1
 flow_conservation_7_0: - flow[9,0] - flow[12,0] + flow[14,0] = 1
 flow_conservation_7_1: - flow[9,1] - flow[12,1] + flow[14,1] = 1
 flow_conservation_7_2: - flow[9,2] - flow[12,2] + flow[14,2] = 1
 flow_conservation_7_3: - flow[9,3] - flow[12,3] + flow[14,3] = 1
Bounds
End
