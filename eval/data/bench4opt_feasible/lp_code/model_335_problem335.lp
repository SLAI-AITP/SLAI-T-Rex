\ Model Network Flow Problem with Time-Dependent Capacities
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  14.31 flow[0,0] + 14.31 flow[0,1] + 14.31 flow[0,2] + 14.31 flow[0,3]
   + 14.31 flow[0,4] + 60.22 flow[1,0] + 60.22 flow[1,1] + 60.22 flow[1,2]
   + 60.22 flow[1,3] + 60.22 flow[1,4] + 80.01 flow[2,0] + 80.01 flow[2,1]
   + 80.01 flow[2,2] + 80.01 flow[2,3] + 80.01 flow[2,4] + 14.08 flow[3,0]
   + 14.08 flow[3,1] + 14.08 flow[3,2] + 14.08 flow[3,3] + 14.08 flow[3,4]
   + 18.21 flow[4,0] + 18.21 flow[4,1] + 18.21 flow[4,2] + 18.21 flow[4,3]
   + 18.21 flow[4,4] + 94.84 flow[5,0] + 94.84 flow[5,1] + 94.84 flow[5,2]
   + 94.84 flow[5,3] + 94.84 flow[5,4] + 47.87 flow[6,0] + 47.87 flow[6,1]
   + 47.87 flow[6,2] + 47.87 flow[6,3] + 47.87 flow[6,4] + 8.35 flow[7,0]
   + 8.35 flow[7,1] + 8.35 flow[7,2] + 8.35 flow[7,3] + 8.35 flow[7,4]
   + 15.47 flow[8,0] + 15.47 flow[8,1] + 15.47 flow[8,2] + 15.47 flow[8,3]
   + 15.47 flow[8,4] + 61.46 flow[9,0] + 61.46 flow[9,1] + 61.46 flow[9,2]
   + 61.46 flow[9,3] + 61.46 flow[9,4] + 78.16 flow[10,0]
   + 78.16 flow[10,1] + 78.16 flow[10,2] + 78.16 flow[10,3]
   + 78.16 flow[10,4] + 65.8 flow[11,0] + 65.8 flow[11,1] + 65.8 flow[11,2]
   + 65.8 flow[11,3] + 65.8 flow[11,4] + 91.85 flow[12,0]
   + 91.85 flow[12,1] + 91.85 flow[12,2] + 91.85 flow[12,3]
   + 91.85 flow[12,4] + 78.25 flow[13,0] + 78.25 flow[13,1]
   + 78.25 flow[13,2] + 78.25 flow[13,3] + 78.25 flow[13,4]
   + 14.31 flow[14,0] + 14.31 flow[14,1] + 14.31 flow[14,2]
   + 14.31 flow[14,3] + 14.31 flow[14,4] + 60.22 flow[15,0]
   + 60.22 flow[15,1] + 60.22 flow[15,2] + 60.22 flow[15,3]
   + 60.22 flow[15,4] + 80.01 flow[16,0] + 80.01 flow[16,1]
   + 80.01 flow[16,2] + 80.01 flow[16,3] + 80.01 flow[16,4]
   + 14.08 flow[17,0] + 14.08 flow[17,1] + 14.08 flow[17,2]
   + 14.08 flow[17,3] + 14.08 flow[17,4]
Subject To
 capacity_0_0: flow[0,0] <= 1000
 capacity_1_0: flow[1,0] <= 1000
 capacity_2_0: flow[2,0] <= 1000
 capacity_3_0: flow[3,0] <= 1000
 capacity_4_0: flow[4,0] <= 1000
 capacity_5_0: flow[5,0] <= 1000
 capacity_6_0: flow[6,0] <= 1000
 capacity_7_0: flow[7,0] <= 1000
 capacity_8_0: flow[8,0] <= 1000
 capacity_9_0: flow[9,0] <= 1000
 capacity_10_0: flow[10,0] <= 1000
 capacity_11_0: flow[11,0] <= 1000
 capacity_12_0: flow[12,0] <= 1000
 capacity_13_0: flow[13,0] <= 1000
 capacity_14_0: flow[14,0] <= 1000
 capacity_15_0: flow[15,0] <= 1000
 capacity_16_0: flow[16,0] <= 1000
 capacity_17_0: flow[17,0] <= 1000
 capacity_0_1: flow[0,1] <= 1000
 capacity_1_1: flow[1,1] <= 1000
 capacity_2_1: flow[2,1] <= 1000
 capacity_3_1: flow[3,1] <= 1000
 capacity_4_1: flow[4,1] <= 1000
 capacity_5_1: flow[5,1] <= 1000
 capacity_6_1: flow[6,1] <= 1000
 capacity_7_1: flow[7,1] <= 1000
 capacity_8_1: flow[8,1] <= 1000
 capacity_9_1: flow[9,1] <= 1000
 capacity_10_1: flow[10,1] <= 1000
 capacity_11_1: flow[11,1] <= 1000
 capacity_12_1: flow[12,1] <= 1000
 capacity_13_1: flow[13,1] <= 1000
 capacity_14_1: flow[14,1] <= 1000
 capacity_15_1: flow[15,1] <= 1000
 capacity_16_1: flow[16,1] <= 1000
 capacity_17_1: flow[17,1] <= 1000
 capacity_0_2: flow[0,2] <= 1000
 capacity_1_2: flow[1,2] <= 1000
 capacity_2_2: flow[2,2] <= 1000
 capacity_3_2: flow[3,2] <= 1000
 capacity_4_2: flow[4,2] <= 1000
 capacity_5_2: flow[5,2] <= 1000
 capacity_6_2: flow[6,2] <= 1000
 capacity_7_2: flow[7,2] <= 1000
 capacity_8_2: flow[8,2] <= 1000
 capacity_9_2: flow[9,2] <= 1000
 capacity_10_2: flow[10,2] <= 1000
 capacity_11_2: flow[11,2] <= 1000
 capacity_12_2: flow[12,2] <= 1000
 capacity_13_2: flow[13,2] <= 1000
 capacity_14_2: flow[14,2] <= 1000
 capacity_15_2: flow[15,2] <= 1000
 capacity_16_2: flow[16,2] <= 1000
 capacity_17_2: flow[17,2] <= 1000
 capacity_0_3: flow[0,3] <= 1000
 capacity_1_3: flow[1,3] <= 1000
 capacity_2_3: flow[2,3] <= 1000
 capacity_3_3: flow[3,3] <= 1000
 capacity_4_3: flow[4,3] <= 1000
 capacity_5_3: flow[5,3] <= 1000
 capacity_6_3: flow[6,3] <= 1000
 capacity_7_3: flow[7,3] <= 1000
 capacity_8_3: flow[8,3] <= 1000
 capacity_9_3: flow[9,3] <= 1000
 capacity_10_3: flow[10,3] <= 1000
 capacity_11_3: flow[11,3] <= 1000
 capacity_12_3: flow[12,3] <= 1000
 capacity_13_3: flow[13,3] <= 1000
 capacity_14_3: flow[14,3] <= 1000
 capacity_15_3: flow[15,3] <= 1000
 capacity_16_3: flow[16,3] <= 1000
 capacity_17_3: flow[17,3] <= 1000
 capacity_0_4: flow[0,4] <= 1000
 capacity_1_4: flow[1,4] <= 1000
 capacity_2_4: flow[2,4] <= 1000
 capacity_3_4: flow[3,4] <= 1000
 capacity_4_4: flow[4,4] <= 1000
 capacity_5_4: flow[5,4] <= 1000
 capacity_6_4: flow[6,4] <= 1000
 capacity_7_4: flow[7,4] <= 1000
 capacity_8_4: flow[8,4] <= 1000
 capacity_9_4: flow[9,4] <= 1000
 capacity_10_4: flow[10,4] <= 1000
 capacity_11_4: flow[11,4] <= 1000
 capacity_12_4: flow[12,4] <= 1000
 capacity_13_4: flow[13,4] <= 1000
 capacity_14_4: flow[14,4] <= 1000
 capacity_15_4: flow[15,4] <= 1000
 capacity_16_4: flow[16,4] <= 1000
 capacity_17_4: flow[17,4] <= 1000
 flow_conservation_0_0: flow[2,0] + flow[16,0] = 0
 flow_conservation_1_0: flow[10,0] - flow[12,0] + flow[13,0] = 0
 flow_conservation_2_0: - flow[3,0] + flow[6,0] + flow[12,0] - flow[13,0]
   - flow[17,0] = 0
 flow_conservation_3_0: - flow[2,0] - flow[7,0] - flow[8,0] - flow[16,0]
   = 0
 flow_conservation_4_0: - flow[5,0] - flow[6,0] + flow[9,0] = 0
 flow_conservation_5_0: - flow[0,0] - flow[4,0] - flow[11,0] - flow[14,0]
   = -10
 flow_conservation_6_0: flow[1,0] + flow[3,0] + flow[7,0] + flow[15,0]
   + flow[17,0] = 0
 flow_conservation_7_0: flow[4,0] + flow[8,0] - flow[9,0] + flow[11,0] = 0
 flow_conservation_8_0: - flow[10,0] = 0
 flow_conservation_9_0: flow[0,0] - flow[1,0] + flow[5,0] + flow[14,0]
   - flow[15,0] = 10
 flow_conservation_0_1: flow[2,1] + flow[16,1] = 0
 flow_conservation_1_1: flow[10,1] - flow[12,1] + flow[13,1] = 0
 flow_conservation_2_1: - flow[3,1] + flow[6,1] + flow[12,1] - flow[13,1]
   - flow[17,1] = 0
 flow_conservation_3_1: - flow[2,1] - flow[7,1] - flow[8,1] - flow[16,1]
   = 0
 flow_conservation_4_1: - flow[5,1] - flow[6,1] + flow[9,1] = 0
 flow_conservation_5_1: - flow[0,1] - flow[4,1] - flow[11,1] - flow[14,1]
   = -10
 flow_conservation_6_1: flow[1,1] + flow[3,1] + flow[7,1] + flow[15,1]
   + flow[17,1] = 0
 flow_conservation_7_1: flow[4,1] + flow[8,1] - flow[9,1] + flow[11,1] = 0
 flow_conservation_8_1: - flow[10,1] = 0
 flow_conservation_9_1: flow[0,1] - flow[1,1] + flow[5,1] + flow[14,1]
   - flow[15,1] = 10
 flow_conservation_0_2: flow[2,2] + flow[16,2] = 0
 flow_conservation_1_2: flow[10,2] - flow[12,2] + flow[13,2] = 0
 flow_conservation_2_2: - flow[3,2] + flow[6,2] + flow[12,2] - flow[13,2]
   - flow[17,2] = 0
 flow_conservation_3_2: - flow[2,2] - flow[7,2] - flow[8,2] - flow[16,2]
   = 0
 flow_conservation_4_2: - flow[5,2] - flow[6,2] + flow[9,2] = 0
 flow_conservation_5_2: - flow[0,2] - flow[4,2] - flow[11,2] - flow[14,2]
   = -10
 flow_conservation_6_2: flow[1,2] + flow[3,2] + flow[7,2] + flow[15,2]
   + flow[17,2] = 0
 flow_conservation_7_2: flow[4,2] + flow[8,2] - flow[9,2] + flow[11,2] = 0
 flow_conservation_8_2: - flow[10,2] = 0
 flow_conservation_9_2: flow[0,2] - flow[1,2] + flow[5,2] + flow[14,2]
   - flow[15,2] = 10
 flow_conservation_0_3: flow[2,3] + flow[16,3] = 0
 flow_conservation_1_3: flow[10,3] - flow[12,3] + flow[13,3] = 0
 flow_conservation_2_3: - flow[3,3] + flow[6,3] + flow[12,3] - flow[13,3]
   - flow[17,3] = 0
 flow_conservation_3_3: - flow[2,3] - flow[7,3] - flow[8,3] - flow[16,3]
   = 0
 flow_conservation_4_3: - flow[5,3] - flow[6,3] + flow[9,3] = 0
 flow_conservation_5_3: - flow[0,3] - flow[4,3] - flow[11,3] - flow[14,3]
   = -10
 flow_conservation_6_3: flow[1,3] + flow[3,3] + flow[7,3] + flow[15,3]
   + flow[17,3] = 0
 flow_conservation_7_3: flow[4,3] + flow[8,3] - flow[9,3] + flow[11,3] = 0
 flow_conservation_8_3: - flow[10,3] = 0
 flow_conservation_9_3: flow[0,3] - flow[1,3] + flow[5,3] + flow[14,3]
   - flow[15,3] = 10
 flow_conservation_0_4: flow[2,4] + flow[16,4] = 0
 flow_conservation_1_4: flow[10,4] - flow[12,4] + flow[13,4] = 0
 flow_conservation_2_4: - flow[3,4] + flow[6,4] + flow[12,4] - flow[13,4]
   - flow[17,4] = 0
 flow_conservation_3_4: - flow[2,4] - flow[7,4] - flow[8,4] - flow[16,4]
   = 0
 flow_conservation_4_4: - flow[5,4] - flow[6,4] + flow[9,4] = 0
 flow_conservation_5_4: - flow[0,4] - flow[4,4] - flow[11,4] - flow[14,4]
   = -10
 flow_conservation_6_4: flow[1,4] + flow[3,4] + flow[7,4] + flow[15,4]
   + flow[17,4] = 0
 flow_conservation_7_4: flow[4,4] + flow[8,4] - flow[9,4] + flow[11,4] = 0
 flow_conservation_8_4: - flow[10,4] = 0
 flow_conservation_9_4: flow[0,4] - flow[1,4] + flow[5,4] + flow[14,4]
   - flow[15,4] = 10
Bounds
End
