\ Model Network Flow Problem with Time-Dependent Demand
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2 flow[0,0] + 2 flow[0,1] + 2 flow[0,2] + 2 flow[0,3] + 2 flow[0,4]
   + 2 flow[1,0] + 2 flow[1,1] + 2 flow[1,2] + 2 flow[1,3] + 2 flow[1,4]
   + 2 flow[2,0] + 2 flow[2,1] + 2 flow[2,2] + 2 flow[2,3] + 2 flow[2,4]
   + 2 flow[3,0] + 2 flow[3,1] + 2 flow[3,2] + 2 flow[3,3] + 2 flow[3,4]
   + 2 flow[4,0] + 2 flow[4,1] + 2 flow[4,2] + 2 flow[4,3] + 2 flow[4,4]
   + 2 flow[5,0] + 2 flow[5,1] + 2 flow[5,2] + 2 flow[5,3] + 2 flow[5,4]
   + 2 flow[6,0] + 2 flow[6,1] + 2 flow[6,2] + 2 flow[6,3] + 2 flow[6,4]
   + 2 flow[7,0] + 2 flow[7,1] + 2 flow[7,2] + 2 flow[7,3] + 2 flow[7,4]
   + 2 flow[8,0] + 2 flow[8,1] + 2 flow[8,2] + 2 flow[8,3] + 2 flow[8,4]
   + 2 flow[9,0] + 2 flow[9,1] + 2 flow[9,2] + 2 flow[9,3] + 2 flow[9,4]
   + 2 flow[10,0] + 2 flow[10,1] + 2 flow[10,2] + 2 flow[10,3]
   + 2 flow[10,4] + 2 flow[11,0] + 2 flow[11,1] + 2 flow[11,2]
   + 2 flow[11,3] + 2 flow[11,4] + 2 flow[12,0] + 2 flow[12,1]
   + 2 flow[12,2] + 2 flow[12,3] + 2 flow[12,4] + 2 flow[13,0]
   + 2 flow[13,1] + 2 flow[13,2] + 2 flow[13,3] + 2 flow[13,4]
Subject To
 cap_0_t0: flow[0,0] <= 1000
 cap_0_t1: flow[0,1] <= 1000
 cap_0_t2: flow[0,2] <= 1000
 cap_0_t3: flow[0,3] <= 1000
 cap_0_t4: flow[0,4] <= 1000
 cap_1_t0: flow[1,0] <= 1000
 cap_1_t1: flow[1,1] <= 1000
 cap_1_t2: flow[1,2] <= 1000
 cap_1_t3: flow[1,3] <= 1000
 cap_1_t4: flow[1,4] <= 1000
 cap_2_t0: flow[2,0] <= 1000
 cap_2_t1: flow[2,1] <= 1000
 cap_2_t2: flow[2,2] <= 1000
 cap_2_t3: flow[2,3] <= 1000
 cap_2_t4: flow[2,4] <= 1000
 cap_3_t0: flow[3,0] <= 1000
 cap_3_t1: flow[3,1] <= 1000
 cap_3_t2: flow[3,2] <= 1000
 cap_3_t3: flow[3,3] <= 1000
 cap_3_t4: flow[3,4] <= 1000
 cap_4_t0: flow[4,0] <= 1000
 cap_4_t1: flow[4,1] <= 1000
 cap_4_t2: flow[4,2] <= 1000
 cap_4_t3: flow[4,3] <= 1000
 cap_4_t4: flow[4,4] <= 1000
 cap_5_t0: flow[5,0] <= 1000
 cap_5_t1: flow[5,1] <= 1000
 cap_5_t2: flow[5,2] <= 1000
 cap_5_t3: flow[5,3] <= 1000
 cap_5_t4: flow[5,4] <= 1000
 cap_6_t0: flow[6,0] <= 1000
 cap_6_t1: flow[6,1] <= 1000
 cap_6_t2: flow[6,2] <= 1000
 cap_6_t3: flow[6,3] <= 1000
 cap_6_t4: flow[6,4] <= 1000
 cap_7_t0: flow[7,0] <= 1000
 cap_7_t1: flow[7,1] <= 1000
 cap_7_t2: flow[7,2] <= 1000
 cap_7_t3: flow[7,3] <= 1000
 cap_7_t4: flow[7,4] <= 1000
 cap_8_t0: flow[8,0] <= 1000
 cap_8_t1: flow[8,1] <= 1000
 cap_8_t2: flow[8,2] <= 1000
 cap_8_t3: flow[8,3] <= 1000
 cap_8_t4: flow[8,4] <= 1000
 cap_9_t0: flow[9,0] <= 1000
 cap_9_t1: flow[9,1] <= 1000
 cap_9_t2: flow[9,2] <= 1000
 cap_9_t3: flow[9,3] <= 1000
 cap_9_t4: flow[9,4] <= 1000
 cap_10_t0: flow[10,0] <= 1000
 cap_10_t1: flow[10,1] <= 1000
 cap_10_t2: flow[10,2] <= 1000
 cap_10_t3: flow[10,3] <= 1000
 cap_10_t4: flow[10,4] <= 1000
 cap_11_t0: flow[11,0] <= 1000
 cap_11_t1: flow[11,1] <= 1000
 cap_11_t2: flow[11,2] <= 1000
 cap_11_t3: flow[11,3] <= 1000
 cap_11_t4: flow[11,4] <= 1000
 cap_12_t0: flow[12,0] <= 1000
 cap_12_t1: flow[12,1] <= 1000
 cap_12_t2: flow[12,2] <= 1000
 cap_12_t3: flow[12,3] <= 1000
 cap_12_t4: flow[12,4] <= 1000
 cap_13_t0: flow[13,0] <= 1000
 cap_13_t1: flow[13,1] <= 1000
 cap_13_t2: flow[13,2] <= 1000
 cap_13_t3: flow[13,3] <= 1000
 cap_13_t4: flow[13,4] <= 1000
 node_0_t0: flow[0,0] - flow[4,0] - flow[5,0] + flow[7,0] + flow[9,0] = 10
 node_0_t1: flow[0,1] - flow[4,1] - flow[5,1] + flow[7,1] + flow[9,1] = 10
 node_0_t2: flow[0,2] - flow[4,2] - flow[5,2] + flow[7,2] + flow[9,2] = 10
 node_0_t3: flow[0,3] - flow[4,3] - flow[5,3] + flow[7,3] + flow[9,3] = 10
 node_0_t4: flow[0,4] - flow[4,4] - flow[5,4] + flow[7,4] + flow[9,4] = 10
 node_1_t0: - flow[1,0] + flow[3,0] - flow[6,0] - flow[7,0] + flow[12,0]
   = 0
 node_1_t1: - flow[1,1] + flow[3,1] - flow[6,1] - flow[7,1] + flow[12,1]
   = 0
 node_1_t2: - flow[1,2] + flow[3,2] - flow[6,2] - flow[7,2] + flow[12,2]
   = 0
 node_1_t3: - flow[1,3] + flow[3,3] - flow[6,3] - flow[7,3] + flow[12,3]
   = 0
 node_1_t4: - flow[1,4] + flow[3,4] - flow[6,4] - flow[7,4] + flow[12,4]
   = 0
 node_2_t0: - flow[0,0] + flow[4,0] + flow[6,0] + flow[8,0] - flow[10,0]
   = -10
 node_2_t1: - flow[0,1] + flow[4,1] + flow[6,1] + flow[8,1] - flow[10,1]
   = -10
 node_2_t2: - flow[0,2] + flow[4,2] + flow[6,2] + flow[8,2] - flow[10,2]
   = -10
 node_2_t3: - flow[0,3] + flow[4,3] + flow[6,3] + flow[8,3] - flow[10,3]
   = -10
 node_2_t4: - flow[0,4] + flow[4,4] + flow[6,4] + flow[8,4] - flow[10,4]
   = -10
 node_3_t0: flow[1,0] - flow[2,0] - flow[3,0] - flow[9,0] + flow[11,0] = 0
 node_3_t1: flow[1,1] - flow[2,1] - flow[3,1] - flow[9,1] + flow[11,1] = 0
 node_3_t2: flow[1,2] - flow[2,2] - flow[3,2] - flow[9,2] + flow[11,2] = 0
 node_3_t3: flow[1,3] - flow[2,3] - flow[3,3] - flow[9,3] + flow[11,3] = 0
 node_3_t4: flow[1,4] - flow[2,4] - flow[3,4] - flow[9,4] + flow[11,4] = 0
 node_4_t0: flow[2,0] + flow[5,0] - flow[8,0] - flow[11,0] - flow[12,0]
   - flow[13,0] = 0
 node_4_t1: flow[2,1] + flow[5,1] - flow[8,1] - flow[11,1] - flow[12,1]
   - flow[13,1] = 0
 node_4_t2: flow[2,2] + flow[5,2] - flow[8,2] - flow[11,2] - flow[12,2]
   - flow[13,2] = 0
 node_4_t3: flow[2,3] + flow[5,3] - flow[8,3] - flow[11,3] - flow[12,3]
   - flow[13,3] = 0
 node_4_t4: flow[2,4] + flow[5,4] - flow[8,4] - flow[11,4] - flow[12,4]
   - flow[13,4] = 0
 node_5_t0: flow[10,0] + flow[13,0] = 0
 node_5_t1: flow[10,1] + flow[13,1] = 0
 node_5_t2: flow[10,2] + flow[13,2] = 0
 node_5_t3: flow[10,3] + flow[13,3] = 0
 node_5_t4: flow[10,4] + flow[13,4] = 0
Bounds
End
