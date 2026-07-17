\ Model Network Flow Problem with Time-Dependent Costs
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  5.88 flow[0,0] + 5.71 flow[0,1] + 7.89 flow[0,2] + 2.02 flow[0,3]
   + 2.59 flow[1,0] + 6.35 flow[1,1] + 5.88 flow[1,2] + 4.43 flow[1,3]
   + 9.44 flow[2,0] + 9.15 flow[2,1] + 1.89 flow[2,2] + 1.79 flow[2,3]
   + 4.7 flow[3,0] + 5.47 flow[3,1] + 7.77 flow[3,2] + 9.35 flow[3,3]
   + 8.25 flow[4,0] + 7.01 flow[4,1] + 8.67 flow[4,2] + 2.91 flow[4,3]
   + 8.71 flow[5,0] + 9.55 flow[5,1] + 4.16 flow[5,2] + 8.14 flow[5,3]
   + 6.8 flow[6,0] + 8.06 flow[6,1] + 4.56 flow[6,2] + 9.46 flow[6,3]
   + 6.93 flow[7,0] + 3.77 flow[7,1] + 9.12 flow[7,2] + 9.18 flow[7,3]
   + 4.73 flow[8,0] + 1.14 flow[8,1] + 2.38 flow[8,2] + 4.37 flow[8,3]
   + 9.94 flow[9,0] + 3.18 flow[9,1] + 9.76 flow[9,2] + 2.73 flow[9,3]
   + 6.2 flow[10,0] + 2.16 flow[10,1] + 6.73 flow[10,2] + 9.8 flow[10,3]
   + 5.66 flow[11,0] + 4.94 flow[11,1] + 1.9 flow[11,2] + 8.21 flow[11,3]
   + 6.65 flow[12,0] + 3.34 flow[12,1] + 5.98 flow[12,2] + 9.54 flow[12,3]
   + 4.94 flow[13,0] + 6.66 flow[13,1] + 6.34 flow[13,2] + 1.25 flow[13,3]
   + 7.61 flow[14,0] + 9.85 flow[14,1] + 6.71 flow[14,2] + 8 flow[14,3]
   + 3 flow[15,0] + 2.74 flow[15,1] + 8.56 flow[15,2] + 7.54 flow[15,3]
   + 3.47 flow[16,0] + 9.41 flow[16,1] + 6.65 flow[16,2] + 6.75 flow[16,3]
   + 2.1 flow[17,0] + 2.28 flow[17,1] + 3.36 flow[17,2] + 2.63 flow[17,3]
   + 9.21 flow[18,0] + 2.87 flow[18,1] + 1.91 flow[18,2] + 8.16 flow[18,3]
   + 6.63 flow[19,0] + 2.89 flow[19,1] + 4.78 flow[19,2] + 9.01 flow[19,3]
Subject To
 capacity_0_0: flow[0,0] <= 416.99
 capacity_0_1: flow[0,1] <= 416.99
 capacity_0_2: flow[0,2] <= 416.99
 capacity_0_3: flow[0,3] <= 416.99
 capacity_1_0: flow[1,0] <= 485.55
 capacity_1_1: flow[1,1] <= 485.55
 capacity_1_2: flow[1,2] <= 485.55
 capacity_1_3: flow[1,3] <= 485.55
 capacity_2_0: flow[2,0] <= 731.88
 capacity_2_1: flow[2,1] <= 731.88
 capacity_2_2: flow[2,2] <= 731.88
 capacity_2_3: flow[2,3] <= 731.88
 capacity_3_0: flow[3,0] <= 654.63
 capacity_3_1: flow[3,1] <= 654.63
 capacity_3_2: flow[3,2] <= 654.63
 capacity_3_3: flow[3,3] <= 654.63
 capacity_4_0: flow[4,0] <= 553.03
 capacity_4_1: flow[4,1] <= 553.03
 capacity_4_2: flow[4,2] <= 553.03
 capacity_4_3: flow[4,3] <= 553.03
 capacity_5_0: flow[5,0] <= 203.05
 capacity_5_1: flow[5,1] <= 203.05
 capacity_5_2: flow[5,2] <= 203.05
 capacity_5_3: flow[5,3] <= 203.05
 capacity_6_0: flow[6,0] <= 917.19
 capacity_6_1: flow[6,1] <= 917.19
 capacity_6_2: flow[6,2] <= 917.19
 capacity_6_3: flow[6,3] <= 917.19
 capacity_7_0: flow[7,0] <= 340.85
 capacity_7_1: flow[7,1] <= 340.85
 capacity_7_2: flow[7,2] <= 340.85
 capacity_7_3: flow[7,3] <= 340.85
 capacity_8_0: flow[8,0] <= 910.17
 capacity_8_1: flow[8,1] <= 910.17
 capacity_8_2: flow[8,2] <= 910.17
 capacity_8_3: flow[8,3] <= 910.17
 capacity_9_0: flow[9,0] <= 963.24
 capacity_9_1: flow[9,1] <= 963.24
 capacity_9_2: flow[9,2] <= 963.24
 capacity_9_3: flow[9,3] <= 963.24
 capacity_10_0: flow[10,0] <= 821.2
 capacity_10_1: flow[10,1] <= 821.2
 capacity_10_2: flow[10,2] <= 821.2
 capacity_10_3: flow[10,3] <= 821.2
 capacity_11_0: flow[11,0] <= 913.75
 capacity_11_1: flow[11,1] <= 913.75
 capacity_11_2: flow[11,2] <= 913.75
 capacity_11_3: flow[11,3] <= 913.75
 capacity_12_0: flow[12,0] <= 858.73
 capacity_12_1: flow[12,1] <= 858.73
 capacity_12_2: flow[12,2] <= 858.73
 capacity_12_3: flow[12,3] <= 858.73
 capacity_13_0: flow[13,0] <= 183.94
 capacity_13_1: flow[13,1] <= 183.94
 capacity_13_2: flow[13,2] <= 183.94
 capacity_13_3: flow[13,3] <= 183.94
 capacity_14_0: flow[14,0] <= 122.76
 capacity_14_1: flow[14,1] <= 122.76
 capacity_14_2: flow[14,2] <= 122.76
 capacity_14_3: flow[14,3] <= 122.76
 capacity_15_0: flow[15,0] <= 687.27
 capacity_15_1: flow[15,1] <= 687.27
 capacity_15_2: flow[15,2] <= 687.27
 capacity_15_3: flow[15,3] <= 687.27
 capacity_16_0: flow[16,0] <= 693.33
 capacity_16_1: flow[16,1] <= 693.33
 capacity_16_2: flow[16,2] <= 693.33
 capacity_16_3: flow[16,3] <= 693.33
 capacity_17_0: flow[17,0] <= 934.48
 capacity_17_1: flow[17,1] <= 934.48
 capacity_17_2: flow[17,2] <= 934.48
 capacity_17_3: flow[17,3] <= 934.48
 capacity_18_0: flow[18,0] <= 860.16
 capacity_18_1: flow[18,1] <= 860.16
 capacity_18_2: flow[18,2] <= 860.16
 capacity_18_3: flow[18,3] <= 860.16
 capacity_19_0: flow[19,0] <= 537.87
 capacity_19_1: flow[19,1] <= 537.87
 capacity_19_2: flow[19,2] <= 537.87
 capacity_19_3: flow[19,3] <= 537.87
 flow_conservation_0_0: flow[3,0] + flow[8,0] + flow[10,0] = 13.36
 flow_conservation_0_1: flow[3,1] + flow[8,1] + flow[10,1] = -4.91
 flow_conservation_0_2: flow[3,2] + flow[8,2] + flow[10,2] = -33.89
 flow_conservation_0_3: flow[3,3] + flow[8,3] + flow[10,3] = -20.12
 flow_conservation_1_0: - flow[3,0] - flow[4,0] - flow[12,0] + flow[13,0]
   - flow[14,0] - flow[17,0] - flow[18,0] + flow[19,0] = 40.36
 flow_conservation_1_1: - flow[3,1] - flow[4,1] - flow[12,1] + flow[13,1]
   - flow[14,1] - flow[17,1] - flow[18,1] + flow[19,1] = 15.97
 flow_conservation_1_2: - flow[3,2] - flow[4,2] - flow[12,2] + flow[13,2]
   - flow[14,2] - flow[17,2] - flow[18,2] + flow[19,2] = 13.12
 flow_conservation_1_3: - flow[3,3] - flow[4,3] - flow[12,3] + flow[13,3]
   - flow[14,3] - flow[17,3] - flow[18,3] + flow[19,3] = 35.48
 flow_conservation_2_0: - flow[1,0] + flow[9,0] - flow[11,0] - flow[15,0]
   = -38.28
 flow_conservation_2_1: - flow[1,1] + flow[9,1] - flow[11,1] - flow[15,1]
   = -15.72
 flow_conservation_2_2: - flow[1,2] + flow[9,2] - flow[11,2] - flow[15,2]
   = -15.32
 flow_conservation_2_3: - flow[1,3] + flow[9,3] - flow[11,3] - flow[15,3]
   = 49.49
 flow_conservation_3_0: - flow[7,0] - flow[8,0] - flow[13,0] + flow[15,0]
   = 10
 flow_conservation_3_1: - flow[7,1] - flow[8,1] - flow[13,1] + flow[15,1]
   = -49.5
 flow_conservation_3_2: - flow[7,2] - flow[8,2] - flow[13,2] + flow[15,2]
   = 2.17
 flow_conservation_3_3: - flow[7,3] - flow[8,3] - flow[13,3] + flow[15,3]
   = -24.55
 flow_conservation_4_0: flow[2,0] + flow[7,0] + flow[16,0] = -37.84
 flow_conservation_4_1: flow[2,1] + flow[7,1] + flow[16,1] = 10.61
 flow_conservation_4_2: flow[2,2] + flow[7,2] + flow[16,2] = 3.6
 flow_conservation_4_3: flow[2,3] + flow[7,3] + flow[16,3] = 1.8
 flow_conservation_5_0: flow[0,0] - flow[9,0] = 26.1
 flow_conservation_5_1: flow[0,1] - flow[9,1] = -28.37
 flow_conservation_5_2: flow[0,2] - flow[9,2] = -13.51
 flow_conservation_5_3: flow[0,3] - flow[9,3] = 39.86
 flow_conservation_6_0: - flow[0,0] - flow[2,0] + flow[5,0] + flow[6,0]
   - flow[10,0] + flow[11,0] + flow[12,0] + flow[14,0] - flow[16,0]
   - flow[19,0] = -47.05
 flow_conservation_6_1: - flow[0,1] - flow[2,1] + flow[5,1] + flow[6,1]
   - flow[10,1] + flow[11,1] + flow[12,1] + flow[14,1] - flow[16,1]
   - flow[19,1] = -36.14
 flow_conservation_6_2: - flow[0,2] - flow[2,2] + flow[5,2] + flow[6,2]
   - flow[10,2] + flow[11,2] + flow[12,2] + flow[14,2] - flow[16,2]
   - flow[19,2] = -41.66
 flow_conservation_6_3: - flow[0,3] - flow[2,3] + flow[5,3] + flow[6,3]
   - flow[10,3] + flow[11,3] + flow[12,3] + flow[14,3] - flow[16,3]
   - flow[19,3] = 19.16
 flow_conservation_7_0: flow[1,0] + flow[4,0] - flow[5,0] - flow[6,0]
   + flow[17,0] + flow[18,0] = -37.34
 flow_conservation_7_1: flow[1,1] + flow[4,1] - flow[5,1] - flow[6,1]
   + flow[17,1] + flow[18,1] = -22.63
 flow_conservation_7_2: flow[1,2] + flow[4,2] - flow[5,2] - flow[6,2]
   + flow[17,2] + flow[18,2] = -30.08
 flow_conservation_7_3: flow[1,3] + flow[4,3] - flow[5,3] - flow[6,3]
   + flow[17,3] + flow[18,3] = -40.8
Bounds
End
