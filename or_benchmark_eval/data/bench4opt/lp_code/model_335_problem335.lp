\ Model Network Flow Problem with Time-Dependent Capacities
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  74.77 flow[0,0] + 74.77 flow[0,1] + 74.77 flow[0,2] + 74.77 flow[0,3]
   + 74.77 flow[0,4] + 51.75 flow[1,0] + 51.75 flow[1,1] + 51.75 flow[1,2]
   + 51.75 flow[1,3] + 51.75 flow[1,4] + 25.76 flow[2,0] + 25.76 flow[2,1]
   + 25.76 flow[2,2] + 25.76 flow[2,3] + 25.76 flow[2,4] + 96.23 flow[3,0]
   + 96.23 flow[3,1] + 96.23 flow[3,2] + 96.23 flow[3,3] + 96.23 flow[3,4]
   + 79.52 flow[4,0] + 79.52 flow[4,1] + 79.52 flow[4,2] + 79.52 flow[4,3]
   + 79.52 flow[4,4] + 17.4 flow[5,0] + 17.4 flow[5,1] + 17.4 flow[5,2]
   + 17.4 flow[5,3] + 17.4 flow[5,4] + 43.91 flow[6,0] + 43.91 flow[6,1]
   + 43.91 flow[6,2] + 43.91 flow[6,3] + 43.91 flow[6,4] + 60.04 flow[7,0]
   + 60.04 flow[7,1] + 60.04 flow[7,2] + 60.04 flow[7,3] + 60.04 flow[7,4]
   + 62.67 flow[8,0] + 62.67 flow[8,1] + 62.67 flow[8,2] + 62.67 flow[8,3]
   + 62.67 flow[8,4] + 43.5 flow[9,0] + 43.5 flow[9,1] + 43.5 flow[9,2]
   + 43.5 flow[9,3] + 43.5 flow[9,4] + 38.02 flow[10,0] + 38.02 flow[10,1]
   + 38.02 flow[10,2] + 38.02 flow[10,3] + 38.02 flow[10,4]
   + 17.56 flow[11,0] + 17.56 flow[11,1] + 17.56 flow[11,2]
   + 17.56 flow[11,3] + 17.56 flow[11,4] + 26.05 flow[12,0]
   + 26.05 flow[12,1] + 26.05 flow[12,2] + 26.05 flow[12,3]
   + 26.05 flow[12,4] + 60.78 flow[13,0] + 60.78 flow[13,1]
   + 60.78 flow[13,2] + 60.78 flow[13,3] + 60.78 flow[13,4]
   + 40.43 flow[14,0] + 40.43 flow[14,1] + 40.43 flow[14,2]
   + 40.43 flow[14,3] + 40.43 flow[14,4] + 81.1 flow[15,0]
   + 81.1 flow[15,1] + 81.1 flow[15,2] + 81.1 flow[15,3] + 81.1 flow[15,4]
   + 36.42 flow[16,0] + 36.42 flow[16,1] + 36.42 flow[16,2]
   + 36.42 flow[16,3] + 36.42 flow[16,4] + 66.04 flow[17,0]
   + 66.04 flow[17,1] + 66.04 flow[17,2] + 66.04 flow[17,3]
   + 66.04 flow[17,4]
Subject To
 capacity_0_0: flow[0,0] <= 44.7
 capacity_1_0: flow[1,0] <= 64.16
 capacity_2_0: flow[2,0] <= 57.45
 capacity_3_0: flow[3,0] <= 19.26
 capacity_4_0: flow[4,0] <= 35.02
 capacity_5_0: flow[5,0] <= 64.71
 capacity_6_0: flow[6,0] <= 89.5
 capacity_7_0: flow[7,0] <= 87.39
 capacity_8_0: flow[8,0] <= 16.73
 capacity_9_0: flow[9,0] <= 90.34
 capacity_10_0: flow[10,0] <= 47.96
 capacity_11_0: flow[11,0] <= 78.06
 capacity_12_0: flow[12,0] <= 38.6
 capacity_13_0: flow[13,0] <= 38.81
 capacity_14_0: flow[14,0] <= 27.74
 capacity_15_0: flow[15,0] <= 82.25
 capacity_16_0: flow[16,0] <= 54.88
 capacity_17_0: flow[17,0] <= 99.12
 capacity_0_1: flow[0,1] <= 47.91
 capacity_1_1: flow[1,1] <= 21.17
 capacity_2_1: flow[2,1] <= 25.4
 capacity_3_1: flow[3,1] <= 39.42
 capacity_4_1: flow[4,1] <= 44.46
 capacity_5_1: flow[5,1] <= 61.88
 capacity_6_1: flow[6,1] <= 35.65
 capacity_7_1: flow[7,1] <= 67.93
 capacity_8_1: flow[8,1] <= 87.05
 capacity_9_1: flow[9,1] <= 98.92
 capacity_10_1: flow[10,1] <= 92.05
 capacity_11_1: flow[11,1] <= 35.59
 capacity_12_1: flow[12,1] <= 85.75
 capacity_13_1: flow[13,1] <= 46.25
 capacity_14_1: flow[14,1] <= 90.62
 capacity_15_1: flow[15,1] <= 92.62
 capacity_16_1: flow[16,1] <= 13.52
 capacity_17_1: flow[17,1] <= 68.03
 capacity_0_2: flow[0,2] <= 16.25
 capacity_1_2: flow[1,2] <= 18.56
 capacity_2_2: flow[2,2] <= 82.23
 capacity_3_2: flow[3,2] <= 75.09
 capacity_4_2: flow[4,2] <= 34.72
 capacity_5_2: flow[5,2] <= 77.83
 capacity_6_2: flow[6,2] <= 46.03
 capacity_7_2: flow[7,2] <= 60.08
 capacity_8_2: flow[8,2] <= 38.19
 capacity_9_2: flow[9,2] <= 65.11
 capacity_10_2: flow[10,2] <= 45.4
 capacity_11_2: flow[11,2] <= 72.68
 capacity_12_2: flow[12,2] <= 50.91
 capacity_13_2: flow[13,2] <= 86.38
 capacity_14_2: flow[14,2] <= 97.04
 capacity_15_2: flow[15,2] <= 20.93
 capacity_16_2: flow[16,2] <= 43.65
 capacity_17_2: flow[17,2] <= 65.62
 capacity_0_3: flow[0,3] <= 12.64
 capacity_1_3: flow[1,3] <= 14.31
 capacity_2_3: flow[2,3] <= 10.94
 capacity_3_3: flow[3,3] <= 19.46
 capacity_4_3: flow[4,3] <= 39.06
 capacity_5_3: flow[5,3] <= 71.88
 capacity_6_3: flow[6,3] <= 49.21
 capacity_7_3: flow[7,3] <= 25.32
 capacity_8_3: flow[8,3] <= 27.2
 capacity_9_3: flow[9,3] <= 91.2
 capacity_10_3: flow[10,3] <= 43.2
 capacity_11_3: flow[11,3] <= 91.26
 capacity_12_3: flow[12,3] <= 99.06
 capacity_13_3: flow[13,3] <= 94.07
 capacity_14_3: flow[14,3] <= 88.12
 capacity_15_3: flow[15,3] <= 33.37
 capacity_16_3: flow[16,3] <= 80.64
 capacity_17_3: flow[17,3] <= 19.97
 capacity_0_4: flow[0,4] <= 95.24
 capacity_1_4: flow[1,4] <= 16.75
 capacity_2_4: flow[2,4] <= 39.64
 capacity_3_4: flow[3,4] <= 65.15
 capacity_4_4: flow[4,4] <= 83.12
 capacity_5_4: flow[5,4] <= 34.29
 capacity_6_4: flow[6,4] <= 25.33
 capacity_7_4: flow[7,4] <= 22.4
 capacity_8_4: flow[8,4] <= 51.46
 capacity_9_4: flow[9,4] <= 21.37
 capacity_10_4: flow[10,4] <= 87.57
 capacity_11_4: flow[11,4] <= 77.77
 capacity_12_4: flow[12,4] <= 89.9
 capacity_13_4: flow[13,4] <= 96.93
 capacity_14_4: flow[14,4] <= 47.2
 capacity_15_4: flow[15,4] <= 50.21
 capacity_16_4: flow[16,4] <= 21.01
 capacity_17_4: flow[17,4] <= 57.48
 flow_conservation_0_0: - flow[12,0] = -12.8
 flow_conservation_1_0: - flow[4,0] - flow[11,0] - flow[15,0] - flow[17,0]
   = 27.95
 flow_conservation_2_0: flow[15,0] + flow[16,0] = -37.77
 flow_conservation_3_0: flow[2,0] - flow[3,0] - flow[6,0] + flow[17,0]
   = -18.14
 flow_conservation_4_0: - flow[0,0] + flow[3,0] + flow[5,0] - flow[8,0]
   + flow[13,0] = -48.91
 flow_conservation_5_0: flow[8,0] + flow[10,0] + flow[14,0] = -17.11
 flow_conservation_6_0: flow[1,0] - flow[2,0] + flow[7,0] + flow[9,0]
   - flow[13,0] - flow[16,0] = -3.46
 flow_conservation_7_0: flow[0,0] - flow[1,0] + flow[6,0] - flow[7,0]
   + flow[11,0] = 29.52
 flow_conservation_8_0: flow[12,0] - flow[14,0] = 39.87
 flow_conservation_9_0: flow[4,0] - flow[5,0] - flow[9,0] - flow[10,0]
   = 6.16
 flow_conservation_0_1: - flow[12,1] = -12.8
 flow_conservation_1_1: - flow[4,1] - flow[11,1] - flow[15,1] - flow[17,1]
   = 27.95
 flow_conservation_2_1: flow[15,1] + flow[16,1] = -37.77
 flow_conservation_3_1: flow[2,1] - flow[3,1] - flow[6,1] + flow[17,1]
   = -18.14
 flow_conservation_4_1: - flow[0,1] + flow[3,1] + flow[5,1] - flow[8,1]
   + flow[13,1] = -48.91
 flow_conservation_5_1: flow[8,1] + flow[10,1] + flow[14,1] = -17.11
 flow_conservation_6_1: flow[1,1] - flow[2,1] + flow[7,1] + flow[9,1]
   - flow[13,1] - flow[16,1] = -3.46
 flow_conservation_7_1: flow[0,1] - flow[1,1] + flow[6,1] - flow[7,1]
   + flow[11,1] = 29.52
 flow_conservation_8_1: flow[12,1] - flow[14,1] = 39.87
 flow_conservation_9_1: flow[4,1] - flow[5,1] - flow[9,1] - flow[10,1]
   = 6.16
 flow_conservation_0_2: - flow[12,2] = -12.8
 flow_conservation_1_2: - flow[4,2] - flow[11,2] - flow[15,2] - flow[17,2]
   = 27.95
 flow_conservation_2_2: flow[15,2] + flow[16,2] = -37.77
 flow_conservation_3_2: flow[2,2] - flow[3,2] - flow[6,2] + flow[17,2]
   = -18.14
 flow_conservation_4_2: - flow[0,2] + flow[3,2] + flow[5,2] - flow[8,2]
   + flow[13,2] = -48.91
 flow_conservation_5_2: flow[8,2] + flow[10,2] + flow[14,2] = -17.11
 flow_conservation_6_2: flow[1,2] - flow[2,2] + flow[7,2] + flow[9,2]
   - flow[13,2] - flow[16,2] = -3.46
 flow_conservation_7_2: flow[0,2] - flow[1,2] + flow[6,2] - flow[7,2]
   + flow[11,2] = 29.52
 flow_conservation_8_2: flow[12,2] - flow[14,2] = 39.87
 flow_conservation_9_2: flow[4,2] - flow[5,2] - flow[9,2] - flow[10,2]
   = 6.16
 flow_conservation_0_3: - flow[12,3] = -12.8
 flow_conservation_1_3: - flow[4,3] - flow[11,3] - flow[15,3] - flow[17,3]
   = 27.95
 flow_conservation_2_3: flow[15,3] + flow[16,3] = -37.77
 flow_conservation_3_3: flow[2,3] - flow[3,3] - flow[6,3] + flow[17,3]
   = -18.14
 flow_conservation_4_3: - flow[0,3] + flow[3,3] + flow[5,3] - flow[8,3]
   + flow[13,3] = -48.91
 flow_conservation_5_3: flow[8,3] + flow[10,3] + flow[14,3] = -17.11
 flow_conservation_6_3: flow[1,3] - flow[2,3] + flow[7,3] + flow[9,3]
   - flow[13,3] - flow[16,3] = -3.46
 flow_conservation_7_3: flow[0,3] - flow[1,3] + flow[6,3] - flow[7,3]
   + flow[11,3] = 29.52
 flow_conservation_8_3: flow[12,3] - flow[14,3] = 39.87
 flow_conservation_9_3: flow[4,3] - flow[5,3] - flow[9,3] - flow[10,3]
   = 6.16
 flow_conservation_0_4: - flow[12,4] = -12.8
 flow_conservation_1_4: - flow[4,4] - flow[11,4] - flow[15,4] - flow[17,4]
   = 27.95
 flow_conservation_2_4: flow[15,4] + flow[16,4] = -37.77
 flow_conservation_3_4: flow[2,4] - flow[3,4] - flow[6,4] + flow[17,4]
   = -18.14
 flow_conservation_4_4: - flow[0,4] + flow[3,4] + flow[5,4] - flow[8,4]
   + flow[13,4] = -48.91
 flow_conservation_5_4: flow[8,4] + flow[10,4] + flow[14,4] = -17.11
 flow_conservation_6_4: flow[1,4] - flow[2,4] + flow[7,4] + flow[9,4]
   - flow[13,4] - flow[16,4] = -3.46
 flow_conservation_7_4: flow[0,4] - flow[1,4] + flow[6,4] - flow[7,4]
   + flow[11,4] = 29.52
 flow_conservation_8_4: flow[12,4] - flow[14,4] = 39.87
 flow_conservation_9_4: flow[4,4] - flow[5,4] - flow[9,4] - flow[10,4]
   = 6.16
Bounds
End
