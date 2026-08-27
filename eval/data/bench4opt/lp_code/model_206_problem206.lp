\ Model Staff Scheduling Problem with Overtime Consideration
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  49.78 x[0,0] + 49.78 x[0,1] + 49.78 x[0,2] + 49.78 x[0,3] + 49.78 x[0,4]
   + 49.78 x[0,5] + 28.14 x[1,0] + 28.14 x[1,1] + 28.14 x[1,2]
   + 28.14 x[1,3] + 28.14 x[1,4] + 28.14 x[1,5] + 38 x[2,0] + 38 x[2,1]
   + 38 x[2,2] + 38 x[2,3] + 38 x[2,4] + 38 x[2,5] + 26.11 x[3,0]
   + 26.11 x[3,1] + 26.11 x[3,2] + 26.11 x[3,3] + 26.11 x[3,4]
   + 26.11 x[3,5] + 47.53 x[4,0] + 47.53 x[4,1] + 47.53 x[4,2]
   + 47.53 x[4,3] + 47.53 x[4,4] + 47.53 x[4,5] + 19.44 x[5,0]
   + 19.44 x[5,1] + 19.44 x[5,2] + 19.44 x[5,3] + 19.44 x[5,4]
   + 19.44 x[5,5] + 26.2 x[6,0] + 26.2 x[6,1] + 26.2 x[6,2] + 26.2 x[6,3]
   + 26.2 x[6,4] + 26.2 x[6,5] + 41.2 x[7,0] + 41.2 x[7,1] + 41.2 x[7,2]
   + 41.2 x[7,3] + 41.2 x[7,4] + 41.2 x[7,5] + 15.86 x[8,0] + 15.86 x[8,1]
   + 15.86 x[8,2] + 15.86 x[8,3] + 15.86 x[8,4] + 15.86 x[8,5]
   + 41.7 x[9,0] + 41.7 x[9,1] + 41.7 x[9,2] + 41.7 x[9,3] + 41.7 x[9,4]
   + 41.7 x[9,5] + 24.6 x[10,0] + 24.6 x[10,1] + 24.6 x[10,2]
   + 24.6 x[10,3] + 24.6 x[10,4] + 24.6 x[10,5] + 10.09 x[11,0]
   + 10.09 x[11,1] + 10.09 x[11,2] + 10.09 x[11,3] + 10.09 x[11,4]
   + 10.09 x[11,5] + 59.5 o[0] + 57.17 o[1] + 45.9 o[2] + 33.22 o[3]
   + 21.77 o[4] + 18.28 o[5] + 44.31 o[6] + 57.17 o[7] + 18.27 o[8]
   + 40.68 o[9] + 29.73 o[10] + 32.91 o[11]
Subject To
 shift_0_requirement: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] + x[5,0]
   + x[6,0] + x[7,0] + x[8,0] + x[9,0] + x[10,0] + x[11,0] >= 7.1
 shift_1_requirement: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] + x[5,1]
   + x[6,1] + x[7,1] + x[8,1] + x[9,1] + x[10,1] + x[11,1] >= 1.71
 shift_2_requirement: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] + x[5,2]
   + x[6,2] + x[7,2] + x[8,2] + x[9,2] + x[10,2] + x[11,2] >= 2.47
 shift_3_requirement: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] + x[5,3]
   + x[6,3] + x[7,3] + x[8,3] + x[9,3] + x[10,3] + x[11,3] >= 4.42
 shift_4_requirement: x[0,4] + x[1,4] + x[2,4] + x[3,4] + x[4,4] + x[5,4]
   + x[6,4] + x[7,4] + x[8,4] + x[9,4] + x[10,4] + x[11,4] >= 2.97
 shift_5_requirement: x[0,5] + x[1,5] + x[2,5] + x[3,5] + x[4,5] + x[5,5]
   + x[6,5] + x[7,5] + x[8,5] + x[9,5] + x[10,5] + x[11,5] >= 4.8
 availability_0_0: x[0,0] <= 0.83
 availability_0_1: x[0,1] <= 0.98
 availability_0_2: x[0,2] <= 0.32
 availability_0_3: x[0,3] <= 0.47
 availability_0_4: x[0,4] <= 0.55
 availability_0_5: x[0,5] <= 0.98
 availability_1_0: x[1,0] <= 0.38
 availability_1_1: x[1,1] <= 0.19
 availability_1_2: x[1,2] <= 0.45
 availability_1_3: x[1,3] <= 0.89
 availability_1_4: x[1,4] <= 0.55
 availability_1_5: x[1,5] <= 0.31
 availability_2_0: x[2,0] <= 0.59
 availability_2_1: x[2,1] <= 0.73
 availability_2_2: x[2,2] <= 0.17
 availability_2_3: x[2,3] <= 0.58
 availability_2_4: x[2,4] <= 0.05
 availability_2_5: x[2,5] <= 0.86
 availability_3_0: x[3,0] <= 0.29
 availability_3_1: x[3,1] <= 0.04
 availability_3_2: x[3,2] <= 0.82
 availability_3_3: x[3,3] <= 0.43
 availability_3_4: x[3,4] <= 0.28
 availability_3_5: x[3,5] <= 0.05
 availability_4_0: x[4,0] <= 0.45
 availability_4_1: x[4,1] <= 0.68
 availability_4_2: x[4,2] <= 0.83
 availability_4_3: x[4,3] <= 0.4
 availability_4_4: x[4,4] <= 0.53
 availability_4_5: x[4,5] <= 0.93
 availability_5_0: x[5,0] <= 0.91
 availability_5_1: x[5,1] <= 0.74
 availability_5_2: x[5,2] <= 0.5
 availability_5_3: x[5,3] <= 0.51
 availability_5_4: x[5,4] <= 0.92
 availability_5_5: x[5,5] <= 0.08
 availability_6_0: x[6,0] <= 0.83
 availability_6_1: x[6,1] <= 0.64
 availability_6_2: x[6,2] <= 0.88
 availability_6_3: x[6,3] <= 0.89
 availability_6_4: x[6,4] <= 0.59
 availability_6_5: x[6,5] <= 0.83
 availability_7_0: x[7,0] <= 0.1
 availability_7_1: x[7,1] <= 0.19
 availability_7_2: x[7,2] <= 0.52
 availability_7_3: x[7,3] <= 0.97
 availability_7_4: x[7,4] <= 0.17
 availability_7_5: x[7,5] <= 0.55
 availability_8_0: x[8,0] <= 0.73
 availability_8_1: x[8,1] <= 0.49
 availability_8_2: x[8,2] <= 0.73
 availability_8_3: x[8,3] <= 0.61
 availability_8_4: x[8,4] <= 0.23
 availability_8_5: x[8,5] <= 0.43
 availability_9_0: x[9,0] <= 0.83
 availability_9_1: x[9,1] <= 0.87
 availability_9_2: x[9,2] <= 0.54
 availability_9_3: x[9,3] <= 0.88
 availability_9_4: x[9,4] <= 0.85
 availability_9_5: x[9,5] <= 0.18
 availability_10_0: x[10,0] <= 0.68
 availability_10_1: x[10,1] <= 0.32
 availability_10_2: x[10,2] <= 0.03
 availability_10_3: x[10,3] <= 0.23
 availability_10_4: x[10,4] <= 0.75
 availability_10_5: x[10,5] <= 0.96
 availability_11_0: x[11,0] <= 0.02
 availability_11_1: x[11,1] <= 0.79
 availability_11_2: x[11,2] <= 0.59
 availability_11_3: x[11,3] <= 0.72
 availability_11_4: x[11,4] <= 0.43
 availability_11_5: x[11,5] <= 0.68
 overtime_0: x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] - o[0]
   <= 30.81
 overtime_1: x[1,0] + x[1,1] + x[1,2] + x[1,3] + x[1,4] + x[1,5] - o[1]
   <= 21.69
 overtime_2: x[2,0] + x[2,1] + x[2,2] + x[2,3] + x[2,4] + x[2,5] - o[2]
   <= 36.81
 overtime_3: x[3,0] + x[3,1] + x[3,2] + x[3,3] + x[3,4] + x[3,5] - o[3]
   <= 23.61
 overtime_4: x[4,0] + x[4,1] + x[4,2] + x[4,3] + x[4,4] + x[4,5] - o[4]
   <= 32.58
 overtime_5: x[5,0] + x[5,1] + x[5,2] + x[5,3] + x[5,4] + x[5,5] - o[5]
   <= 26.56
 overtime_6: x[6,0] + x[6,1] + x[6,2] + x[6,3] + x[6,4] + x[6,5] - o[6]
   <= 23.18
 overtime_7: x[7,0] + x[7,1] + x[7,2] + x[7,3] + x[7,4] + x[7,5] - o[7]
   <= 39.84
 overtime_8: x[8,0] + x[8,1] + x[8,2] + x[8,3] + x[8,4] + x[8,5] - o[8]
   <= 33.57
 overtime_9: x[9,0] + x[9,1] + x[9,2] + x[9,3] + x[9,4] + x[9,5] - o[9]
   <= 33.19
 overtime_10: x[10,0] + x[10,1] + x[10,2] + x[10,3] + x[10,4] + x[10,5]
   - o[10] <= 20.3
 overtime_11: x[11,0] + x[11,1] + x[11,2] + x[11,3] + x[11,4] + x[11,5]
   - o[11] <= 26.7
Bounds
End
