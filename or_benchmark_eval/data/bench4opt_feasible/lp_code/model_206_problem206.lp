\ Model Staff Scheduling Problem with Overtime Consideration
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  29.32 x[0,0] + 29.32 x[0,1] + 29.32 x[0,2] + 29.32 x[0,3] + 29.32 x[0,4]
   + 29.32 x[0,5] + 26.84 x[1,0] + 26.84 x[1,1] + 26.84 x[1,2]
   + 26.84 x[1,3] + 26.84 x[1,4] + 26.84 x[1,5] + 26.04 x[2,0]
   + 26.04 x[2,1] + 26.04 x[2,2] + 26.04 x[2,3] + 26.04 x[2,4]
   + 26.04 x[2,5] + 29.95 x[3,0] + 29.95 x[3,1] + 29.95 x[3,2]
   + 29.95 x[3,3] + 29.95 x[3,4] + 29.95 x[3,5] + 36.19 x[4,0]
   + 36.19 x[4,1] + 36.19 x[4,2] + 36.19 x[4,3] + 36.19 x[4,4]
   + 36.19 x[4,5] + 35.67 x[5,0] + 35.67 x[5,1] + 35.67 x[5,2]
   + 35.67 x[5,3] + 35.67 x[5,4] + 35.67 x[5,5] + 43.56 x[6,0]
   + 43.56 x[6,1] + 43.56 x[6,2] + 43.56 x[6,3] + 43.56 x[6,4]
   + 43.56 x[6,5] + 37.04 x[7,0] + 37.04 x[7,1] + 37.04 x[7,2]
   + 37.04 x[7,3] + 37.04 x[7,4] + 37.04 x[7,5] + 31.24 x[8,0]
   + 31.24 x[8,1] + 31.24 x[8,2] + 31.24 x[8,3] + 31.24 x[8,4]
   + 31.24 x[8,5] + 38.14 x[9,0] + 38.14 x[9,1] + 38.14 x[9,2]
   + 38.14 x[9,3] + 38.14 x[9,4] + 38.14 x[9,5] + 44.13 x[10,0]
   + 44.13 x[10,1] + 44.13 x[10,2] + 44.13 x[10,3] + 44.13 x[10,4]
   + 44.13 x[10,5] + 41.76 x[11,0] + 41.76 x[11,1] + 41.76 x[11,2]
   + 41.76 x[11,3] + 41.76 x[11,4] + 41.76 x[11,5] + 29.08 o[0]
   + 17.72 o[1] + 46.05 o[2] + 41.87 o[3] + 48.23 o[4] + 49.63 o[5]
   + 18.79 o[6] + 25.63 o[7] + 50.63 o[8] + 58.99 o[9] + 30.98 o[10]
   + 56.52 o[11]
Subject To
 shift_0_requirement: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] + x[5,0]
   + x[6,0] + x[7,0] + x[8,0] + x[9,0] + x[10,0] + x[11,0] >= 1
 shift_1_requirement: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] + x[5,1]
   + x[6,1] + x[7,1] + x[8,1] + x[9,1] + x[10,1] + x[11,1] >= 1
 shift_2_requirement: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] + x[5,2]
   + x[6,2] + x[7,2] + x[8,2] + x[9,2] + x[10,2] + x[11,2] >= 1
 shift_3_requirement: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] + x[5,3]
   + x[6,3] + x[7,3] + x[8,3] + x[9,3] + x[10,3] + x[11,3] >= 1
 shift_4_requirement: x[0,4] + x[1,4] + x[2,4] + x[3,4] + x[4,4] + x[5,4]
   + x[6,4] + x[7,4] + x[8,4] + x[9,4] + x[10,4] + x[11,4] >= 1
 shift_5_requirement: x[0,5] + x[1,5] + x[2,5] + x[3,5] + x[4,5] + x[5,5]
   + x[6,5] + x[7,5] + x[8,5] + x[9,5] + x[10,5] + x[11,5] >= 1
 availability_0_0: x[0,0] <= 100
 availability_0_1: x[0,1] <= 100
 availability_0_2: x[0,2] <= 100
 availability_0_3: x[0,3] <= 100
 availability_0_4: x[0,4] <= 100
 availability_0_5: x[0,5] <= 100
 availability_1_0: x[1,0] <= 100
 availability_1_1: x[1,1] <= 100
 availability_1_2: x[1,2] <= 100
 availability_1_3: x[1,3] <= 100
 availability_1_4: x[1,4] <= 100
 availability_1_5: x[1,5] <= 100
 availability_2_0: x[2,0] <= 100
 availability_2_1: x[2,1] <= 100
 availability_2_2: x[2,2] <= 100
 availability_2_3: x[2,3] <= 100
 availability_2_4: x[2,4] <= 100
 availability_2_5: x[2,5] <= 100
 availability_3_0: x[3,0] <= 100
 availability_3_1: x[3,1] <= 100
 availability_3_2: x[3,2] <= 100
 availability_3_3: x[3,3] <= 100
 availability_3_4: x[3,4] <= 100
 availability_3_5: x[3,5] <= 100
 availability_4_0: x[4,0] <= 100
 availability_4_1: x[4,1] <= 100
 availability_4_2: x[4,2] <= 100
 availability_4_3: x[4,3] <= 100
 availability_4_4: x[4,4] <= 100
 availability_4_5: x[4,5] <= 100
 availability_5_0: x[5,0] <= 100
 availability_5_1: x[5,1] <= 100
 availability_5_2: x[5,2] <= 100
 availability_5_3: x[5,3] <= 100
 availability_5_4: x[5,4] <= 100
 availability_5_5: x[5,5] <= 100
 availability_6_0: x[6,0] <= 100
 availability_6_1: x[6,1] <= 100
 availability_6_2: x[6,2] <= 100
 availability_6_3: x[6,3] <= 100
 availability_6_4: x[6,4] <= 100
 availability_6_5: x[6,5] <= 100
 availability_7_0: x[7,0] <= 100
 availability_7_1: x[7,1] <= 100
 availability_7_2: x[7,2] <= 100
 availability_7_3: x[7,3] <= 100
 availability_7_4: x[7,4] <= 100
 availability_7_5: x[7,5] <= 100
 availability_8_0: x[8,0] <= 100
 availability_8_1: x[8,1] <= 100
 availability_8_2: x[8,2] <= 100
 availability_8_3: x[8,3] <= 100
 availability_8_4: x[8,4] <= 100
 availability_8_5: x[8,5] <= 100
 availability_9_0: x[9,0] <= 100
 availability_9_1: x[9,1] <= 100
 availability_9_2: x[9,2] <= 100
 availability_9_3: x[9,3] <= 100
 availability_9_4: x[9,4] <= 100
 availability_9_5: x[9,5] <= 100
 availability_10_0: x[10,0] <= 100
 availability_10_1: x[10,1] <= 100
 availability_10_2: x[10,2] <= 100
 availability_10_3: x[10,3] <= 100
 availability_10_4: x[10,4] <= 100
 availability_10_5: x[10,5] <= 100
 availability_11_0: x[11,0] <= 100
 availability_11_1: x[11,1] <= 100
 availability_11_2: x[11,2] <= 100
 availability_11_3: x[11,3] <= 100
 availability_11_4: x[11,4] <= 100
 availability_11_5: x[11,5] <= 100
 overtime_0: x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] - o[0]
   <= 24.79
 overtime_1: x[1,0] + x[1,1] + x[1,2] + x[1,3] + x[1,4] + x[1,5] - o[1]
   <= 38.6
 overtime_2: x[2,0] + x[2,1] + x[2,2] + x[2,3] + x[2,4] + x[2,5] - o[2]
   <= 31.16
 overtime_3: x[3,0] + x[3,1] + x[3,2] + x[3,3] + x[3,4] + x[3,5] - o[3]
   <= 27.81
 overtime_4: x[4,0] + x[4,1] + x[4,2] + x[4,3] + x[4,4] + x[4,5] - o[4]
   <= 24.81
 overtime_5: x[5,0] + x[5,1] + x[5,2] + x[5,3] + x[5,4] + x[5,5] - o[5]
   <= 29.5
 overtime_6: x[6,0] + x[6,1] + x[6,2] + x[6,3] + x[6,4] + x[6,5] - o[6]
   <= 29.64
 overtime_7: x[7,0] + x[7,1] + x[7,2] + x[7,3] + x[7,4] + x[7,5] - o[7]
   <= 31.78
 overtime_8: x[8,0] + x[8,1] + x[8,2] + x[8,3] + x[8,4] + x[8,5] - o[8]
   <= 26.96
 overtime_9: x[9,0] + x[9,1] + x[9,2] + x[9,3] + x[9,4] + x[9,5] - o[9]
   <= 30.96
 overtime_10: x[10,0] + x[10,1] + x[10,2] + x[10,3] + x[10,4] + x[10,5]
   - o[10] <= 32.93
 overtime_11: x[11,0] + x[11,1] + x[11,2] + x[11,3] + x[11,4] + x[11,5]
   - o[11] <= 23.25
Bounds
End
