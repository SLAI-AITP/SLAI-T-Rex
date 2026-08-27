\ Model workforce_scheduling_variant
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  5.91 Z[0] + 8.32 Z[1] + 2.58 Z[2] + 8.01 Z[3] + 5.18 Z[4] + 7.26 Z[5]
   + 6.69 Z[6] + 8.3 Z[7] + 1.57 Z[8] + 7.99 Z[9] + 5.12 Z[10] + 3.64 Z[11]
   + 1.39 Z[12] + 2.8 Z[13] + 1.38 Z[14] + 9.4 Z[15] + 5.64 Z[16]
   + 9.9 Z[17] + 5.89 Z[18] + 3.28 Z[19] + 7.78 Z[20] + 2.72 Z[21]
   + 4.21 Z[22] + 8.03 Z[23] + 8.79 Z[24] + 3.99 Z[25] + 2.12 Z[26]
   + 4.31 Z[27] + 9.01 Z[28] + 7.69 Z[29] + 9.05 Z[30] + 4.48 Z[31]
   + 9.76 Z[32] + 5.47 Z[33] + 5.48 Z[34] + 9.32 Z[35] + 5.67 Z[36]
   + 8.21 Z[37] + 7.54 Z[38] + 1.71 Z[39] + 6.42 Z[40] + 8.4 Z[41]
   + 5.91 Z[42] + 3.89 Z[43] + 1.72 Z[44] + 6.95 Z[45] + 3.76 Z[46]
   + 6.42 Z[47] + 4.84 Z[48] + 7.21 Z[49] + 67.58 W[0] + 52.12 W[1]
   + 93.5 W[2] + 67.63 W[3] + 99.91 W[4] + 63.73 W[5] + 99 W[6] + 97.4 W[7]
   + 53.75 W[8] + 81.88 W[9] + 68.17 W[10] + 90.05 W[11] + 83.97 W[12]
   + 97.64 W[13] + 57.14 W[14] + 80.38 W[15] + 89.07 W[16] + 51.74 W[17]
   + 53.36 W[18] + 88.93 W[19] + 68.32 W[20] + 69.14 W[21] + 78.36 W[22]
   + 80.25 W[23] + 83.95 W[24] + 97.44 W[25] + 68.6 W[26] + 88.15 W[27]
   + 78.7 W[28] + 76.47 W[29] + 69.9 W[30] + 82.48 W[31] + 62.48 W[32]
   + 55.67 W[33] + 86.78 W[34] + 74.95 W[35] + 69.35 W[36] + 78.08 W[37]
   + 63.09 W[38] + 63.01 W[39] + 72.31 W[40] + 99.82 W[41] + 64.28 W[42]
   + 95.82 W[43] + 74.56 W[44] + 56.13 W[45] + 92.64 W[46] + 72.6 W[47]
   + 94.93 W[48] + 72.26 W[49]
Subject To
 TaskRequirement_0: Z[0] + Z[1] + Z[2] + Z[3] + Z[4] = 13.31
 TaskRequirement_1: Z[5] + Z[6] + Z[7] + Z[8] + Z[9] = 10.86
 TaskRequirement_2: Z[10] + Z[11] + Z[12] + Z[13] + Z[14] = 12.05
 TaskRequirement_3: Z[15] + Z[16] + Z[17] + Z[18] + Z[19] = 14.54
 TaskRequirement_4: Z[20] + Z[21] + Z[22] + Z[23] + Z[24] = 19.72
 TaskRequirement_5: Z[25] + Z[26] + Z[27] + Z[28] + Z[29] = 8.8
 TaskRequirement_6: Z[30] + Z[31] + Z[32] + Z[33] + Z[34] = 5.24
 TaskRequirement_7: Z[35] + Z[36] + Z[37] + Z[38] + Z[39] = 16.83
 TaskRequirement_8: Z[40] + Z[41] + Z[42] + Z[43] + Z[44] = 10.17
 TaskRequirement_9: Z[45] + Z[46] + Z[47] + Z[48] + Z[49] = 15.99
 WorkerCapacity_0: Z[0] + Z[5] + Z[10] + Z[15] + Z[20] + Z[25] + Z[30]
   + Z[35] + Z[40] + Z[45] <= 42.57
 WorkerCapacity_1: Z[1] + Z[6] + Z[11] + Z[16] + Z[21] + Z[26] + Z[31]
   + Z[36] + Z[41] + Z[46] <= 45.43
 WorkerCapacity_2: Z[2] + Z[7] + Z[12] + Z[17] + Z[22] + Z[27] + Z[32]
   + Z[37] + Z[42] + Z[47] <= 44.7
 WorkerCapacity_3: Z[3] + Z[8] + Z[13] + Z[18] + Z[23] + Z[28] + Z[33]
   + Z[38] + Z[43] + Z[48] <= 36.65
 WorkerCapacity_4: Z[4] + Z[9] + Z[14] + Z[19] + Z[24] + Z[29] + Z[34]
   + Z[39] + Z[44] + Z[49] <= 30.89
 LinkingConstraint_0: Z[0] - 5.88 W[0] <= 0
 LinkingConstraint_1: Z[1] - 11.82 W[1] <= 0
 LinkingConstraint_2: Z[2] - 13.46 W[2] <= 0
 LinkingConstraint_3: Z[3] - 8.2 W[3] <= 0
 LinkingConstraint_4: Z[4] - 8.47 W[4] <= 0
 LinkingConstraint_5: Z[5] - 5.65 W[5] <= 0
 LinkingConstraint_6: Z[6] - 10.42 W[6] <= 0
 LinkingConstraint_7: Z[7] - 13.91 W[7] <= 0
 LinkingConstraint_8: Z[8] - 13.51 W[8] <= 0
 LinkingConstraint_9: Z[9] - 12.12 W[9] <= 0
 LinkingConstraint_10: Z[10] - 14.27 W[10] <= 0
 LinkingConstraint_11: Z[11] - 11.38 W[11] <= 0
 LinkingConstraint_12: Z[12] - 12.94 W[12] <= 0
 LinkingConstraint_13: Z[13] - 10.09 W[13] <= 0
 LinkingConstraint_14: Z[14] - 6.21 W[14] <= 0
 LinkingConstraint_15: Z[15] - 7.01 W[15] <= 0
 LinkingConstraint_16: Z[16] - 6.39 W[16] <= 0
 LinkingConstraint_17: Z[17] - 12.9 W[17] <= 0
 LinkingConstraint_18: Z[18] - 5.26 W[18] <= 0
 LinkingConstraint_19: Z[19] - 10.54 W[19] <= 0
 LinkingConstraint_20: Z[20] - 8.69 W[20] <= 0
 LinkingConstraint_21: Z[21] - 13.04 W[21] <= 0
 LinkingConstraint_22: Z[22] - 10.52 W[22] <= 0
 LinkingConstraint_23: Z[23] - 11.12 W[23] <= 0
 LinkingConstraint_24: Z[24] - 5.86 W[24] <= 0
 LinkingConstraint_25: Z[25] - 8.09 W[25] <= 0
 LinkingConstraint_26: Z[26] - 15 W[26] <= 0
 LinkingConstraint_27: Z[27] - 12.19 W[27] <= 0
 LinkingConstraint_28: Z[28] - 10.26 W[28] <= 0
 LinkingConstraint_29: Z[29] - 12.69 W[29] <= 0
 LinkingConstraint_30: Z[30] - 13.23 W[30] <= 0
 LinkingConstraint_31: Z[31] - 5.74 W[31] <= 0
 LinkingConstraint_32: Z[32] - 14.72 W[32] <= 0
 LinkingConstraint_33: Z[33] - 11.42 W[33] <= 0
 LinkingConstraint_34: Z[34] - 9.5 W[34] <= 0
 LinkingConstraint_35: Z[35] - 11.8 W[35] <= 0
 LinkingConstraint_36: Z[36] - 8.45 W[36] <= 0
 LinkingConstraint_37: Z[37] - 13.78 W[37] <= 0
 LinkingConstraint_38: Z[38] - 12.8 W[38] <= 0
 LinkingConstraint_39: Z[39] - 11.4 W[39] <= 0
 LinkingConstraint_40: Z[40] - 6.82 W[40] <= 0
 LinkingConstraint_41: Z[41] - 14.66 W[41] <= 0
 LinkingConstraint_42: Z[42] - 9.33 W[42] <= 0
 LinkingConstraint_43: Z[43] - 14.11 W[43] <= 0
 LinkingConstraint_44: Z[44] - 5.55 W[44] <= 0
 LinkingConstraint_45: Z[45] - 6.24 W[45] <= 0
 LinkingConstraint_46: Z[46] - 6.53 W[46] <= 0
 LinkingConstraint_47: Z[47] - 6.65 W[47] <= 0
 LinkingConstraint_48: Z[48] - 8.23 W[48] <= 0
 LinkingConstraint_49: Z[49] - 12.09 W[49] <= 0
 MaxTasksPerWorker_0: W[0] + W[5] + W[10] + W[15] + W[20] + W[25] + W[30]
   + W[35] + W[40] + W[45] <= 4
 MaxTasksPerWorker_1: W[1] + W[6] + W[11] + W[16] + W[21] + W[26] + W[31]
   + W[36] + W[41] + W[46] <= 4
 MaxTasksPerWorker_2: W[2] + W[7] + W[12] + W[17] + W[22] + W[27] + W[32]
   + W[37] + W[42] + W[47] <= 2
 MaxTasksPerWorker_3: W[3] + W[8] + W[13] + W[18] + W[23] + W[28] + W[33]
   + W[38] + W[43] + W[48] <= 4
 MaxTasksPerWorker_4: W[4] + W[9] + W[14] + W[19] + W[24] + W[29] + W[34]
   + W[39] + W[44] + W[49] <= 3
Bounds
 Z[0] <= 15
 Z[1] <= 15
 Z[2] <= 15
 Z[3] <= 15
 Z[4] <= 15
 Z[5] <= 15
 Z[6] <= 15
 Z[7] <= 15
 Z[8] <= 15
 Z[9] <= 15
 Z[10] <= 15
 Z[11] <= 15
 Z[12] <= 15
 Z[13] <= 15
 Z[14] <= 15
 Z[15] <= 15
 Z[16] <= 15
 Z[17] <= 15
 Z[18] <= 15
 Z[19] <= 15
 Z[20] <= 15
 Z[21] <= 15
 Z[22] <= 15
 Z[23] <= 15
 Z[24] <= 15
 Z[25] <= 15
 Z[26] <= 15
 Z[27] <= 15
 Z[28] <= 15
 Z[29] <= 15
 Z[30] <= 15
 Z[31] <= 15
 Z[32] <= 15
 Z[33] <= 15
 Z[34] <= 15
 Z[35] <= 15
 Z[36] <= 15
 Z[37] <= 15
 Z[38] <= 15
 Z[39] <= 15
 Z[40] <= 15
 Z[41] <= 15
 Z[42] <= 15
 Z[43] <= 15
 Z[44] <= 15
 Z[45] <= 15
 Z[46] <= 15
 Z[47] <= 15
 Z[48] <= 15
 Z[49] <= 15
Binaries
 W[0] W[1] W[2] W[3] W[4] W[5] W[6] W[7] W[8] W[9] W[10] W[11] W[12] W[13]
 W[14] W[15] W[16] W[17] W[18] W[19] W[20] W[21] W[22] W[23] W[24] W[25]
 W[26] W[27] W[28] W[29] W[30] W[31] W[32] W[33] W[34] W[35] W[36] W[37]
 W[38] W[39] W[40] W[41] W[42] W[43] W[44] W[45] W[46] W[47] W[48] W[49]
End
