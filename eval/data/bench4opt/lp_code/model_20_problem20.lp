\ Model workforce_scheduling_variant
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  4.75 W[0] + 6.75 W[1] + 5.48 W[2] + 6.64 W[3] + 3.61 W[4] + 9.61 W[5]
   + 5.35 W[6] + 8.24 W[7] + 7.16 W[8] + 3.68 W[9] + 1.66 W[10]
   + 1.54 W[11] + 4.96 W[12] + 5.36 W[13] + 2.84 W[14] + 6.46 W[15]
   + 3.81 W[16] + 7.47 W[17] + 7.61 W[18] + 8.75 W[19] + 9.78 W[20]
   + 2.18 W[21] + 4.33 W[22] + 6.05 W[23] + 3.87 W[24] + 5.2 W[25]
   + 3.41 W[26] + 3.23 W[27] + 1.87 W[28] + 3.61 W[29] + 4.46 W[30]
   + 6.54 W[31] + 3.23 W[32] + 8.79 W[33] + 2.44 W[34] + 3.95 W[35]
   + 6.2 W[36] + 3.81 W[37] + 7.87 W[38] + 5.48 W[39] + 5.63 W[40]
   + 5.49 W[41] + 3.78 W[42] + 1.21 W[43] + 9.51 W[44] + 5.55 W[45]
   + 9.7 W[46] + 2.94 W[47] + 4.18 W[48] + 1.45 W[49] + 5.45 W[50]
   + 8.94 W[51] + 6.89 W[52] + 5.24 W[53] + 5.83 W[54] + 8.62 W[55]
   + 4.88 W[56] + 8.94 W[57] + 7.55 W[58] + 7.87 W[59] + 4.29 W[60]
   + 4.61 W[61] + 6.13 W[62] + 2.75 W[63] + 5.98 W[64] + 1.66 W[65]
   + 5.54 W[66] + 7.88 W[67] + 3.52 W[68] + 9.9 W[69] + 7.12 W[70]
   + 2.07 W[71] + 9.78 W[72] + 4.55 W[73] + 8.15 W[74] + 4.05 W[75]
   + 9.45 W[76] + 7.79 W[77] + 2.79 W[78] + 5.58 W[79] + 125.01 Z[0]
   + 56.8 Z[1] + 70.56 Z[2] + 99.96 Z[3] + 121.06 Z[4] + 118.55 Z[5]
   + 140.94 Z[6] + 127.33 Z[7] + 99.19 Z[8] + 141.96 Z[9] + 74.38 Z[10]
   + 198.59 Z[11] + 160.9 Z[12] + 94.89 Z[13] + 100.46 Z[14] + 174.24 Z[15]
   + 129.85 Z[16] + 156.31 Z[17] + 94.97 Z[18] + 172.36 Z[19]
   + 105.25 Z[20] + 151.07 Z[21] + 196.98 Z[22] + 137.56 Z[23]
   + 169.51 Z[24] + 158.8 Z[25] + 153.21 Z[26] + 54 Z[27] + 121.19 Z[28]
   + 195.06 Z[29] + 167.44 Z[30] + 166.42 Z[31] + 136.65 Z[32]
   + 158.21 Z[33] + 137.53 Z[34] + 75.58 Z[35] + 144.35 Z[36]
   + 142.96 Z[37] + 176.18 Z[38] + 72.17 Z[39] + 152.11 Z[40] + 54.74 Z[41]
   + 192.23 Z[42] + 66.48 Z[43] + 52.84 Z[44] + 97.05 Z[45] + 72.71 Z[46]
   + 153.58 Z[47] + 111.56 Z[48] + 166.25 Z[49] + 188.08 Z[50]
   + 180.92 Z[51] + 160.38 Z[52] + 59.34 Z[53] + 70.71 Z[54] + 81.1 Z[55]
   + 98.76 Z[56] + 149.33 Z[57] + 128.82 Z[58] + 97.06 Z[59] + 75.98 Z[60]
   + 186.82 Z[61] + 101.35 Z[62] + 103.14 Z[63] + 165.8 Z[64]
   + 158.14 Z[65] + 146.5 Z[66] + 154 Z[67] + 141.51 Z[68] + 78.84 Z[69]
   + 86.98 Z[70] + 133.71 Z[71] + 83.73 Z[72] + 195.94 Z[73] + 94.64 Z[74]
   + 93.35 Z[75] + 81.09 Z[76] + 155.75 Z[77] + 97.56 Z[78] + 102.32 Z[79]
Subject To
 TaskRequirement0: W[0] + W[1] + W[2] + W[3] + W[4] + W[5] + W[6] + W[7]
   = 9.54
 TaskRequirement1: W[8] + W[9] + W[10] + W[11] + W[12] + W[13] + W[14]
   + W[15] = 13
 TaskRequirement2: W[16] + W[17] + W[18] + W[19] + W[20] + W[21] + W[22]
   + W[23] = 12.54
 TaskRequirement3: W[24] + W[25] + W[26] + W[27] + W[28] + W[29] + W[30]
   + W[31] = 7.53
 TaskRequirement4: W[32] + W[33] + W[34] + W[35] + W[36] + W[37] + W[38]
   + W[39] = 19.12
 TaskRequirement5: W[40] + W[41] + W[42] + W[43] + W[44] + W[45] + W[46]
   + W[47] = 7.31
 TaskRequirement6: W[48] + W[49] + W[50] + W[51] + W[52] + W[53] + W[54]
   + W[55] = 14.88
 TaskRequirement7: W[56] + W[57] + W[58] + W[59] + W[60] + W[61] + W[62]
   + W[63] = 15.81
 TaskRequirement8: W[64] + W[65] + W[66] + W[67] + W[68] + W[69] + W[70]
   + W[71] = 14.08
 TaskRequirement9: W[72] + W[73] + W[74] + W[75] + W[76] + W[77] + W[78]
   + W[79] = 17.64
 WorkerAvailability0: W[0] + W[8] + W[16] + W[24] + W[32] + W[40] + W[48]
   + W[56] + W[64] + W[72] <= 26.91
 WorkerAvailability1: W[1] + W[9] + W[17] + W[25] + W[33] + W[41] + W[49]
   + W[57] + W[65] + W[73] <= 34.76
 WorkerAvailability2: W[2] + W[10] + W[18] + W[26] + W[34] + W[42] + W[50]
   + W[58] + W[66] + W[74] <= 10.85
 WorkerAvailability3: W[3] + W[11] + W[19] + W[27] + W[35] + W[43] + W[51]
   + W[59] + W[67] + W[75] <= 11.36
 WorkerAvailability4: W[4] + W[12] + W[20] + W[28] + W[36] + W[44] + W[52]
   + W[60] + W[68] + W[76] <= 29.24
 WorkerAvailability5: W[5] + W[13] + W[21] + W[29] + W[37] + W[45] + W[53]
   + W[61] + W[69] + W[77] <= 27.3
 WorkerAvailability6: W[6] + W[14] + W[22] + W[30] + W[38] + W[46] + W[54]
   + W[62] + W[70] + W[78] <= 29.53
 WorkerAvailability7: W[7] + W[15] + W[23] + W[31] + W[39] + W[47] + W[55]
   + W[63] + W[71] + W[79] <= 33.01
 LinkingConstraint0: W[0] - 7.54 Z[0] <= 0
 LinkingConstraint1: W[1] - 6.57 Z[1] <= 0
 LinkingConstraint2: W[2] - 2.91 Z[2] <= 0
 LinkingConstraint3: W[3] - 1.85 Z[3] <= 0
 LinkingConstraint4: W[4] - 5.74 Z[4] <= 0
 LinkingConstraint5: W[5] - 3.66 Z[5] <= 0
 LinkingConstraint6: W[6] - 7.86 Z[6] <= 0
 LinkingConstraint7: W[7] - 6.73 Z[7] <= 0
 LinkingConstraint8: W[8] - 7.68 Z[8] <= 0
 LinkingConstraint9: W[9] - 6.63 Z[9] <= 0
 LinkingConstraint10: W[10] - 3.03 Z[10] <= 0
 LinkingConstraint11: W[11] - 3.01 Z[11] <= 0
 LinkingConstraint12: W[12] - 6 Z[12] <= 0
 LinkingConstraint13: W[13] - 3.42 Z[13] <= 0
 LinkingConstraint14: W[14] - 4.1 Z[14] <= 0
 LinkingConstraint15: W[15] - 2.8 Z[15] <= 0
 LinkingConstraint16: W[16] - 4.35 Z[16] <= 0
 LinkingConstraint17: W[17] - 2.41 Z[17] <= 0
 LinkingConstraint18: W[18] - 4.77 Z[18] <= 0
 LinkingConstraint19: W[19] - 7.53 Z[19] <= 0
 LinkingConstraint20: W[20] - 5.87 Z[20] <= 0
 LinkingConstraint21: W[21] - 1.96 Z[21] <= 0
 LinkingConstraint22: W[22] - 5.31 Z[22] <= 0
 LinkingConstraint23: W[23] - 5.11 Z[23] <= 0
 LinkingConstraint24: W[24] - 2.7 Z[24] <= 0
 LinkingConstraint25: W[25] - 5.69 Z[25] <= 0
 LinkingConstraint26: W[26] - 4.72 Z[26] <= 0
 LinkingConstraint27: W[27] - 5.47 Z[27] <= 0
 LinkingConstraint28: W[28] - 1.37 Z[28] <= 0
 LinkingConstraint29: W[29] - 3.89 Z[29] <= 0
 LinkingConstraint30: W[30] - 6.02 Z[30] <= 0
 LinkingConstraint31: W[31] - 1.7 Z[31] <= 0
 LinkingConstraint32: W[32] - 6.4 Z[32] <= 0
 LinkingConstraint33: W[33] - 1.04 Z[33] <= 0
 LinkingConstraint34: W[34] - 4.85 Z[34] <= 0
 LinkingConstraint35: W[35] - 7.5 Z[35] <= 0
 LinkingConstraint36: W[36] - 3.85 Z[36] <= 0
 LinkingConstraint37: W[37] - 7.55 Z[37] <= 0
 LinkingConstraint38: W[38] - 7.15 Z[38] <= 0
 LinkingConstraint39: W[39] - 4.34 Z[39] <= 0
 LinkingConstraint40: W[40] - 2.4 Z[40] <= 0
 LinkingConstraint41: W[41] - 7.75 Z[41] <= 0
 LinkingConstraint42: W[42] - 3.25 Z[42] <= 0
 LinkingConstraint43: W[43] - 5.52 Z[43] <= 0
 LinkingConstraint44: W[44] - 7.36 Z[44] <= 0
 LinkingConstraint45: W[45] - 1.63 Z[45] <= 0
 LinkingConstraint46: W[46] - 5.02 Z[46] <= 0
 LinkingConstraint47: W[47] - 4.75 Z[47] <= 0
 LinkingConstraint48: W[48] - 6.06 Z[48] <= 0
 LinkingConstraint49: W[49] - 7.56 Z[49] <= 0
 LinkingConstraint50: W[50] - 7.39 Z[50] <= 0
 LinkingConstraint51: W[51] - 2.23 Z[51] <= 0
 LinkingConstraint52: W[52] - 7.18 Z[52] <= 0
 LinkingConstraint53: W[53] - 2.23 Z[53] <= 0
 LinkingConstraint54: W[54] - 7.44 Z[54] <= 0
 LinkingConstraint55: W[55] - 7.98 Z[55] <= 0
 LinkingConstraint56: W[56] - 3.78 Z[56] <= 0
 LinkingConstraint57: W[57] - 4.47 Z[57] <= 0
 LinkingConstraint58: W[58] - 7.56 Z[58] <= 0
 LinkingConstraint59: W[59] - 7.73 Z[59] <= 0
 LinkingConstraint60: W[60] - 7.48 Z[60] <= 0
 LinkingConstraint61: W[61] - 7.14 Z[61] <= 0
 LinkingConstraint62: W[62] - 1.06 Z[62] <= 0
 LinkingConstraint63: W[63] - 4.98 Z[63] <= 0
 LinkingConstraint64: W[64] - 1.75 Z[64] <= 0
 LinkingConstraint65: W[65] - 7.88 Z[65] <= 0
 LinkingConstraint66: W[66] - 2.99 Z[66] <= 0
 LinkingConstraint67: W[67] - 7.92 Z[67] <= 0
 LinkingConstraint68: W[68] - 4.8 Z[68] <= 0
 LinkingConstraint69: W[69] - 4.46 Z[69] <= 0
 LinkingConstraint70: W[70] - 7.57 Z[70] <= 0
 LinkingConstraint71: W[71] - 6.96 Z[71] <= 0
 LinkingConstraint72: W[72] - 4.28 Z[72] <= 0
 LinkingConstraint73: W[73] - 2.35 Z[73] <= 0
 LinkingConstraint74: W[74] - 1.79 Z[74] <= 0
 LinkingConstraint75: W[75] - 2.14 Z[75] <= 0
 LinkingConstraint76: W[76] - 4.21 Z[76] <= 0
 LinkingConstraint77: W[77] - 2.8 Z[77] <= 0
 LinkingConstraint78: W[78] - 2.3 Z[78] <= 0
 LinkingConstraint79: W[79] - 6.16 Z[79] <= 0
 TaskAssignment0: Z[0] + Z[1] + Z[2] + Z[3] + Z[4] + Z[5] + Z[6] + Z[7]
   >= 2
 TaskAssignment1: Z[8] + Z[9] + Z[10] + Z[11] + Z[12] + Z[13] + Z[14]
   + Z[15] >= 2
 TaskAssignment2: Z[16] + Z[17] + Z[18] + Z[19] + Z[20] + Z[21] + Z[22]
   + Z[23] >= 2
 TaskAssignment3: Z[24] + Z[25] + Z[26] + Z[27] + Z[28] + Z[29] + Z[30]
   + Z[31] >= 2
 TaskAssignment4: Z[32] + Z[33] + Z[34] + Z[35] + Z[36] + Z[37] + Z[38]
   + Z[39] >= 2
 TaskAssignment5: Z[40] + Z[41] + Z[42] + Z[43] + Z[44] + Z[45] + Z[46]
   + Z[47] >= 2
 TaskAssignment6: Z[48] + Z[49] + Z[50] + Z[51] + Z[52] + Z[53] + Z[54]
   + Z[55] >= 2
 TaskAssignment7: Z[56] + Z[57] + Z[58] + Z[59] + Z[60] + Z[61] + Z[62]
   + Z[63] >= 2
 TaskAssignment8: Z[64] + Z[65] + Z[66] + Z[67] + Z[68] + Z[69] + Z[70]
   + Z[71] >= 2
 TaskAssignment9: Z[72] + Z[73] + Z[74] + Z[75] + Z[76] + Z[77] + Z[78]
   + Z[79] >= 2
Bounds
 W[0] <= 7.98
 W[1] <= 7.98
 W[2] <= 7.98
 W[3] <= 7.98
 W[4] <= 7.98
 W[5] <= 7.98
 W[6] <= 7.98
 W[7] <= 7.98
 W[8] <= 7.98
 W[9] <= 7.98
 W[10] <= 7.98
 W[11] <= 7.98
 W[12] <= 7.98
 W[13] <= 7.98
 W[14] <= 7.98
 W[15] <= 7.98
 W[16] <= 7.98
 W[17] <= 7.98
 W[18] <= 7.98
 W[19] <= 7.98
 W[20] <= 7.98
 W[21] <= 7.98
 W[22] <= 7.98
 W[23] <= 7.98
 W[24] <= 7.98
 W[25] <= 7.98
 W[26] <= 7.98
 W[27] <= 7.98
 W[28] <= 7.98
 W[29] <= 7.98
 W[30] <= 7.98
 W[31] <= 7.98
 W[32] <= 7.98
 W[33] <= 7.98
 W[34] <= 7.98
 W[35] <= 7.98
 W[36] <= 7.98
 W[37] <= 7.98
 W[38] <= 7.98
 W[39] <= 7.98
 W[40] <= 7.98
 W[41] <= 7.98
 W[42] <= 7.98
 W[43] <= 7.98
 W[44] <= 7.98
 W[45] <= 7.98
 W[46] <= 7.98
 W[47] <= 7.98
 W[48] <= 7.98
 W[49] <= 7.98
 W[50] <= 7.98
 W[51] <= 7.98
 W[52] <= 7.98
 W[53] <= 7.98
 W[54] <= 7.98
 W[55] <= 7.98
 W[56] <= 7.98
 W[57] <= 7.98
 W[58] <= 7.98
 W[59] <= 7.98
 W[60] <= 7.98
 W[61] <= 7.98
 W[62] <= 7.98
 W[63] <= 7.98
 W[64] <= 7.98
 W[65] <= 7.98
 W[66] <= 7.98
 W[67] <= 7.98
 W[68] <= 7.98
 W[69] <= 7.98
 W[70] <= 7.98
 W[71] <= 7.98
 W[72] <= 7.98
 W[73] <= 7.98
 W[74] <= 7.98
 W[75] <= 7.98
 W[76] <= 7.98
 W[77] <= 7.98
 W[78] <= 7.98
 W[79] <= 7.98
Binaries
 Z[0] Z[1] Z[2] Z[3] Z[4] Z[5] Z[6] Z[7] Z[8] Z[9] Z[10] Z[11] Z[12] Z[13]
 Z[14] Z[15] Z[16] Z[17] Z[18] Z[19] Z[20] Z[21] Z[22] Z[23] Z[24] Z[25]
 Z[26] Z[27] Z[28] Z[29] Z[30] Z[31] Z[32] Z[33] Z[34] Z[35] Z[36] Z[37]
 Z[38] Z[39] Z[40] Z[41] Z[42] Z[43] Z[44] Z[45] Z[46] Z[47] Z[48] Z[49]
 Z[50] Z[51] Z[52] Z[53] Z[54] Z[55] Z[56] Z[57] Z[58] Z[59] Z[60] Z[61]
 Z[62] Z[63] Z[64] Z[65] Z[66] Z[67] Z[68] Z[69] Z[70] Z[71] Z[72] Z[73]
 Z[74] Z[75] Z[76] Z[77] Z[78] Z[79]
End
