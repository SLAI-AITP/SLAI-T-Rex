\ Model workforce_scheduling_variant
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  20.78 H[0] + 11.61 H[1] + 40.25 H[2] + 28.82 H[3] + 36.06 H[4]
   + 46.64 H[5] + 17.26 H[6] + 33.41 H[7] + 35.39 H[8] + 29.67 H[9]
   + 13.65 H[10] + 23.92 H[11] + 23.33 H[12] + 36.81 H[13] + 44.31 H[14]
   + 23.19 H[15] + 37.75 H[16] + 21.53 H[17] + 47.81 H[18] + 42.54 H[19]
   + 32 H[20] + 28.19 H[21] + 22.58 H[22] + 22.93 H[23] + 48.81 H[24]
   + 26.17 H[25] + 30.58 H[26] + 49.52 H[27] + 36.31 H[28] + 31.7 H[29]
   + 26.53 H[30] + 17.5 H[31] + 24.47 H[32] + 40.26 H[33] + 35.02 H[34]
   + 40.4 H[35] + 18.14 H[36] + 31.97 H[37] + 47.11 H[38] + 27.52 H[39]
   + 37.93 H[40] + 14.86 H[41] + 48.93 H[42] + 34.35 H[43] + 19.57 H[44]
   + 16.34 H[45] + 32.03 H[46] + 32.09 H[47] + 13.73 H[48] + 49.69 H[49]
   + 46.52 H[50] + 28.46 H[51] + 14.7 H[52] + 43.29 H[53] + 29.94 H[54]
   + 38.66 H[55] + 30.35 H[56] + 20.94 H[57] + 43.39 H[58] + 49.21 H[59]
   + 19.75 H[60] + 32.05 H[61] + 25.34 H[62] + 46.87 H[63] + 30.33 H[64]
   + 45.17 H[65] + 44.56 H[66] + 21.05 H[67] + 41.6 H[68] + 26.6 H[69]
   + 47.37 H[70] + 30.31 H[71] + 42.82 H[72] + 21.31 H[73] + 21.94 H[74]
   + 33.48 H[75] + 49.96 H[76] + 29.59 H[77] + 15.94 H[78] + 31.54 H[79]
   + 238.05 A[0] + 320.77 A[1] + 317.37 A[2] + 282.14 A[3] + 228.71 A[4]
   + 175.46 A[5] + 379 A[6] + 328.72 A[7] + 193.42 A[8] + 410.22 A[9]
   + 117.46 A[10] + 397.88 A[11] + 382.09 A[12] + 424.56 A[13]
   + 254.43 A[14] + 365.48 A[15] + 428.3 A[16] + 492.33 A[17]
   + 298.13 A[18] + 114.81 A[19] + 300.92 A[20] + 336.07 A[21]
   + 447.88 A[22] + 449.68 A[23] + 276.12 A[24] + 310.38 A[25]
   + 282.77 A[26] + 388.98 A[27] + 263.99 A[28] + 361.91 A[29]
   + 161.74 A[30] + 287.8 A[31] + 487.68 A[32] + 235.42 A[33]
   + 377.08 A[34] + 359.93 A[35] + 440.71 A[36] + 440.94 A[37]
   + 443.74 A[38] + 252 A[39] + 226.66 A[40] + 387.49 A[41] + 403.76 A[42]
   + 448.95 A[43] + 114.36 A[44] + 127.37 A[45] + 352.46 A[46]
   + 468.37 A[47] + 498.97 A[48] + 398.71 A[49] + 273.59 A[50]
   + 139.38 A[51] + 353.5 A[52] + 449.03 A[53] + 277.47 A[54] + 377.6 A[55]
   + 461.37 A[56] + 118.4 A[57] + 418.46 A[58] + 217.35 A[59]
   + 249.94 A[60] + 158.23 A[61] + 312.47 A[62] + 326.37 A[63]
   + 417.01 A[64] + 167.99 A[65] + 131.59 A[66] + 448.34 A[67]
   + 347.88 A[68] + 196.33 A[69] + 465.13 A[70] + 157.25 A[71]
   + 284.46 A[72] + 201.59 A[73] + 202.13 A[74] + 103.76 A[75]
   + 421.85 A[76] + 460.48 A[77] + 371.04 A[78] + 163.19 A[79]
Subject To
 TaskCompletion0: H[0] + H[1] + H[2] + H[3] + H[4] + H[5] + H[6] + H[7]
   = 6.87
 TaskCompletion1: H[8] + H[9] + H[10] + H[11] + H[12] + H[13] + H[14]
   + H[15] = 15.33
 TaskCompletion2: H[16] + H[17] + H[18] + H[19] + H[20] + H[21] + H[22]
   + H[23] = 15.5
 TaskCompletion3: H[24] + H[25] + H[26] + H[27] + H[28] + H[29] + H[30]
   + H[31] = 19.14
 TaskCompletion4: H[32] + H[33] + H[34] + H[35] + H[36] + H[37] + H[38]
   + H[39] = 12.51
 TaskCompletion5: H[40] + H[41] + H[42] + H[43] + H[44] + H[45] + H[46]
   + H[47] = 12.41
 TaskCompletion6: H[48] + H[49] + H[50] + H[51] + H[52] + H[53] + H[54]
   + H[55] = 6.21
 TaskCompletion7: H[56] + H[57] + H[58] + H[59] + H[60] + H[61] + H[62]
   + H[63] = 5.6
 TaskCompletion8: H[64] + H[65] + H[66] + H[67] + H[68] + H[69] + H[70]
   + H[71] = 11.48
 TaskCompletion9: H[72] + H[73] + H[74] + H[75] + H[76] + H[77] + H[78]
   + H[79] = 9.83
 EmployeeAvailability0: H[0] + H[8] + H[16] + H[24] + H[32] + H[40] + H[48]
   + H[56] + H[64] + H[72] <= 35.01
 EmployeeAvailability1: H[1] + H[9] + H[17] + H[25] + H[33] + H[41] + H[49]
   + H[57] + H[65] + H[73] <= 31.83
 EmployeeAvailability2: H[2] + H[10] + H[18] + H[26] + H[34] + H[42]
   + H[50] + H[58] + H[66] + H[74] <= 49.24
 EmployeeAvailability3: H[3] + H[11] + H[19] + H[27] + H[35] + H[43]
   + H[51] + H[59] + H[67] + H[75] <= 46.72
 EmployeeAvailability4: H[4] + H[12] + H[20] + H[28] + H[36] + H[44]
   + H[52] + H[60] + H[68] + H[76] <= 41.5
 EmployeeAvailability5: H[5] + H[13] + H[21] + H[29] + H[37] + H[45]
   + H[53] + H[61] + H[69] + H[77] <= 49.02
 EmployeeAvailability6: H[6] + H[14] + H[22] + H[30] + H[38] + H[46]
   + H[54] + H[62] + H[70] + H[78] <= 49.99
 EmployeeAvailability7: H[7] + H[15] + H[23] + H[31] + H[39] + H[47]
   + H[55] + H[63] + H[71] + H[79] <= 43.45
 LinkingConstraint0: H[0] - 9.42 A[0] <= 0
 LinkingConstraint1: H[1] - 8.46 A[1] <= 0
 LinkingConstraint2: H[2] - 10.88 A[2] <= 0
 LinkingConstraint3: H[3] - 11.39 A[3] <= 0
 LinkingConstraint4: H[4] - 9.24 A[4] <= 0
 LinkingConstraint5: H[5] - 7.5 A[5] <= 0
 LinkingConstraint6: H[6] - 13.45 A[6] <= 0
 LinkingConstraint7: H[7] - 6.99 A[7] <= 0
 LinkingConstraint8: H[8] - 8.85 A[8] <= 0
 LinkingConstraint9: H[9] - 9.83 A[9] <= 0
 LinkingConstraint10: H[10] - 7.37 A[10] <= 0
 LinkingConstraint11: H[11] - 10.72 A[11] <= 0
 LinkingConstraint12: H[12] - 10.75 A[12] <= 0
 LinkingConstraint13: H[13] - 14.93 A[13] <= 0
 LinkingConstraint14: H[14] - 7.95 A[14] <= 0
 LinkingConstraint15: H[15] - 14.78 A[15] <= 0
 LinkingConstraint16: H[16] - 11.58 A[16] <= 0
 LinkingConstraint17: H[17] - 7.74 A[17] <= 0
 LinkingConstraint18: H[18] - 10.66 A[18] <= 0
 LinkingConstraint19: H[19] - 11.86 A[19] <= 0
 LinkingConstraint20: H[20] - 12.45 A[20] <= 0
 LinkingConstraint21: H[21] - 5.49 A[21] <= 0
 LinkingConstraint22: H[22] - 11.06 A[22] <= 0
 LinkingConstraint23: H[23] - 9.97 A[23] <= 0
 LinkingConstraint24: H[24] - 14.04 A[24] <= 0
 LinkingConstraint25: H[25] - 7.86 A[25] <= 0
 LinkingConstraint26: H[26] - 12.99 A[26] <= 0
 LinkingConstraint27: H[27] - 11.07 A[27] <= 0
 LinkingConstraint28: H[28] - 8.52 A[28] <= 0
 LinkingConstraint29: H[29] - 11.37 A[29] <= 0
 LinkingConstraint30: H[30] - 11.21 A[30] <= 0
 LinkingConstraint31: H[31] - 11.78 A[31] <= 0
 LinkingConstraint32: H[32] - 12.21 A[32] <= 0
 LinkingConstraint33: H[33] - 11.59 A[33] <= 0
 LinkingConstraint34: H[34] - 13.38 A[34] <= 0
 LinkingConstraint35: H[35] - 11.28 A[35] <= 0
 LinkingConstraint36: H[36] - 14.03 A[36] <= 0
 LinkingConstraint37: H[37] - 11.46 A[37] <= 0
 LinkingConstraint38: H[38] - 8.09 A[38] <= 0
 LinkingConstraint39: H[39] - 9.41 A[39] <= 0
 LinkingConstraint40: H[40] - 10.8 A[40] <= 0
 LinkingConstraint41: H[41] - 12.32 A[41] <= 0
 LinkingConstraint42: H[42] - 5.9 A[42] <= 0
 LinkingConstraint43: H[43] - 7.95 A[43] <= 0
 LinkingConstraint44: H[44] - 12.47 A[44] <= 0
 LinkingConstraint45: H[45] - 6.76 A[45] <= 0
 LinkingConstraint46: H[46] - 6.32 A[46] <= 0
 LinkingConstraint47: H[47] - 10.39 A[47] <= 0
 LinkingConstraint48: H[48] - 14.71 A[48] <= 0
 LinkingConstraint49: H[49] - 10.31 A[49] <= 0
 LinkingConstraint50: H[50] - 14.13 A[50] <= 0
 LinkingConstraint51: H[51] - 13.3 A[51] <= 0
 LinkingConstraint52: H[52] - 7.57 A[52] <= 0
 LinkingConstraint53: H[53] - 13.25 A[53] <= 0
 LinkingConstraint54: H[54] - 9.82 A[54] <= 0
 LinkingConstraint55: H[55] - 13.06 A[55] <= 0
 LinkingConstraint56: H[56] - 12.47 A[56] <= 0
 LinkingConstraint57: H[57] - 8.39 A[57] <= 0
 LinkingConstraint58: H[58] - 6.15 A[58] <= 0
 LinkingConstraint59: H[59] - 14.63 A[59] <= 0
 LinkingConstraint60: H[60] - 6.41 A[60] <= 0
 LinkingConstraint61: H[61] - 14.67 A[61] <= 0
 LinkingConstraint62: H[62] - 13.6 A[62] <= 0
 LinkingConstraint63: H[63] - 12.24 A[63] <= 0
 LinkingConstraint64: H[64] - 14.8 A[64] <= 0
 LinkingConstraint65: H[65] - 14.67 A[65] <= 0
 LinkingConstraint66: H[66] - 13.05 A[66] <= 0
 LinkingConstraint67: H[67] - 8.66 A[67] <= 0
 LinkingConstraint68: H[68] - 12.91 A[68] <= 0
 LinkingConstraint69: H[69] - 5.14 A[69] <= 0
 LinkingConstraint70: H[70] - 10.37 A[70] <= 0
 LinkingConstraint71: H[71] - 9.55 A[71] <= 0
 LinkingConstraint72: H[72] - 11.73 A[72] <= 0
 LinkingConstraint73: H[73] - 11.72 A[73] <= 0
 LinkingConstraint74: H[74] - 10.85 A[74] <= 0
 LinkingConstraint75: H[75] - 13.22 A[75] <= 0
 LinkingConstraint76: H[76] - 14.4 A[76] <= 0
 LinkingConstraint77: H[77] - 6.08 A[77] <= 0
 LinkingConstraint78: H[78] - 7.34 A[78] <= 0
 LinkingConstraint79: H[79] - 5.25 A[79] <= 0
 TotalOvertimeHours: H[0] + H[1] + H[2] + H[3] + H[4] + H[5] + H[6] + H[7]
   + H[8] + H[9] + H[10] + H[11] + H[12] + H[13] + H[14] + H[15] + H[16]
   + H[17] + H[18] + H[19] + H[20] + H[21] + H[22] + H[23] + H[24] + H[25]
   + H[26] + H[27] + H[28] + H[29] + H[30] + H[31] + H[32] + H[33] + H[34]
   + H[35] + H[36] + H[37] + H[38] + H[39] + H[40] + H[41] + H[42] + H[43]
   + H[44] + H[45] + H[46] + H[47] + H[48] + H[49] + H[50] + H[51] + H[52]
   + H[53] + H[54] + H[55] + H[56] + H[57] + H[58] + H[59] + H[60] + H[61]
   + H[62] + H[63] + H[64] + H[65] + H[66] + H[67] + H[68] + H[69] + H[70]
   + H[71] + H[72] + H[73] + H[74] + H[75] + H[76] + H[77] + H[78] + H[79]
   <= 392.13
Bounds
 H[0] <= 9.42
 H[1] <= 8.46
 H[2] <= 10.88
 H[3] <= 11.39
 H[4] <= 9.24
 H[5] <= 7.5
 H[6] <= 13.45
 H[7] <= 6.99
 H[8] <= 8.85
 H[9] <= 9.83
 H[10] <= 7.37
 H[11] <= 10.72
 H[12] <= 10.75
 H[13] <= 14.93
 H[14] <= 7.95
 H[15] <= 14.78
 H[16] <= 11.58
 H[17] <= 7.74
 H[18] <= 10.66
 H[19] <= 11.86
 H[20] <= 12.45
 H[21] <= 5.49
 H[22] <= 11.06
 H[23] <= 9.97
 H[24] <= 14.04
 H[25] <= 7.86
 H[26] <= 12.99
 H[27] <= 11.07
 H[28] <= 8.52
 H[29] <= 11.37
 H[30] <= 11.21
 H[31] <= 11.78
 H[32] <= 12.21
 H[33] <= 11.59
 H[34] <= 13.38
 H[35] <= 11.28
 H[36] <= 14.03
 H[37] <= 11.46
 H[38] <= 8.09
 H[39] <= 9.41
 H[40] <= 10.8
 H[41] <= 12.32
 H[42] <= 5.9
 H[43] <= 7.95
 H[44] <= 12.47
 H[45] <= 6.76
 H[46] <= 6.32
 H[47] <= 10.39
 H[48] <= 14.71
 H[49] <= 10.31
 H[50] <= 14.13
 H[51] <= 13.3
 H[52] <= 7.57
 H[53] <= 13.25
 H[54] <= 9.82
 H[55] <= 13.06
 H[56] <= 12.47
 H[57] <= 8.39
 H[58] <= 6.15
 H[59] <= 14.63
 H[60] <= 6.41
 H[61] <= 14.67
 H[62] <= 13.6
 H[63] <= 12.24
 H[64] <= 14.8
 H[65] <= 14.67
 H[66] <= 13.05
 H[67] <= 8.66
 H[68] <= 12.91
 H[69] <= 5.14
 H[70] <= 10.37
 H[71] <= 9.55
 H[72] <= 11.73
 H[73] <= 11.72
 H[74] <= 10.85
 H[75] <= 13.22
 H[76] <= 14.4
 H[77] <= 6.08
 H[78] <= 7.34
 H[79] <= 5.25
Binaries
 A[0] A[1] A[2] A[3] A[4] A[5] A[6] A[7] A[8] A[9] A[10] A[11] A[12] A[13]
 A[14] A[15] A[16] A[17] A[18] A[19] A[20] A[21] A[22] A[23] A[24] A[25]
 A[26] A[27] A[28] A[29] A[30] A[31] A[32] A[33] A[34] A[35] A[36] A[37]
 A[38] A[39] A[40] A[41] A[42] A[43] A[44] A[45] A[46] A[47] A[48] A[49]
 A[50] A[51] A[52] A[53] A[54] A[55] A[56] A[57] A[58] A[59] A[60] A[61]
 A[62] A[63] A[64] A[65] A[66] A[67] A[68] A[69] A[70] A[71] A[72] A[73]
 A[74] A[75] A[76] A[77] A[78] A[79]
End
