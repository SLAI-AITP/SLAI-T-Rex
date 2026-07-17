\ Model medical_resource_allocation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  44.48 V[0] + 33.03 V[1] + 32.94 V[2] + 38.38 V[3] + 26.71 V[4]
   + 14.61 V[5] + 10.83 V[6] + 22.99 V[7] + 42.05 V[8] + 34.73 V[9]
   + 43.28 V[10] + 46.79 V[11] + 13.53 V[12] + 43.78 V[13] + 19.73 V[14]
   + 33.55 V[15] + 30.96 V[16] + 25.83 V[17] + 22.41 V[18] + 23.58 V[19]
   + 23.32 V[20] + 16.73 V[21] + 30.42 V[22] + 14.56 V[23] + 30.4 V[24]
   + 46.24 V[25] + 23.98 V[26] + 39.1 V[27] + 42.76 V[28] + 42.6 V[29]
   + 19.45 V[30] + 15.86 V[31] + 17.89 V[32] + 34.1 V[33] + 40.41 V[34]
   + 36.22 V[35] + 17.09 V[36] + 40.91 V[37] + 29.76 V[38] + 40.18 V[39]
   + 40.4 V[40] + 27.96 V[41] + 46.97 V[42] + 32.58 V[43] + 35.41 V[44]
   + 34.98 V[45] + 44.57 V[46] + 35.09 V[47] + 16.04 V[48] + 12.73 V[49]
   + 2768.83 Z[0] + 2211.28 Z[1] + 2098.69 Z[2] + 1224.69 Z[3]
   + 3029.35 Z[4] + 2241.63 Z[5] + 2807.66 Z[6] + 1227.56 Z[7]
   + 4326.79 Z[8] + 1306.92 Z[9] + 4457 Z[10] + 4421.17 Z[11]
   + 3460.03 Z[12] + 3028.27 Z[13] + 2850.85 Z[14] + 3217.27 Z[15]
   + 4167.27 Z[16] + 4583.51 Z[17] + 2798.93 Z[18] + 4239.26 Z[19]
   + 3607.35 Z[20] + 2286.11 Z[21] + 2902.52 Z[22] + 1603.44 Z[23]
   + 1247.49 Z[24] + 1414.01 Z[25] + 4596.51 Z[26] + 2373.75 Z[27]
   + 3857.26 Z[28] + 3018.2 Z[29] + 1690.24 Z[30] + 1990.97 Z[31]
   + 2751.03 Z[32] + 2757.69 Z[33] + 3090.99 Z[34] + 1634.98 Z[35]
   + 2491.41 Z[36] + 2131.57 Z[37] + 2635.08 Z[38] + 2353.47 Z[39]
   + 3391.54 Z[40] + 4156.91 Z[41] + 3589.22 Z[42] + 1263.65 Z[43]
   + 1378.02 Z[44] + 3713.52 Z[45] + 2136.59 Z[46] + 3894.93 Z[47]
   + 3626.26 Z[48] + 4625.37 Z[49]
Subject To
 DemandSatisfaction_Hospital0: V[0] + V[1] + V[2] + V[3] + V[4] = 88.33
 DemandSatisfaction_Hospital1: V[5] + V[6] + V[7] + V[8] + V[9] = 99.22
 DemandSatisfaction_Hospital2: V[10] + V[11] + V[12] + V[13] + V[14]
   = 51.16
 DemandSatisfaction_Hospital3: V[15] + V[16] + V[17] + V[18] + V[19]
   = 162.05
 DemandSatisfaction_Hospital4: V[20] + V[21] + V[22] + V[23] + V[24]
   = 76.35
 DemandSatisfaction_Hospital5: V[25] + V[26] + V[27] + V[28] + V[29]
   = 107.03
 DemandSatisfaction_Hospital6: V[30] + V[31] + V[32] + V[33] + V[34]
   = 155.55
 DemandSatisfaction_Hospital7: V[35] + V[36] + V[37] + V[38] + V[39]
   = 125.04
 DemandSatisfaction_Hospital8: V[40] + V[41] + V[42] + V[43] + V[44] = 175
 DemandSatisfaction_Hospital9: V[45] + V[46] + V[47] + V[48] + V[49]
   = 170.93
 Capacity_DistributionCenter0: V[0] + V[5] + V[10] + V[15] + V[20] + V[25]
   + V[30] + V[35] + V[40] + V[45] = 314.42
 Capacity_DistributionCenter1: V[1] + V[6] + V[11] + V[16] + V[21] + V[26]
   + V[31] + V[36] + V[41] + V[46] = 472.35
 Capacity_DistributionCenter2: V[2] + V[7] + V[12] + V[17] + V[22] + V[27]
   + V[32] + V[37] + V[42] + V[47] = 308.46
 Capacity_DistributionCenter3: V[3] + V[8] + V[13] + V[18] + V[23] + V[28]
   + V[33] + V[38] + V[43] + V[48] = 303.75
 Capacity_DistributionCenter4: V[4] + V[9] + V[14] + V[19] + V[24] + V[29]
   + V[34] + V[39] + V[44] + V[49] = 484.23
 LinkingConstraint_Route0: V[0] - 137.33 Z[0] <= 0
 LinkingConstraint_Route1: V[1] - 83.34 Z[1] <= 0
 LinkingConstraint_Route2: V[2] - 108.27 Z[2] <= 0
 LinkingConstraint_Route3: V[3] - 64.14 Z[3] <= 0
 LinkingConstraint_Route4: V[4] - 84.98 Z[4] <= 0
 LinkingConstraint_Route5: V[5] - 146.77 Z[5] <= 0
 LinkingConstraint_Route6: V[6] - 119.85 Z[6] <= 0
 LinkingConstraint_Route7: V[7] - 89.2 Z[7] <= 0
 LinkingConstraint_Route8: V[8] - 109.5 Z[8] <= 0
 LinkingConstraint_Route9: V[9] - 143.8 Z[9] <= 0
 LinkingConstraint_Route10: V[10] - 80.96 Z[10] <= 0
 LinkingConstraint_Route11: V[11] - 87.67 Z[11] <= 0
 LinkingConstraint_Route12: V[12] - 129.17 Z[12] <= 0
 LinkingConstraint_Route13: V[13] - 131.32 Z[13] <= 0
 LinkingConstraint_Route14: V[14] - 117.01 Z[14] <= 0
 LinkingConstraint_Route15: V[15] - 132.9 Z[15] <= 0
 LinkingConstraint_Route16: V[16] - 123.88 Z[16] <= 0
 LinkingConstraint_Route17: V[17] - 118.54 Z[17] <= 0
 LinkingConstraint_Route18: V[18] - 102.64 Z[18] <= 0
 LinkingConstraint_Route19: V[19] - 114.6 Z[19] <= 0
 LinkingConstraint_Route20: V[20] - 92.34 Z[20] <= 0
 LinkingConstraint_Route21: V[21] - 86.18 Z[21] <= 0
 LinkingConstraint_Route22: V[22] - 86.26 Z[22] <= 0
 LinkingConstraint_Route23: V[23] - 68.03 Z[23] <= 0
 LinkingConstraint_Route24: V[24] - 71.42 Z[24] <= 0
 LinkingConstraint_Route25: V[25] - 144.77 Z[25] <= 0
 LinkingConstraint_Route26: V[26] - 98.63 Z[26] <= 0
 LinkingConstraint_Route27: V[27] - 72.65 Z[27] <= 0
 LinkingConstraint_Route28: V[28] - 63.76 Z[28] <= 0
 LinkingConstraint_Route29: V[29] - 57.72 Z[29] <= 0
 LinkingConstraint_Route30: V[30] - 134.44 Z[30] <= 0
 LinkingConstraint_Route31: V[31] - 60.11 Z[31] <= 0
 LinkingConstraint_Route32: V[32] - 127.09 Z[32] <= 0
 LinkingConstraint_Route33: V[33] - 133.51 Z[33] <= 0
 LinkingConstraint_Route34: V[34] - 138.37 Z[34] <= 0
 LinkingConstraint_Route35: V[35] - 53.77 Z[35] <= 0
 LinkingConstraint_Route36: V[36] - 83.68 Z[36] <= 0
 LinkingConstraint_Route37: V[37] - 126.63 Z[37] <= 0
 LinkingConstraint_Route38: V[38] - 63.1 Z[38] <= 0
 LinkingConstraint_Route39: V[39] - 87.67 Z[39] <= 0
 LinkingConstraint_Route40: V[40] - 66.22 Z[40] <= 0
 LinkingConstraint_Route41: V[41] - 133.13 Z[41] <= 0
 LinkingConstraint_Route42: V[42] - 127.11 Z[42] <= 0
 LinkingConstraint_Route43: V[43] - 130.9 Z[43] <= 0
 LinkingConstraint_Route44: V[44] - 66.55 Z[44] <= 0
 LinkingConstraint_Route45: V[45] - 93.77 Z[45] <= 0
 LinkingConstraint_Route46: V[46] - 91.09 Z[46] <= 0
 LinkingConstraint_Route47: V[47] - 117.64 Z[47] <= 0
 LinkingConstraint_Route48: V[48] - 73.75 Z[48] <= 0
 LinkingConstraint_Route49: V[49] - 94.42 Z[49] <= 0
Bounds
Binaries
 Z[0] Z[1] Z[2] Z[3] Z[4] Z[5] Z[6] Z[7] Z[8] Z[9] Z[10] Z[11] Z[12] Z[13]
 Z[14] Z[15] Z[16] Z[17] Z[18] Z[19] Z[20] Z[21] Z[22] Z[23] Z[24] Z[25]
 Z[26] Z[27] Z[28] Z[29] Z[30] Z[31] Z[32] Z[33] Z[34] Z[35] Z[36] Z[37]
 Z[38] Z[39] Z[40] Z[41] Z[42] Z[43] Z[44] Z[45] Z[46] Z[47] Z[48] Z[49]
End
