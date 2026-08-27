\ Model medical_resource_allocation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  11.48 V[0] + 29.2 V[1] + 11.37 V[2] + 13.79 V[3] + 25.37 V[4] + 22.34 V[5]
   + 37.95 V[6] + 29.33 V[7] + 31.03 V[8] + 38.55 V[9] + 38.09 V[10]
   + 38.42 V[11] + 37.02 V[12] + 31.52 V[13] + 20.89 V[14] + 30.91 V[15]
   + 32.3 V[16] + 11.06 V[17] + 13.31 V[18] + 47.54 V[19] + 49.37 V[20]
   + 18.45 V[21] + 25.73 V[22] + 12.66 V[23] + 32.8 V[24] + 44.41 V[25]
   + 48.37 V[26] + 31.19 V[27] + 42.27 V[28] + 23.08 V[29] + 17.68 V[30]
   + 37.42 V[31] + 33.75 V[32] + 42.89 V[33] + 31.49 V[34] + 25.36 V[35]
   + 32.02 V[36] + 20.18 V[37] + 13.03 V[38] + 31.97 V[39] + 26.45 V[40]
   + 17.1 V[41] + 18.07 V[42] + 20.18 V[43] + 48.8 V[44] + 34.46 V[45]
   + 27.11 V[46] + 30.97 V[47] + 22.38 V[48] + 28.53 V[49] + 4427.84 Z[0]
   + 3430.33 Z[1] + 1139.44 Z[2] + 4810.92 Z[3] + 3010.76 Z[4]
   + 4622.86 Z[5] + 3336.68 Z[6] + 1574.34 Z[7] + 3998.51 Z[8]
   + 2547.03 Z[9] + 4717.65 Z[10] + 2202.79 Z[11] + 2125.58 Z[12]
   + 1646.85 Z[13] + 4135.61 Z[14] + 4006.25 Z[15] + 4992.65 Z[16]
   + 1280.98 Z[17] + 4385.87 Z[18] + 3015.35 Z[19] + 1609.69 Z[20]
   + 2841.88 Z[21] + 4885.42 Z[22] + 2892.95 Z[23] + 3271.27 Z[24]
   + 3685.56 Z[25] + 1174.43 Z[26] + 3376.54 Z[27] + 2539.27 Z[28]
   + 1459.47 Z[29] + 3040.17 Z[30] + 1293.62 Z[31] + 2948.25 Z[32]
   + 1700.03 Z[33] + 1813.15 Z[34] + 4946.99 Z[35] + 4648.49 Z[36]
   + 3052.45 Z[37] + 1648.43 Z[38] + 2188 Z[39] + 4277.05 Z[40]
   + 1188.63 Z[41] + 3633.1 Z[42] + 1819.63 Z[43] + 4864.52 Z[44]
   + 1558.47 Z[45] + 2938.68 Z[46] + 1610.45 Z[47] + 4073.75 Z[48]
   + 2676.29 Z[49]
Subject To
 DemandSatisfaction_Hospital0: V[0] + V[1] + V[2] + V[3] + V[4] = 1
 DemandSatisfaction_Hospital1: V[5] + V[6] + V[7] + V[8] + V[9] = 1
 DemandSatisfaction_Hospital2: V[10] + V[11] + V[12] + V[13] + V[14] = 1
 DemandSatisfaction_Hospital3: V[15] + V[16] + V[17] + V[18] + V[19] = 1
 DemandSatisfaction_Hospital4: V[20] + V[21] + V[22] + V[23] + V[24] = 1
 DemandSatisfaction_Hospital5: V[25] + V[26] + V[27] + V[28] + V[29] = 1
 DemandSatisfaction_Hospital6: V[30] + V[31] + V[32] + V[33] + V[34] = 1
 DemandSatisfaction_Hospital7: V[35] + V[36] + V[37] + V[38] + V[39] = 1
 DemandSatisfaction_Hospital8: V[40] + V[41] + V[42] + V[43] + V[44] = 1
 DemandSatisfaction_Hospital9: V[45] + V[46] + V[47] + V[48] + V[49] = 1
 Capacity_DistributionCenter0: V[0] + V[5] + V[10] + V[15] + V[20] + V[25]
   + V[30] + V[35] + V[40] + V[45] = 2
 Capacity_DistributionCenter1: V[1] + V[6] + V[11] + V[16] + V[21] + V[26]
   + V[31] + V[36] + V[41] + V[46] = 2
 Capacity_DistributionCenter2: V[2] + V[7] + V[12] + V[17] + V[22] + V[27]
   + V[32] + V[37] + V[42] + V[47] = 2
 Capacity_DistributionCenter3: V[3] + V[8] + V[13] + V[18] + V[23] + V[28]
   + V[33] + V[38] + V[43] + V[48] = 2
 Capacity_DistributionCenter4: V[4] + V[9] + V[14] + V[19] + V[24] + V[29]
   + V[34] + V[39] + V[44] + V[49] = 2
 LinkingConstraint_Route0: V[0] - 1e+06 Z[0] <= 0
 LinkingConstraint_Route1: V[1] - 1e+06 Z[1] <= 0
 LinkingConstraint_Route2: V[2] - 1e+06 Z[2] <= 0
 LinkingConstraint_Route3: V[3] - 1e+06 Z[3] <= 0
 LinkingConstraint_Route4: V[4] - 1e+06 Z[4] <= 0
 LinkingConstraint_Route5: V[5] - 1e+06 Z[5] <= 0
 LinkingConstraint_Route6: V[6] - 1e+06 Z[6] <= 0
 LinkingConstraint_Route7: V[7] - 1e+06 Z[7] <= 0
 LinkingConstraint_Route8: V[8] - 1e+06 Z[8] <= 0
 LinkingConstraint_Route9: V[9] - 1e+06 Z[9] <= 0
 LinkingConstraint_Route10: V[10] - 1e+06 Z[10] <= 0
 LinkingConstraint_Route11: V[11] - 1e+06 Z[11] <= 0
 LinkingConstraint_Route12: V[12] - 1e+06 Z[12] <= 0
 LinkingConstraint_Route13: V[13] - 1e+06 Z[13] <= 0
 LinkingConstraint_Route14: V[14] - 1e+06 Z[14] <= 0
 LinkingConstraint_Route15: V[15] - 1e+06 Z[15] <= 0
 LinkingConstraint_Route16: V[16] - 1e+06 Z[16] <= 0
 LinkingConstraint_Route17: V[17] - 1e+06 Z[17] <= 0
 LinkingConstraint_Route18: V[18] - 1e+06 Z[18] <= 0
 LinkingConstraint_Route19: V[19] - 1e+06 Z[19] <= 0
 LinkingConstraint_Route20: V[20] - 1e+06 Z[20] <= 0
 LinkingConstraint_Route21: V[21] - 1e+06 Z[21] <= 0
 LinkingConstraint_Route22: V[22] - 1e+06 Z[22] <= 0
 LinkingConstraint_Route23: V[23] - 1e+06 Z[23] <= 0
 LinkingConstraint_Route24: V[24] - 1e+06 Z[24] <= 0
 LinkingConstraint_Route25: V[25] - 1e+06 Z[25] <= 0
 LinkingConstraint_Route26: V[26] - 1e+06 Z[26] <= 0
 LinkingConstraint_Route27: V[27] - 1e+06 Z[27] <= 0
 LinkingConstraint_Route28: V[28] - 1e+06 Z[28] <= 0
 LinkingConstraint_Route29: V[29] - 1e+06 Z[29] <= 0
 LinkingConstraint_Route30: V[30] - 1e+06 Z[30] <= 0
 LinkingConstraint_Route31: V[31] - 1e+06 Z[31] <= 0
 LinkingConstraint_Route32: V[32] - 1e+06 Z[32] <= 0
 LinkingConstraint_Route33: V[33] - 1e+06 Z[33] <= 0
 LinkingConstraint_Route34: V[34] - 1e+06 Z[34] <= 0
 LinkingConstraint_Route35: V[35] - 1e+06 Z[35] <= 0
 LinkingConstraint_Route36: V[36] - 1e+06 Z[36] <= 0
 LinkingConstraint_Route37: V[37] - 1e+06 Z[37] <= 0
 LinkingConstraint_Route38: V[38] - 1e+06 Z[38] <= 0
 LinkingConstraint_Route39: V[39] - 1e+06 Z[39] <= 0
 LinkingConstraint_Route40: V[40] - 1e+06 Z[40] <= 0
 LinkingConstraint_Route41: V[41] - 1e+06 Z[41] <= 0
 LinkingConstraint_Route42: V[42] - 1e+06 Z[42] <= 0
 LinkingConstraint_Route43: V[43] - 1e+06 Z[43] <= 0
 LinkingConstraint_Route44: V[44] - 1e+06 Z[44] <= 0
 LinkingConstraint_Route45: V[45] - 1e+06 Z[45] <= 0
 LinkingConstraint_Route46: V[46] - 1e+06 Z[46] <= 0
 LinkingConstraint_Route47: V[47] - 1e+06 Z[47] <= 0
 LinkingConstraint_Route48: V[48] - 1e+06 Z[48] <= 0
 LinkingConstraint_Route49: V[49] - 1e+06 Z[49] <= 0
Bounds
Binaries
 Z[0] Z[1] Z[2] Z[3] Z[4] Z[5] Z[6] Z[7] Z[8] Z[9] Z[10] Z[11] Z[12] Z[13]
 Z[14] Z[15] Z[16] Z[17] Z[18] Z[19] Z[20] Z[21] Z[22] Z[23] Z[24] Z[25]
 Z[26] Z[27] Z[28] Z[29] Z[30] Z[31] Z[32] Z[33] Z[34] Z[35] Z[36] Z[37]
 Z[38] Z[39] Z[40] Z[41] Z[42] Z[43] Z[44] Z[45] Z[46] Z[47] Z[48] Z[49]
End
