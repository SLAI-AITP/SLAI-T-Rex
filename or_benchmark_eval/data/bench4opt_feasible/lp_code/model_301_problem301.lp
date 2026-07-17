\ Model Capital Budgeting Problem with Multi-Period and Interdependencies
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  787619.8343191599 x[0,0] + 247207.8218117847 x[0,1]
   + 504026.064628215 x[0,2] + 156648.3924321842 x[0,3]
   + 193389.5332005469 x[0,4] + 988043.8118639207 x[1,0]
   + 831148.7679599865 x[1,1] + 627473.0464501749 x[1,2]
   + 860338.209523972 x[1,3] + 462855.613532438 x[1,4]
   + 625014.9181564859 x[2,0] + 596005.5756291385 x[2,1]
   + 146657.1297404984 x[2,2] + 990424.9321004346 x[2,3]
   + 498273.8054630733 x[2,4]
Subject To
 Budget_0: x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] <= 1e+06
 Budget_1: x[1,0] + x[1,1] + x[1,2] + x[1,3] + x[1,4] <= 1e+06
 Budget_2: x[2,0] + x[2,1] + x[2,2] + x[2,3] + x[2,4] <= 1e+06
 Regulatory_0[0]: x[0,0] <= 1e+06
 Regulatory_0[1]: x[1,0] <= 1e+06
 Regulatory_0[2]: x[2,0] <= 1e+06
 Regulatory_1[0]: x[0,1] <= 1e+06
 Regulatory_1[1]: x[1,1] <= 1e+06
 Regulatory_1[2]: x[2,1] <= 1e+06
 Regulatory_2[0]: x[0,2] <= 1e+06
 Regulatory_2[1]: x[1,2] <= 1e+06
 Regulatory_2[2]: x[2,2] <= 1e+06
 Regulatory_3[0]: x[0,3] <= 1e+06
 Regulatory_3[1]: x[1,3] <= 1e+06
 Regulatory_3[2]: x[2,3] <= 1e+06
 Regulatory_4[0]: x[0,4] <= 1e+06
 Regulatory_4[1]: x[1,4] <= 1e+06
 Regulatory_4[2]: x[2,4] <= 1e+06
 Resource_0[0]: x[0,0] <= 1e+06
 Resource_0[1]: x[1,0] <= 1e+06
 Resource_0[2]: x[2,0] <= 1e+06
 Resource_1[0]: x[0,1] <= 1e+06
 Resource_1[1]: x[1,1] <= 1e+06
 Resource_1[2]: x[2,1] <= 1e+06
 Resource_2[0]: x[0,2] <= 1e+06
 Resource_2[1]: x[1,2] <= 1e+06
 Resource_2[2]: x[2,2] <= 1e+06
 Resource_3[0]: x[0,3] <= 1e+06
 Resource_3[1]: x[1,3] <= 1e+06
 Resource_3[2]: x[2,3] <= 1e+06
 Resource_4[0]: x[0,4] <= 1e+06
 Resource_4[1]: x[1,4] <= 1e+06
 Resource_4[2]: x[2,4] <= 1e+06
 Dependency_0_0_0: <= 0
 Dependency_0_0_1: - x[0,0] <= 0
 Dependency_0_0_2: - x[0,0] - x[1,0] <= 0
 Dependency_0_0_0: <= 0
 Dependency_0_0_1: - x[0,0] <= 0
 Dependency_0_0_2: - x[0,0] - x[1,0] <= 0
 Dependency_0_0_0: <= 0
 Dependency_0_0_1: - x[0,0] <= 0
 Dependency_0_0_2: - x[0,0] - x[1,0] <= 0
 Dependency_0_0_0: <= 0
 Dependency_0_0_1: - x[0,0] <= 0
 Dependency_0_0_2: - x[0,0] - x[1,0] <= 0
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[0,3] x[0,4] x[1,0] x[1,1] x[1,2] x[1,3] x[1,4]
 x[2,0] x[2,1] x[2,2] x[2,3] x[2,4]
End
