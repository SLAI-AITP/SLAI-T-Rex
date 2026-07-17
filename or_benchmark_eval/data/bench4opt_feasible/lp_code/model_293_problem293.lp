\ Model Capital Budgeting Problem with Environmental Impact Consideration
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  8.7042360520953387e+07 x[0] + 7.0712675959718168e+07 x[1]
   + 8.5420114324707702e+07 x[2] + 2.7247443937456377e+07 x[3]
   + 5.0515646132565342e+07 x[4]
Subject To
 Budget: x[0] + x[1] + x[2] + x[3] + x[4] <= 1e+06
 Regulatory_0: x[0] <= 1
 Regulatory_1: x[1] <= 1
 Regulatory_2: x[2] <= 1
 Regulatory_3: x[3] <= 1
 Regulatory_4: x[4] <= 1
 EnvironmentalImpact: x[0] + x[1] + x[2] + x[3] + x[4] <= 1e+06
 Dependency_0_0: <= 0
 Dependency_0_0: <= 0
 Dependency_0_0: <= 0
 Dependency_0_0: <= 0
 Dependency_0_0: <= 0
 Dependency_0_0: <= 0
 Dependency_0_0: <= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
