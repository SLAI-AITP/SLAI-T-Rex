\ Model Capital Budgeting Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  58.7835566398 x[0] + 52.7554888377 x[1] + 93.53591951590001 x[2]
   + 21.2216607429 x[3] + 71.9029882782 x[4]
Subject To
 budget: 24.629909309 x[0] + 40.1827988323 x[1] + 67.497332073 x[2]
   + 30.6997109001 x[3] + 32.3015424309 x[4] <= 377.172923745
 dependency_0_2: - x[0] + x[2] <= 0
 dependency_1_3: - x[1] + x[3] <= 0
 dependency_2_1: x[1] - x[2] <= 0
 dependency_2_4: - x[2] + x[4] <= 0
 dependency_3_0: x[0] - x[3] <= 0
 dependency_3_2: x[2] - x[3] <= 0
 dependency_4_0: x[0] - x[4] <= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
