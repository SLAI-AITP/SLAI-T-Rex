\ Model Portfolio Optimization Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  608.0718479823421 x[0] + 674.1603419945382 x[1] + 417.7603937928351 x[2]
   + 236.3446850655531 x[3] + 567.6684261214061 x[4]
   + 398.7607474331104 x[5] + 293.3628937377605 x[6]
Subject To
 Land_Constraint: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6]
   <= 5546.681268499005
 Water_Constraint: 358.1309938156717 x[0] + 382.8613752960301 x[1]
   + 278.2742368760549 x[2] + 11.15933553490268 x[3]
   + 467.1539387151598 x[4] + 137.9232232631313 x[5]
   + 151.7038226937451 x[6] <= 7705.015844946525
Bounds
End
