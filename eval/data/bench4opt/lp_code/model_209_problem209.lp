\ Model Capital Budgeting Problem with Risk and Resource Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  19.676584997 x[0] + 70.0523443344 x[1] + 57.6079665565 x[2]
   + 24.1226821597 x[3] + 77.7789030348 x[4]
Subject To
 BudgetConstraint: 35.1672702276 x[0] + 55.63982429 x[1]
   + 54.7995762329 x[2] + 46.0791170698 x[3] + 32.8559576727 x[4]
   <= 271.5973859635
 RiskConstraint: 5.3210826106 x[0] + 74.725810005 x[1] + 70.8175602215 x[2]
   + 81.32948135389999 x[3] + 39.2217964966 x[4] <= 333.525970595
 ResourceConstraint_0: 82.2540076192 x[0] + 4.6649415232 x[1]
   + 87.10033102289999 x[2] + 53.0691575939 x[3] + 41.5878833549 x[4]
   <= 237.3978474875
 ResourceConstraint_1: 98.1009957273 x[0] + 50.7268238632 x[1]
   + 87.5448470314 x[2] + 46.2358793729 x[3] + 65.82335131630001 x[4]
   <= 484.755797134
 ResourceConstraint_2: 50.037536312 x[0] + 59.4278625016 x[1]
   + 44.5903147673 x[2] + 72.5219389295 x[3] + 16.2817606584 x[4]
   <= 172.5878108555
 Dependency_0_1: x[0] - x[1] <= 0
 Dependency_1_2: x[1] - x[2] <= 0
 Dependency_2_3: x[2] - x[3] <= 0
 Dependency_3_4: x[3] - x[4] <= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
