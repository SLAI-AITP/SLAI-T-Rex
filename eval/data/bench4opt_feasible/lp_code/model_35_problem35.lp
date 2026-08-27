\ Model Cutting Stock Problem with Minimum Waste Constraint
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0] + x[1] + x[2] + x[3] + x[4]
Subject To
 R0: x[0] + x[1] + x[2] + x[3] + x[4] <= 10000
 R1: x[0] >= 1
 R2: x[1] >= 1
 R3: x[2] >= 1
 R4: x[3] >= 1
 R5: x[4] >= 1
 R6: - x[0] - x[1] - x[2] - x[3] - x[4] >= -10000
Bounds
End
