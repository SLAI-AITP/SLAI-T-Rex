\ Model Cutting Stock Problem with Minimum Waste and Maximum Utilization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7] + 10000 Constant
Subject To
 R0: x[0] <= 1200
 R1: x[1] <= 1200
 R2: x[2] <= 1200
 R3: x[3] <= 1200
 R4: x[4] <= 1200
 R5: x[5] <= 1200
 R6: x[6] <= 1200
 R7: x[7] <= 1200
 R8: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] <= 10000
 R9: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] >= 9500
Bounds
 Constant = 1
End
