\ Model Cutting Stock Problem with Minimum Waste Constraint
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9]
Subject To
 R0: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9]
   <= 10000
 R1: x[0] >= 1
 R2: x[1] >= 1
 R3: x[2] >= 1
 R4: x[3] >= 1
 R5: x[4] >= 1
 R6: x[5] >= 1
 R7: x[6] >= 1
 R8: x[7] >= 1
 R9: x[8] >= 1
 R10: x[9] >= 1
 R11: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9]
   >= 9500
Bounds
End
