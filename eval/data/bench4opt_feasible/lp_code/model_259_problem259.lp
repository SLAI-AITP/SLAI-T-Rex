\ Model Assignment Problem with Overtime Penalty
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[1,0] + x[1,1] + x[1,2]
   + x[1,3] + x[1,4] + x[2,0] + x[2,1] + x[2,2] + x[2,3] + x[2,4] + x[3,0]
   + x[3,1] + x[3,2] + x[3,3] + x[3,4] + x[4,0] + x[4,1] + x[4,2] + x[4,3]
   + x[4,4] + 141.01 y[0] + 102.6 y[1] + 84.19 y[2] + 157.11 y[3]
   + 164.33 y[4]
Subject To
 coverage_0: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] >= 1
 coverage_1: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] >= 1
 coverage_2: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] >= 1
 coverage_3: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] >= 1
 coverage_4: x[0,4] + x[1,4] + x[2,4] + x[3,4] + x[4,4] >= 1
 assignment_0: x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] <= 1
 assignment_1: x[1,0] + x[1,1] + x[1,2] + x[1,3] + x[1,4] <= 1
 assignment_2: x[2,0] + x[2,1] + x[2,2] + x[2,3] + x[2,4] <= 1
 assignment_3: x[3,0] + x[3,1] + x[3,2] + x[3,3] + x[3,4] <= 1
 assignment_4: x[4,0] + x[4,1] + x[4,2] + x[4,3] + x[4,4] <= 1
 overtime_0: x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] - y[0] <= 100
 overtime_1: x[1,0] + x[1,1] + x[1,2] + x[1,3] + x[1,4] - y[1] <= 100
 overtime_2: x[2,0] + x[2,1] + x[2,2] + x[2,3] + x[2,4] - y[2] <= 100
 overtime_3: x[3,0] + x[3,1] + x[3,2] + x[3,3] + x[3,4] - y[3] <= 100
 overtime_4: x[4,0] + x[4,1] + x[4,2] + x[4,3] + x[4,4] - y[4] <= 100
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[0,3] x[0,4] x[1,0] x[1,1] x[1,2] x[1,3] x[1,4]
 x[2,0] x[2,1] x[2,2] x[2,3] x[2,4] x[3,0] x[3,1] x[3,2] x[3,3] x[3,4]
 x[4,0] x[4,1] x[4,2] x[4,3] x[4,4]
Generals
 y[0] y[1] y[2] y[3] y[4]
End
