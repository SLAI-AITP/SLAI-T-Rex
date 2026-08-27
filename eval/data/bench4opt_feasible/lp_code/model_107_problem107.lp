\ Model Assignment Problem with Preferences and Workload Balancing
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 4.8575125817 x[0,0] + 2.9744363691 x[0,1] - 1.7089960709 x[0,2]
   - 3.5755609146 x[1,0] + 4.0038662767 x[1,1] + 1.799200612999999 x[1,2]
   + 1.4090438014 x[2,0] - 6.6407473565 x[2,1] + 3.5443477793 x[2,2]
   + 7.5030590346 x[3,0] + 4.4444267566 x[3,1] + 3.2069873064 x[3,2]
   - 6.355984876099999 x[4,0] - 7.2056412348 x[4,1]
   - 8.284202132600001 x[4,2]
Subject To
 coverage_0: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] >= 2
 coverage_1: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] >= 1
 coverage_2: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] >= 1
 assignment_0: x[0,0] + x[0,1] + x[0,2] <= 1
 assignment_1: x[1,0] + x[1,1] + x[1,2] <= 1
 assignment_2: x[2,0] + x[2,1] + x[2,2] <= 1
 assignment_3: x[3,0] + x[3,1] + x[3,2] <= 1
 assignment_4: x[4,0] + x[4,1] + x[4,2] <= 1
 workload_0: x[0,0] + x[0,1] + x[0,2] <= 2
 workload_1: x[1,0] + x[1,1] + x[1,2] <= 1
 workload_2: x[2,0] + x[2,1] + x[2,2] <= 1
 workload_3: x[3,0] + x[3,1] + x[3,2] <= 1
 workload_4: x[4,0] + x[4,1] + x[4,2] <= 1
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[1,0] x[1,1] x[1,2] x[2,0] x[2,1] x[2,2] x[3,0]
 x[3,1] x[3,2] x[4,0] x[4,1] x[4,2]
End
