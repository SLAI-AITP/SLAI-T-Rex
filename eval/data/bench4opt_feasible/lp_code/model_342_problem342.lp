\ Model Capacitated Facility Location Problem with Latency Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  12900.22 y[0] + 12666 y[1] + 6623.25 y[2] + 48.8 x[0,0] + 65.41 x[0,1]
   + 98.87 x[0,2] + 60.7 x[0,3] + 57.29 x[1,0] + 23.4 x[1,1] + 19.6 x[1,2]
   + 60.38 x[1,3] + 67.31 x[2,0] + 39.59 x[2,1] + 82.55 x[2,2]
   + 44.34 x[2,3]
Subject To
 demand_0: x[0,0] + x[1,0] + x[2,0] >= 1
 demand_1: x[0,1] + x[1,1] + x[2,1] >= 1
 demand_2: x[0,2] + x[1,2] + x[2,2] >= 1
 demand_3: x[0,3] + x[1,3] + x[2,3] >= 1
 capacity_0: - 1e+06 y[0] + x[0,0] + x[0,1] + x[0,2] + x[0,3] <= 0
 capacity_1: - 1e+06 y[1] + x[1,0] + x[1,1] + x[1,2] + x[1,3] <= 0
 capacity_2: - 1e+06 y[2] + x[2,0] + x[2,1] + x[2,2] + x[2,3] <= 0
 latency_0_0: - 999967.22 x[0,0] <= 0
 latency_0_1: - 999987.33 x[0,1] <= 0
 latency_0_2: - 999978.7 x[0,2] <= 0
 latency_0_3: - 999957.74 x[0,3] <= 0
 latency_1_0: - 999973.95 x[1,0] <= 0
 latency_1_1: - 999998.33 x[1,1] <= 0
 latency_1_2: - 999991.04 x[1,2] <= 0
 latency_1_3: - 999967.9 x[1,3] <= 0
 latency_2_0: - 999996.95 x[2,0] <= 0
 latency_2_1: - 999986.99 x[2,1] <= 0
 latency_2_2: - 999953.14 x[2,2] <= 0
 latency_2_3: - 999955.42 x[2,3] <= 0
Bounds
Binaries
 y[0] y[1] y[2]
End
