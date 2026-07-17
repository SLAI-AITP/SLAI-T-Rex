\ Model Capital Budgeting Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  30.3296356637 x[0] + 45.1730502059 x[1] + 44.5610773771 x[2]
   + 21.0498961129 x[3] + 41.6002472801 x[4]
Subject To
 BudgetConstraint: 19.7036694891 x[0] + 8.655963591100001 x[1]
   + 13.2689761532 x[2] + 10.7537901987 x[3] + 18.828022249 x[4]
   <= 91.73619727880001
 Dependency_1_2: x[1] - x[2] <= 0
 Dependency_1_3: x[1] - x[3] <= 0
 Dependency_3_4: x[3] - x[4] <= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
