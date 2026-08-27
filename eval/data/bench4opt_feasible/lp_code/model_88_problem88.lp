\ Model Capital Budgeting Problem with Grid Stability Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  5.7529516242358923e+08 x[0] + 3.0433280930953676e+08 x[1]
   + 5.9167093733814228e+08 x[2] + 4.1851589174208438e+08 x[3]
   + 9.8614053963014555e+08 x[4]
Subject To
 BudgetConstraint: x[0] + x[1] + x[2] + x[3] + x[4] <= 1e+06
 CapacityConstraint_0: 1e+06 x[0] <= 1e+06
 CapacityConstraint_1: 1e+06 x[1] <= 1e+06
 CapacityConstraint_2: 1e+06 x[2] <= 1e+06
 CapacityConstraint_3: 1e+06 x[3] <= 1e+06
 CapacityConstraint_4: 1e+06 x[4] <= 1e+06
 MinGenerationConstraint_0: - 1e+06 x[0] <= 0
 MaxGenerationConstraint_0: 1e+06 x[0] <= 1e+06
 MinGenerationConstraint_1: - 1e+06 x[1] <= 0
 MaxGenerationConstraint_1: 1e+06 x[1] <= 1e+06
 MinGenerationConstraint_2: - 1e+06 x[2] <= 0
 MaxGenerationConstraint_2: 1e+06 x[2] <= 1e+06
 MinGenerationConstraint_3: - 1e+06 x[3] <= 0
 MaxGenerationConstraint_3: 1e+06 x[3] <= 1e+06
 MinGenerationConstraint_4: - 1e+06 x[4] <= 0
 MaxGenerationConstraint_4: 1e+06 x[4] <= 1e+06
 GridStabilityRequirement: 1e+06 x[0] + 1e+06 x[1] + 1e+06 x[2]
   + 1e+06 x[3] + 1e+06 x[4] >= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
