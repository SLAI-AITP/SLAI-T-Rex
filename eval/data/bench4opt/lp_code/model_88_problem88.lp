\ Model Capital Budgeting Problem with Grid Stability Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  69.2713940669 x[0] + 36.8234164322 x[1] + 44.2344661773 x[2]
   + 11.4804502062 x[3] + 44.1319761671 x[4]
Subject To
 BudgetConstraint: 41.0981983994 x[0] + 96.65376941229999 x[1]
   + 18.0154448772 x[2] + 94.8138166233 x[3] + 46.5495932875 x[4]
   <= 203.2226590765
 CapacityConstraint_0: 8.7860616608 x[0] <= 8.5615583698
 CapacityConstraint_1: 87.1819172033 x[1] <= 63.6892420299
 CapacityConstraint_2: 74.83467782 x[2] <= 53.5298772565
 CapacityConstraint_3: 22.9356338305 x[3] <= 20.1964669204
 CapacityConstraint_4: 78.1326689917 x[4] <= 77.8465947031
 MinGenerationConstraint_0: 26.9728121 x[0] <= 0
 MaxGenerationConstraint_0: 8.7860616608 x[0] <= 26.3408539782
 MinGenerationConstraint_1: 8.682194347399999 x[1] <= 0
 MaxGenerationConstraint_1: 87.1819172033 x[1] <= 18.1764032218
 MinGenerationConstraint_2: - 9.9358231792 x[2] <= 0
 MaxGenerationConstraint_2: 74.83467782 x[2] <= 16.1341091116
 MinGenerationConstraint_3: - 7.314340066799998 x[3] <= 0
 MaxGenerationConstraint_3: 22.9356338305 x[3] <= 76.15608562449999
 MinGenerationConstraint_4: - 66.9550207417 x[4] <= 0
 MaxGenerationConstraint_4: 78.1326689917 x[4] <= 8.459974084300001
 GridStabilityRequirement: 8.7860616608 x[0] + 87.1819172033 x[1]
   + 74.83467782 x[2] + 22.9356338305 x[3] + 78.1326689917 x[4]
   >= 475.889609587
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
