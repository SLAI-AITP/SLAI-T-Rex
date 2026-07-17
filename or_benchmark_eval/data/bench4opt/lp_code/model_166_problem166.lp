\ Model Production Planning Problem with Energy Consumption Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  345.3775830100178 production_rates[0]
   + 454.9537468072595 production_rates[1]
   + 304.7309619295598 production_rates[2]
   + 485.2244789387723 production_rates[3]
   + 453.2007336607596 production_rates[4]
Subject To
 R0: 4.514827813059184 production_rates[0]
   + 2.011487361319463 production_rates[1]
   + 9.391204130515247 production_rates[2]
   + 9.509840696557694 production_rates[3]
   + 1.287773386800476 production_rates[4] <= 1107.652699211529
 R1: 2.272221624702077 production_rates[0]
   + 0.3346346812846699 production_rates[1]
   + 6.600394633786923 production_rates[2]
   + 6.417020563078838 production_rates[3]
   + 8.578764160621086 production_rates[4] <= 1879.288835282463
 R2: 2.904640489025201 production_rates[0]
   + 1.010218697511635 production_rates[1]
   + 8.37104545717362 production_rates[2]
   + 7.109943547712072 production_rates[3]
   + 1.386081806734118 production_rates[4] <= 1935.165562998473
 R3: 72.60882702904694 production_rates[0]
   + 32.8002164997993 production_rates[1]
   + 52.0260019558892 production_rates[2]
   + 94.25638539205697 production_rates[3]
   + 67.490245705917 production_rates[4] <= 9304.258517602282
Bounds
End
