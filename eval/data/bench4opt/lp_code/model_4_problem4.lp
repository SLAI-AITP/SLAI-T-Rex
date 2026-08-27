\ Model Capital Budgeting Problem with Risk Management
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  5.5751580912 x[0] + 98.4096662554 x[1] + 61.516123333 x[2]
   + 77.08044456099999 x[3] + 46.0862006696 x[4]
Subject To
 budget: 88.727553996 x[0] + 57.9914190415 x[1] + 72.1162456344 x[2]
   + 39.0138682719 x[3] + 40.5656334248 x[4] <= 78.04781582000001
 risk_threshold: 69.074618895 x[0] + 89.37332279490001 x[1]
   + 86.1836159768 x[2] + 88.6610553455 x[3] + 78.06362027 x[4]
   <= 113.326362722
 dependency_0_1: - x[0] + x[1] <= 0
 dependency_1_0: x[0] - x[1] <= 0
 dependency_1_3: - x[1] + x[3] <= 0
 dependency_1_4: - x[1] + x[4] <= 0
 dependency_2_0: x[0] - x[2] <= 0
 dependency_2_1: x[1] - x[2] <= 0
 dependency_2_3: - x[2] + x[3] <= 0
 dependency_3_0: x[0] - x[3] <= 0
 dependency_3_1: x[1] - x[3] <= 0
 dependency_3_2: x[2] - x[3] <= 0
 dependency_4_0: x[0] - x[4] <= 0
 dependency_4_1: x[1] - x[4] <= 0
 dependency_4_3: x[3] - x[4] <= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
