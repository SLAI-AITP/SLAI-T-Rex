\ Model Production Planning Problem with Resource Allocation Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  599163.725524216 production[0] + 167968.7847805666 production[1]
   + 323284.2537803254 production[2] + 316894.2221179244 production[3]
   + 845149.8904770775 production[4] + 80007.73290378692 inventory[0]
   + 27218.7514653491 inventory[1] + 69826.95445616417 inventory[2]
   + 36316.01546238182 inventory[3] + 53488.59028999999 inventory[4]
Subject To
 production_capacity_0: production[0] <= 412582.5099794028
 production_capacity_1: production[1] <= 920986.4701545029
 production_capacity_2: production[2] <= 163866.113344695
 production_capacity_3: production[3] <= 218658.3357483174
 production_capacity_4: production[4] <= 503879.5101709096
 demand_satisfaction_0: production[0] + inventory[0] >= 35227.77382768058
 demand_satisfaction_1: production[1] + inventory[1] >= 14457.37950718757
 demand_satisfaction_2: production[2] + inventory[2] >= 13781.24967415077
 demand_satisfaction_3: production[3] + inventory[3] >= 28213.63422918407
 demand_satisfaction_4: production[4] + inventory[4] >= 77902.90370092848
 resource_allocation: 831.7419812323776 production[0]
   + 795.3116313007449 production[1] + 559.0965591953317 production[2]
   + 689.9631762771663 production[3] + 952.5427185753521 production[4]
   <= 8.4622367993417047e+06
Bounds
End
