\ Model Capacitated Facility Location Problem with Uncertain Demand
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  49833.11 open_facility[0] + 29223.2 open_facility[1]
   + 27065.21 open_facility[2] + 33403.12 open_facility[3]
   + 38069.08 open_facility[4] + 122.61 service[0,0] + 122.61 service[0,1]
   + 122.61 service[0,2] + 122.61 service[0,3] + 122.61 service[0,4]
   + 54.02 service[1,0] + 54.02 service[1,1] + 54.02 service[1,2]
   + 54.02 service[1,3] + 54.02 service[1,4] + 132.18 service[2,0]
   + 132.18 service[2,1] + 132.18 service[2,2] + 132.18 service[2,3]
   + 132.18 service[2,4] + 63.98 service[3,0] + 63.98 service[3,1]
   + 63.98 service[3,2] + 63.98 service[3,3] + 63.98 service[3,4]
   + 187.63 service[4,0] + 187.63 service[4,1] + 187.63 service[4,2]
   + 187.63 service[4,3] + 187.63 service[4,4]
Subject To
 Demand_0: service[0,0] + service[1,0] + service[2,0] + service[3,0]
   + service[4,0] >= 909.21
 Demand_1: service[0,1] + service[1,1] + service[2,1] + service[3,1]
   + service[4,1] >= 575.79
 Demand_2: service[0,2] + service[1,2] + service[2,2] + service[3,2]
   + service[4,2] >= 920.48
 Demand_3: service[0,3] + service[1,3] + service[2,3] + service[3,3]
   + service[4,3] >= 991.96
 Demand_4: service[0,4] + service[1,4] + service[2,4] + service[3,4]
   + service[4,4] >= 955.51
 Capacity_0: - 1e+06 open_facility[0] + service[0,0] + service[0,1]
   + service[0,2] + service[0,3] + service[0,4] <= 0
 Capacity_1: - 1e+06 open_facility[1] + service[1,0] + service[1,1]
   + service[1,2] + service[1,3] + service[1,4] <= 0
 Capacity_2: - 1e+06 open_facility[2] + service[2,0] + service[2,1]
   + service[2,2] + service[2,3] + service[2,4] <= 0
 Capacity_3: - 1e+06 open_facility[3] + service[3,0] + service[3,1]
   + service[3,2] + service[3,3] + service[3,4] <= 0
 Capacity_4: - 1e+06 open_facility[4] + service[4,0] + service[4,1]
   + service[4,2] + service[4,3] + service[4,4] <= 0
Bounds
Binaries
 open_facility[0] open_facility[1] open_facility[2] open_facility[3]
 open_facility[4]
End
