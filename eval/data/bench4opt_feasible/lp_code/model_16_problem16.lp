\ Model Production Planning Problem with Resource Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  8824.966079324404 production[0] + 7335.887099754904 production[1]
   + 4493.633194689824 production[2] + 4416.344061288057 production[3]
   + 7790.54845846001 production[4] + 8824.966079324404 production[5]
Subject To
 demand_0: production[0] >= 1
 demand_1: production[1] >= 1
 demand_2: production[2] >= 1
 demand_3: production[3] >= 1
 demand_4: production[4] >= 1
 demand_5: production[5] >= 1
 storage_0: production[0] <= 1e+06
 storage_1: production[1] <= 1e+06
 storage_2: production[2] <= 1e+06
 storage_3: production[3] <= 1e+06
 storage_4: production[4] <= 1e+06
 storage_5: production[5] <= 1e+06
 shelf_life_0: production[0] <= 1e+06
 shelf_life_1: production[1] <= 1e+06
 shelf_life_2: production[2] <= 1e+06
 shelf_life_3: production[3] <= 1e+06
 shelf_life_4: production[4] <= 1e+06
 shelf_life_5: production[5] <= 1e+06
 resource_0: 98.98460132946798 production[0]
   + 42.9270329241517 production[1] + 45.71630928020895 production[2]
   + 96.79123102638343 production[3] + 62.8664020420575 production[4]
   + 98.98460132946798 production[5] <= 1e+06
 resource_1: 67.70716854545036 production[0]
   + 69.90016202448312 production[1] + 25.46662318756779 production[2]
   + 71.24555660757039 production[3] + 84.67765300808365 production[4]
   + 67.70716854545036 production[5] <= 1e+06
 resource_2: 52.43997232007282 production[0]
   + 47.81197370281808 production[1] + 13.97250916770933 production[2]
   + 71.25410651980721 production[3] + 7.087276707301998 production[4]
   + 52.43997232007282 production[5] <= 1e+06
 resource_3: 98.98460132946798 production[0]
   + 42.9270329241517 production[1] + 45.71630928020895 production[2]
   + 96.79123102638343 production[3] + 62.8664020420575 production[4]
   + 98.98460132946798 production[5] <= 1e+06
Bounds
End
