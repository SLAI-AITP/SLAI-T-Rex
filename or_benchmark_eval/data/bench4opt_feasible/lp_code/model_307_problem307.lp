\ Model Advanced Production Planning Problem with Chemical Usage Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  228.4561307191432 flow_rate[0] + 476.280357235978 flow_rate[1]
   + 489.5636387452182 flow_rate[2] + 891.5839554481686 flow_rate[3]
   + 153.4918592423462 flow_rate[4] + 928.608245278442 flow_rate[5]
Subject To
 capacity_constraint_0: flow_rate[0] <= 254.7458926597784
 capacity_constraint_1: flow_rate[1] <= 224.2986367498955
 capacity_constraint_2: flow_rate[2] <= 111.3820765613249
 capacity_constraint_3: flow_rate[3] <= 373.9195472551116
 capacity_constraint_4: flow_rate[4] <= 334.4095026058374
 capacity_constraint_5: flow_rate[5] <= 489.000151093025
 quality_standard_constraint: 78.11855261700714 flow_rate[0]
   + 33.28709024601 flow_rate[1] + 10.70834924634335 flow_rate[2]
   + 74.03703873490505 flow_rate[3] + 87.11874399181922 flow_rate[4]
   + 38.17770639989636 flow_rate[5] >= 281.989801692234
 chemical_usage_constraint: 1.359396470470253 flow_rate[0]
   + 6.952884242160557 flow_rate[1] + 0.6905520243792319 flow_rate[2]
   + 1.451364843547208 flow_rate[3] + 8.531348528778475 flow_rate[4]
   + 4.639931357208012 flow_rate[5] <= 127.0405925225676
Bounds
End
