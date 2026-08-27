\ Model Advanced Production Planning Problem with Energy Consumption
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  10499.3155847577 flow_rate[0] + 91893.21512654997 flow_rate[1]
   + 30739.59255101331 flow_rate[2] + 45584.99718401091 flow_rate[3]
   + 57919.07093720665 flow_rate[4] + 21792.39988065238 flow_rate[5]
   + 104833.717434707 flow_rate[6] + 106583.3639302501 flow_rate[7]
   + 66467.14669478447 flow_rate[8] + 87339.99968836812 flow_rate[9]
Subject To
 capacity_constraint_0: flow_rate[0] <= 9.8679534057664201e+06
 capacity_constraint_1: flow_rate[1] <= 5.0803321585101523e+06
 capacity_constraint_2: flow_rate[2] <= 4.6475179106415026e+06
 capacity_constraint_3: flow_rate[3] <= 6.2683778092017639e+06
 capacity_constraint_4: flow_rate[4] <= 2.0890059416853008e+06
 capacity_constraint_5: flow_rate[5] <= 540159.5025155891
 capacity_constraint_6: flow_rate[6] <= 1.1356644324076436e+06
 capacity_constraint_7: flow_rate[7] <= 1.5006819956420732e+06
 capacity_constraint_8: flow_rate[8] <= 8.4393791284118034e+06
 capacity_constraint_9: flow_rate[9] <= 9.1157288730843682e+06
 quality_standard_constraint: 127765.5267861043 flow_rate[0]
   + 143944.7733773729 flow_rate[1] + 271678.7978378219 flow_rate[2]
   + 450968.7118799931 flow_rate[3] + 240014.505163964 flow_rate[4]
   + 175374.3870327431 flow_rate[5] + 691981.8960918753 flow_rate[6]
   + 596675.6987483833 flow_rate[7] + 717481.5568356793 flow_rate[8]
   + 612437.7678509635 flow_rate[9] >= 4.9801732296606518e+07
 energy_usage_constraint: 1605.998339390194 flow_rate[0]
   + 9227.503664200938 flow_rate[1] + 549.7358671736697 flow_rate[2]
   + 3648.288760094304 flow_rate[3] + 2103.801630604041 flow_rate[4]
   + 2521.303979378963 flow_rate[5] + 6172.025311093666 flow_rate[6]
   + 8901.895137552143 flow_rate[7] + 4536.944801082773 flow_rate[8]
   + 1053.166784769033 flow_rate[9] <= 534041.2296307446
 environmental_impact_constraint: 537.8232790508249 flow_rate[0]
   + 158.887565066714 flow_rate[1] + 30.03615993690737 flow_rate[2]
   + 515.3888138901369 flow_rate[3] + 71.78863899546926 flow_rate[4]
   + 236.0645312121635 flow_rate[5] + 445.3760718637254 flow_rate[6]
   + 580.182014000761 flow_rate[7] + 419.6472078826475 flow_rate[8]
   + 747.3396667702175 flow_rate[9] <= 898474.3841242736
Bounds
End
