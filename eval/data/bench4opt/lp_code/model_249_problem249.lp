\ Model Production Planning Problem with Renewable Energy Integration
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  53.51247486412298 power_output[0] + 88.64911844675628 power_output[1]
   + 37.9322838023155 power_output[2] + 39.46041563821251 power_output[3]
   + 79.25332574890486 power_output[4] + 337.482883419551 penalty
Subject To
 demand: power_output[0] + power_output[1] + power_output[2]
   + power_output[3] + power_output[4] + renewable_output[0]
   + renewable_output[1] + renewable_output[2] + renewable_output[3]
   + renewable_output[4] + renewable_output[5] >= 9421.32042345578
 capacity_0: power_output[0] <= 5553.952233675219
 capacity_1: power_output[1] <= 8191.227505683665
 capacity_2: power_output[2] <= 9144.602205607956
 capacity_3: power_output[3] <= 2386.699994742163
 capacity_4: power_output[4] <= 4061.043138151846
 renewable_generation_0: renewable_output[0] <= 1066.540284856708
 renewable_generation_1: renewable_output[1] <= 2678.639040394167
 renewable_generation_2: renewable_output[2] <= 398.9982054358372
 renewable_generation_3: renewable_output[3] <= 2321.922536276686
 renewable_generation_4: renewable_output[4] <= 2262.108703003781
 renewable_generation_5: renewable_output[5] <= 165.1927573600672
 emission_cap: 0.9840461264865209 power_output[0]
   + 0.5645319581705746 power_output[1] + 0.993448309745233 power_output[2]
   + 0.2846683990367768 power_output[3]
   + 0.1328972832369176 power_output[4] <= 4488.293941491071
 renewable_target: renewable_output[0] + renewable_output[1]
   + renewable_output[2] + renewable_output[3] + renewable_output[4]
   + renewable_output[5] + penalty >= 7826.444386014983
Bounds
End
