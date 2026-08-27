\ Model Production Planning Problem with Storage
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  76.50705101789687 power_output[0] + 69.32400763182926 power_output[1]
   + 43.27901469696806 power_output[2] + 25.78594284886668 power_output[3]
   + 80.42749347589715 power_output[4] + 31.25762927265961 power_output[5]
   + 26.27282717781872 power_output[6] + 57.46499693081751 power_output[7]
   + 12.97814742368697 power_output[8] + 42.88248658573931 storage_level
Subject To
 demand: power_output[0] + power_output[1] + power_output[2]
   + power_output[3] + power_output[4] + power_output[5] + power_output[6]
   + power_output[7] + power_output[8] + storage_level
   >= 49283.22459606875
 capacity_0: power_output[0] <= 1691.760336332077
 capacity_1: power_output[1] <= 8053.865399216407
 capacity_2: power_output[2] <= 3268.45877016223
 capacity_3: power_output[3] <= 6542.179841762389
 capacity_4: power_output[4] <= 3476.954033380191
 capacity_5: power_output[5] <= 3312.601485310696
 capacity_6: power_output[6] <= 7672.862492531125
 capacity_7: power_output[7] <= 6434.393857207332
 capacity_8: power_output[8] <= 9191.183398112142
 emission_cap: 0.9706110703242065 power_output[0]
   + 0.6359896826784883 power_output[1] + 0.839551390996093 power_output[2]
   + 0.8538527976449836 power_output[3]
   + 0.7774851671921239 power_output[4]
   + 0.4716530888775812 power_output[5]
   + 0.4448502936523608 power_output[6]
   + 0.6946005597424991 power_output[7]
   + 0.7533991826610058 power_output[8] <= 2505.549348128666
 storage_capacity: storage_level <= 6199.816544422879
 initial_storage: storage_level >= 336.8099102279476
Bounds
End
