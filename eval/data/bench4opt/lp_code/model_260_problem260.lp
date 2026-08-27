\ Model Production Planning Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  90.17187083309486 power_output[0] + 23.62525748366189 power_output[1]
   + 59.47130396806027 power_output[2] + 50.69942735782108 power_output[3]
   + 45.13285517278387 power_output[4] + 42.5350548063373 power_output[5]
   + 86.39908975609215 power_output[6] + 87.37349594377028 power_output[7]
   + 32.13506783574798 power_output[8] + 35.68868536217759 power_output[9]
Subject To
 demand: power_output[0] + power_output[1] + power_output[2]
   + power_output[3] + power_output[4] + power_output[5] + power_output[6]
   + power_output[7] + power_output[8] + power_output[9]
   >= 13838.50690492513
 capacity_0: power_output[0] <= 8144.791193109833
 capacity_1: power_output[1] <= 9771.86666536823
 capacity_2: power_output[2] <= 7456.750411413353
 capacity_3: power_output[3] <= 9727.068530032371
 capacity_4: power_output[4] <= 4541.690382662764
 capacity_5: power_output[5] <= 3398.950513641398
 capacity_6: power_output[6] <= 8201.805272872085
 capacity_7: power_output[7] <= 5845.887873379701
 capacity_8: power_output[8] <= 7201.686523362028
 capacity_9: power_output[9] <= 3977.720301710462
 emission_cap: 0.7482573605272502 power_output[0]
   + 0.9668856972674781 power_output[1]
   + 0.1342471987913477 power_output[2]
   + 0.8763846308875304 power_output[3]
   + 0.3364399284431758 power_output[4]
   + 0.7875020111655662 power_output[5]
   + 0.1905809006959366 power_output[6]
   + 0.8934035016112739 power_output[7]
   + 0.5683965931245731 power_output[8]
   + 0.7415561261842285 power_output[9] <= 1378.852549500021
Bounds
End
