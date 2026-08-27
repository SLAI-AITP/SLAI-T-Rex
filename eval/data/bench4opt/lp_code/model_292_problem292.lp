\ Model Capital Budgeting Problem with Environmental Impact Consideration
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  209.4674137765 x[0] + 238.1980439478 x[1] + 75.64691169690001 x[2]
   + 53.0663508439 x[3] + 121.0848317843 x[4]
Subject To
 Budget: 150.6331839392 x[0] + 118.2062803933 x[1] + 121.8104080956 x[2]
   + 216.8546578211 x[3] + 477.6358459183 x[4] <= 3832.0739428441
 Regulatory_0: x[0] <= 1
 Regulatory_1: x[1] <= 1
 Regulatory_2: x[2] <= 1
 Regulatory_3: x[3] <= 0
 Regulatory_4: x[4] <= 0
 EnvironmentalImpact: 20.0991356197 x[0] + 44.4689185851 x[1]
   + 14.563485629 x[2] + 12.6137374046 x[3] + 29.6698624116 x[4]
   <= 661.6169251767
 Dependency_0_1: x[0] - x[1] <= 0
 Dependency_0_2: x[0] - x[2] <= 0
 Dependency_0_3: x[0] - x[3] <= 0
 Dependency_1_2: x[1] - x[2] <= 0
 Dependency_1_3: x[1] - x[3] <= 0
 Dependency_1_4: x[1] - x[4] <= 0
 Dependency_2_4: x[2] - x[4] <= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
