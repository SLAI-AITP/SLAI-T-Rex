\ Model Blending Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  991.4565553276492 x[0] + 375.3839019060086 x[1] + 567.5577478898763 x[2]
   + 813.1982680520919 x[3] + 17.89424913108536 x[4]
Subject To
 availability_0: x[0] <= 480.1264293423527
 availability_1: x[1] <= 881.3987240752231
 availability_2: x[2] <= 249.949154212278
 availability_3: x[3] <= 214.1161261043733
 availability_4: x[4] <= 144.8489957294794
 demand: 2.55121418305337 x[0] + 3.486627506269281 x[1]
   + 6.884232171310924 x[2] + 7.981931495878626 x[3]
   + 5.690425410261613 x[4] >= 5106.254557554944
Bounds
End
