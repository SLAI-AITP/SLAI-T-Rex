\ Model CuttingStock
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  y
Subject To
 Demand_0: x[0] >= 14.49503657329202
 Demand_1: x[1] >= 32.38912339818125
 Demand_2: x[2] >= 46.56708619410414
 Demand_3: x[3] >= 58.71631773789142
 Demand_4: x[4] >= 68.53359233636255
 Demand_5: x[5] >= 59.07420890597445
 WidthConstraint: 474.3469927348574 x[0] + 435.2271314964912 x[1]
   + 319.8572706850389 x[2] + 435.2626662970513 x[3]
   + 169.5042439329162 x[4] + 338.7802280963542 x[5] - 3164.36765238127 y
   <= 0
Bounds
Generals
 x[0] x[1] x[2] x[3] x[4] x[5] y
End
