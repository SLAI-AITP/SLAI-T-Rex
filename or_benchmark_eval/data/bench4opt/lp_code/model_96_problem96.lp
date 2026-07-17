\ Model ContainerUtilization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  n_A
Subject To
 Balance: 31354 x_small - 40520 n_A - 51981 n_B <= 0
Bounds
 x_small >= 1
Generals
 x_small n_A n_B
End
