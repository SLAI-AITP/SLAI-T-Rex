\ Model ContainerUtilization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  n_A
Subject To
 Balance: 30137 x_small - 41530 n_A - 50414 n_B = 0
Bounds
 x_small >= 1
Generals
 x_small n_A n_B
End
