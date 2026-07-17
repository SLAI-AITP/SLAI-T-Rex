\ Model ContainerUtilization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  x_small
Subject To
 Balance: 31496 x_small - 40499 n_A - 50924 n_B <= 0
Bounds
 x_small >= 1
Generals
 x_small n_A n_B
End
