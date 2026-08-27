\ Model ContainerUtilization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  40808 n_A + 50904 n_B
Subject To
 Balance: 30062 x_small - 40808 n_A - 50904 n_B <= 0
Bounds
 x_small >= 1
Generals
 x_small n_A n_B
End
